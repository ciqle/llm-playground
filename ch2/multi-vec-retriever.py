import json
import uuid

import psycopg
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# 设置PostgreSQL数据库连接字符串
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
# 设置向量集合名称
collection_name = "summaries"
# 初始化嵌入模型，使用本地运行的Ollama服务和nomic-embed-text模型生成文本嵌入向量
embedding_model = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
)

handler = StdOutCallbackHandler()
callback_manager = CallbackManager([handler])

# 初始化大语言模型，使用本地运行的Ollama服务和gemma3模型
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:12b",
    callback_manager=callback_manager,
)

# 创建PostgreSQL向量存储，用于存储和检索向量嵌入
vectorstore = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,  # 使用JSONB格式存储元数据，提高查询效率
)


# 创建PostgreSQL文档存储类，用于替代InMemoryStore
class PostgresDocStore:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        # 确保文档存储表存在
        with psycopg.connect(
            self.connection_string.replace("postgresql+psycopg://", "")
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_store (
                        key TEXT PRIMARY KEY,
                        document JSONB
                    )
                """
                )
            conn.commit()

    def mget(self, keys):
        # 批量获取文档
        if not keys:
            return []
        with psycopg.connect(
            self.connection_string.replace("postgresql+psycopg://", "")
        ) as conn:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(keys))
                query = f"SELECT key, document FROM document_store WHERE key IN ({placeholders})"
                cur.execute(query, keys)
                results = cur.fetchall()

                # 构建结果字典
                docs_dict = {}
                for key, doc_json in results:
                    # 将JSON转换回Document对象
                    doc_data = json.loads(doc_json)
                    doc = Document(
                        page_content=doc_data["page_content"],
                        metadata=doc_data.get("metadata", {}),
                    )
                    docs_dict[key] = doc

                # 按照输入keys的顺序返回文档
                return [docs_dict.get(key) for key in keys]

    def mset(self, key_doc_pairs):
        # 批量存储文档
        if not key_doc_pairs:
            return

        with psycopg.connect(
            self.connection_string.replace("postgresql+psycopg://", "")
        ) as conn:
            with conn.cursor() as cur:
                for key, doc in key_doc_pairs:
                    # 将Document对象转换为JSON
                    doc_dict = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    doc_json = json.dumps(doc_dict)

                    # 使用UPSERT操作
                    cur.execute(
                        "INSERT INTO document_store (key, document) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET document = %s",
                        (key, doc_json, doc_json),
                    )
            conn.commit()

    def delete(self, keys):
        # 删除文档
        if not keys:
            return
        with psycopg.connect(
            self.connection_string.replace("postgresql+psycopg://", "")
        ) as conn:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(keys))
                query = f"DELETE FROM document_store WHERE key IN ({placeholders})"
                cur.execute(query, keys)
            conn.commit()


if __name__ == "__main__":
    # 加载文本文档
    loader = TextLoader("./resources/test.txt")
    docs = loader.load()  # 加载完整文档

    print("length of loaded docs: ", len(docs[0].page_content))
    # 将文档分割成较小的块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)

    # 为每个文档块创建摘要的提示模板
    prompt_text = "Summarize the following document:\n\n{doc}"

    # 创建聊天提示模板
    prompt = ChatPromptTemplate.from_template(prompt_text)
    # 创建一个处理链：提取文档内容 -> 应用提示模板 -> 传入LLM -> 解析输出为字符串
    summarize_chain = (
        {"doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()
    )

    # 对所有文档块批量生成摘要
    summaries = summarize_chain.batch(chunks, callbacks=[handler])

    # 创建PostgreSQL文档存储，用于存储原始文档块
    store = PostgresDocStore(connection)
    # 定义文档ID的键名
    id_key = "doc_id"

    # 创建多向量检索器，将向量存储和文档存储关联起来
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key=id_key
    )

    # 为每个文档块生成唯一ID
    doc_ids = [str(uuid.uuid4()) for _ in chunks]

    # 创建摘要文档，每个文档包含摘要内容和对应的文档ID
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # 将摘要文档添加到向量存储中（会自动计算嵌入向量）
    retriever.vectorstore.add_documents(summary_docs)

    # 将原始文档块与其ID关联，存储到PostgreSQL文档存储中
    retriever.docstore.mset(list(zip(doc_ids, chunks)))
