import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embedding_model = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
)

llm = ChatOllama(base_url="http://localhost:11434", model="qwen")

vectorstore = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

if __name__ == "__main__":
    # Load the document
    loader = TextLoader("./resources/test.txt")
    docs = loader.load()  # doc is the full text

    print("length of loaded docs: ", len(docs[0].page_content))
    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)

    # The rest of your code remains the same, starting from:
    prompt_text = "Summarize the following document:\n\n{doc}"

    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = (
        {"doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()
    )

    summaries = summarize_chain.batch(chunks)

    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key=id_key
    )

    doc_ids = [str(uuid.uuid4()) for _ in chunks]

    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)

    retriever.docstore.mset(list(zip(doc_ids, chunks)))
