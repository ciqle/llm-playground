from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    # Load the document
    loader = TextLoader("./resources/test.txt")
    doc = loader.load()  # doc is the full text

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    chunks = text_splitter.split_documents(doc)  # Type of chunks is List[Document]

    ## Generate embeddings for each chunk
    embedding_ollama_model = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
    )

    # Input: list[str]
    # Output: embeddings, type is List[List[float]]
    input_chunks = [
        chunk.page_content for chunk in chunks
    ]  # chunk.page_content is the text of the chunk
    embeddings = embedding_ollama_model.embed_documents(texts=input_chunks)
    print(embeddings[0])
