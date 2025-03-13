from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""

if __name__ == "__main__":
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=50, chunk_overlap=0
    )
    python_docs = python_splitter.create_documents(
        texts=[PYTHON_CODE], metadatas=[{"source": "Python Code"}]
    )
    for doc in python_docs:
        print(doc)
