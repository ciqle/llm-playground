import ast
import os
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

model = ChatOllama(
    model="qwen2.5:32b", base_url="http://localhost:11434", temperature=0.1
)

# from_documnets直接创建向量存储
# 这里把所有tool的描述存储为文档，并且把tool的名称作为metadata
tools_retriever = InMemoryVectorStore.from_documents(
    documents=[
        Document(page_content=tool.description, metadata={"name": tool.name})
        for tool in tools
    ],
    embedding=embeddings,
).as_retriever()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}


def select_tools(state: State) -> State:
    query = state["messages"][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}


builder = StateGraph(State)
builder.add_node("select_tools", select_tools)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

if __name__ == "__main__":
    os.environ["HTTP_PROXY"] = "http://localhost:10086"
    os.environ["HTTPS_PROXY"] = "http://localhost:10086"
    input = {
        "messages": [HumanMessage("How old is Donald Trump when the UK's Queen died?")]
    }
    for chunk in graph.stream(input):
        print(chunk)
