import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

# useful to generate SQL query
model_low_temp = ChatOllama(
    base_url="http://localhost:11434", model="qwen2.5:32b", temperature=0.1
)
# useful to generate natural language outputs
model_high_temp = ChatOllama(
    base_url="http://localhost:11434", model="qwen2.5:32b", temperature=0.7
)


class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input 指的是用户提的问题
    user_query: str
    # output
    sql_query: str
    sql_explanation: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    sql_query: str
    sql_explanation: str


# 您是一名乐于助人的数据分析师，可根据用户的问题为其生成SQL查询
generate_prompt = SystemMessage(
    "You are a helpful data analyst who generates SQL queries for users based on their questions.\
    Just provide generated SQL queries and nothing else."
)


# 这是一个用于生成SQL查询的节点（Node）
# 该节点接收用户的问题，然后生成SQL查询
def generate_sql(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [generate_prompt, *state["messages"], user_message]
    logger.debug(f"messages: {messages}")
    res = model_low_temp.invoke(messages)
    return {
        "sql_query": res.content,
        "messages": [user_message, res],
    }


# 你是一名乐于助人的数据分析师，负责向用户解释SQL查询。
explain_prompt = SystemMessage(
    "You are a helpful data analyst who explains SQL queries to users."
)


# 这是一个用于解释SQL查询的节点（Node）
# 该节点接收用户的问题和上一步生成的SQL查询，然后解释SQL查询
def explain_sql(state: State) -> State:
    messages = [
        explain_prompt,
        # contains user's query and SQL query from prev step
        *state["messages"],
    ]
    res = model_high_temp.invoke(messages)
    return {
        "sql_explanation": res.content,  # 存储SQL查询的解释内容
        "messages": res,  # 将模型的回复添加到对话历史中
    }


if __name__ == "__main__":
    # 创建状态图构建器，定义输入输出类型
    builder = StateGraph(State, input=Input, output=Output)
    # 添加处理节点
    builder.add_node("generate_sql", generate_sql)  # 用于生成SQL查询的节点
    builder.add_node("explain_sql", explain_sql)  # 用于解释SQL查询的节点

    # 设置工作流程：START -> generate_sql -> explain_sql -> END
    builder.add_edge(START, "generate_sql")  # 工作流开始，先生成SQL
    builder.add_edge("generate_sql", "explain_sql")  # SQL生成后进行解释
    builder.add_edge("explain_sql", END)  # 解释完成后结束工作流

    # 编译图形，使其可执行
    graph = builder.compile()
    res = graph.invoke({"user_query": "What is the total sales for each product?"})
    print(res)
