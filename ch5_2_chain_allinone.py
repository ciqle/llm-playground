from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 初始化模型
# 低温度模型用于生成SQL查询（温度越低，输出越确定性）
# useful to generate SQL query
model_low_temp = ChatOllama(
    base_url="http://localhost:11434", model="qwen2.5:32b", temperature=0.1
)
# 高温度模型用于生成自然语言解释（温度越高，输出越多样化）
# useful to generate natural language outputs
model_high_temp = ChatOllama(
    base_url="http://localhost:11434", model="deepseek-r1:32b", temperature=0.7
)


# 定义状态类型
class State(TypedDict):
    # 用于跟踪对话历史
    messages: Annotated[list, add_messages]
    # 输入：用户查询
    user_query: str
    # 输出：SQL查询和解释
    sql_query: str
    sql_explanation: str


# 定义输入类型
class Input(TypedDict):
    user_query: str


# 定义输出类型
class Output(TypedDict):
    sql_query: str
    sql_explanation: str


# 创建系统提示
# 生成SQL的系统提示
generate_prompt = SystemMessage(
    "You are a helpful data analyst who generates SQL queries for users based on their questions."
)

# 解释SQL的系统提示
explain_prompt = SystemMessage(
    "You are a helpful data analyst who explains SQL queries to users."
)


# 定义第一个节点函数, 从一个state转化为另一个state
def generate_sql(state: State) -> State:
    # 回忆一下State由4个参数组成：
    # messages: Annotated[List], user_query: str, sql_query: str, sel_explanation: str

    # 将用户查询转换为消息对象
    user_message = HumanMessage(state["user_query"])
    # 组合系统提示和历史消息
    # generate_prompt是系统消息，state["messages"]是历史消息，用*解构
    messages = [generate_prompt, *state["messages"], user_message]
    # 调用低温度模型生成SQL查询
    res = model_low_temp.invoke(messages)
    # 这里返回一个新状态，更新了新的sql_query和新的messages
    return {
        "sql_query": res.content,
        # 更新对话历史
        "messages": [user_message, res],
    }


# 解释SQL查询的函数
def explain_sql(state: State) -> State:
    # 组合系统提示和历史消息（包含用户查询和前一步生成的SQL）
    messages = [
        explain_prompt,
        # 包含用户查询和前一步生成的SQL
        *state["messages"],
    ]
    # 调用高温度模型生成解释
    res = model_high_temp.invoke(messages)
    return {
        "sql_explanation": res.content,
        # 更新对话历史
        "messages": res,
    }


# 创建图（工作流）
# 初始化状态图，指定状态、输入和输出类型
builder = StateGraph(State, input=Input, output=Output)
# 添加节点
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
# 添加边：定义工作流程
builder.add_edge(START, "generate_sql")  # 开始 -> 生成SQL
builder.add_edge("generate_sql", "explain_sql")  # 生成SQL -> 解释SQL
builder.add_edge("explain_sql", END)  # 解释SQL -> 结束

# 编译图
graph = builder.compile()

# 示例用法
if __name__ == "__main__":
    # 调用图进行处理，输入用户查询
    result = graph.invoke({"user_query": "What is the total sales for each product?"})

    # 打印结果
    print("生成的SQL查询:")  # Generated SQL Query
    print(result["sql_query"])
    print("\n解释:")  # Explanation
    print(result["sql_explanation"])
