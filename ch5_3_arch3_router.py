from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores.in_memory import InMemoryVectorStore

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434", model="nomic-embed-text"
)

# useful to generate SQL query
model_low_temp = ChatOllama(
    base_url="http://localhost:11434", model="qwen2.5:32b", temperature=0.1
)
# useful to generate natural language outputs
model_high_temp = ChatOllama(
    base_url="http://localhost:11434", model="qwen2.5:32b", temperature=0.7
)


# State里面保存了输入和输出的所有变量，以及状态改变中传递的所有中间变量
class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output: all the three vars below
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    documents: list[Document]
    answer: str


medical_records_store = InMemoryVectorStore.from_documents(
    documents=[], embedding=embeddings
)
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = InMemoryVectorStore.from_documents(
    documents=[], embedding=embeddings
)
insurance_faqs_retriever = insurance_faqs_store.as_retriever()

# 您需要决定将用户查询路由到哪个域。您有两个域供您选择：
#     - 记录：包含病人的医疗记录，如诊断、治疗和处方。
#     - 保险：包含有关保险保单、索赔和承保范围的常见问题。
#     仅输出域的名字。
router_prompt = SystemMessage(
    """You need to decide which domain to route the user query to. You have two 
        domains to choose from:
          - records: contains medical records of the patient, such as 
          diagnosis, treatment, and prescriptions.
          - insurance: contains frequently asked questions about insurance 
          policies, claims, and coverage.

        Output only the domain name.
    """
)


def router_node(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [router_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    return {
        "domain": res.content,
        # update conversation history
        "messages": [user_message, res],
    }


def pick_retriever(
    state: State,
) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    else:
        return "retrieve_insurance_faqs"


# user_query在后面会通过参数传进来
def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state["user_query"])
    return {
        "documents": documents,
    }


# 和上面一样，user_query在后面会通过参数传进来
def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(state["user_query"])
    return {
        "documents": documents,
    }


# 你是一个乐于助人的医疗聊天机器人，会根据病人的医疗记录回答问题，如诊断、治疗和患者的医疗记录，如诊断、治疗和处方。
medical_records_prompt = SystemMessage(
    """You are a helpful medical chatbot who answers questions based on the 
        patient's medical records, such as diagnosis, treatment, and 
        prescriptions."""
)

# 您是一个乐于助人的医疗保险聊天机器人，会回答有关保险政策、理赔和承保范围的常见问题。
insurance_faqs_prompt = SystemMessage(
    """You are a helpful medical insurance chatbot who answers frequently asked questions about insurance policies, claims, and coverage."""
)


def generate_answer(state: State) -> State:
    prompt = (
        medical_records_prompt
        if state["domain"] == "records"
        else insurance_faqs_prompt
    )
    messages = [
        prompt,
        *state["messages"],
        HumanMessage(f"Documents: {state['documents']}"),
    ]
    res = model_high_temp.invoke(messages)
    return {
        "answer": res.content,
        # update conversation history
        "messages": res,
    }


if __name__ == "__main__":
    # 创建一个StateGraph对象，指定状态类型、输入和输出
    # 这是LangGraph的核心，用于定义整个工作流程
    builder = StateGraph(State, input=Input, output=Output)

    # 添加所有的节点到图中
    # 每个节点都是一个处理步骤，接收状态并返回更新后的状态
    builder.add_node(
        node="router", action=router_node
    )  # 路由节点，决定查询应该被发送到哪个领域
    builder.add_node(
        node="retrieve_medical_records", action=retrieve_medical_records
    )  # 检索医疗记录的节点
    builder.add_node(
        node="retrieve_insurance_faqs", action=retrieve_insurance_faqs
    )  # 检索保险FAQ的节点
    builder.add_node(
        node="generate_answer", action=generate_answer
    )  # 生成最终回答的节点

    # 添加边来定义图的执行流程
    # START是特殊常量，表示工作流的起点
    builder.add_edge(start_key=START, end_key="router")  # 工作流始于路由器节点

    # 添加条件边，根据路由结果决定后续执行路径
    # pick_retriever函数会根据domain的值返回下一个节点名称
    builder.add_conditional_edges(source="router", path=pick_retriever)

    # 定义从检索节点到生成答案节点的边
    builder.add_edge(
        start_key="retrieve_medical_records", end_key="generate_answer"
    )  # 医疗记录检索完成后生成回答
    builder.add_edge(
        start_key="retrieve_insurance_faqs", end_key="generate_answer"
    )  # 保险FAQ检索完成后生成回答

    # 定义工作流的终点
    # END是特殊常量，表示工作流的终点
    builder.add_edge(start_key="generate_answer", end_key=END)  # 生成答案后结束工作流

    # 编译图，使其可执行
    # 这一步会验证图的结构并优化执行
    graph = builder.compile()

    # 创建输入并执行图
    # 这里我们测试一个关于COVID-19保险覆盖的问题
    input = {"user_query": "Am I covered for COVID-19 treatment?"}

    # 使用stream方法执行图并打印每一步的结果
    # stream方法会逐步返回每个节点执行后的状态
    for chunk in graph.stream(input):
        print(chunk)  # 打印每一步执行的结果，便于调试和观察

# 输出结果说明:
# 首先，路由器确定查询与"insurance"领域相关
# 然后，系统检索保险FAQ（这里没有实际文档，所以返回空列表）
# 最后，生成答案节点提供了关于COVID-19保险覆盖的详细回复

# Outputs:
# {
#     "router": {
#         "domain": "insurance",
#         "messages": [
#             HumanMessage(
#                 content="Am I covered for COVID-19 treatment?",
#                 additional_kwargs={},
#                 response_metadata={},
#                 id="7432a71b-58a2-4cad-93c2-36c2e5fa71a4",
#             ),
#             AIMessage(
#                 content="insurance",
#                 additional_kwargs={},
#                 response_metadata={
#                     "model": "qwen2.5:32b",
#                     "created_at": "2025-03-23T16:57:25.816254Z",
#                     "done": True,
#                     "done_reason": "stop",
#                     "total_duration": 8310361542,
#                     "load_duration": 650403500,
#                     "prompt_eval_count": 95,
#                     "prompt_eval_duration": 7601000000,
#                     "eval_count": 2,
#                     "eval_duration": 51000000,
#                     "message": Message(
#                         role="assistant", content="", images=None, tool_calls=None
#                     ),
#                 },
#                 id="run-d01fe7ac-b5cc-4dc7-ac2e-f31481e9e8ef-0",
#                 usage_metadata={
#                     "input_tokens": 95,
#                     "output_tokens": 2,
#                     "total_tokens": 97,
#                 },
#             ),
#         ],
#     }
# }
#
#
# {"retrieve_insurance_faqs": {"documents": []}}
#
#
# {
#     "generate_answer": {
#         "answer": "To clarify your coverage for COVID-19 treatment, it would be best to review the specifics of your medical insurance policy. Generally, many health insurance plans cover diagnostic testing and treatments related to COVID-19, but the extent of this coverage can vary based on your insurer and plan type.\n\nHere are a few key points you might want to check:\n\n1. **Diagnostic Testing**: Most plans now cover COVID-19 tests without cost-sharing when provided by in-network providers.\n2. **Treatment Costs**: If you need hospitalization or other medical treatments for COVID-19, these should be covered under your plan’s medical benefits.\n3. **Preventive Services**: Vaccinations are typically covered at no cost to you as preventive services.\n\nTo get precise information about your coverage:\n- Log in to your insurance provider's website and review your policy details.\n- Contact your insurer directly for a detailed explanation of what is included under your plan regarding COVID-19 treatments.\n- Check if there are any specific networks or providers you need to use for full coverage.\n\nIf you have any specific questions about your coverage, feel free to ask!",
#         "messages": AIMessage(
#             content="To clarify your coverage for COVID-19 treatment, it would be best to review the specifics of your medical insurance policy. Generally, many health insurance plans cover diagnostic testing and treatments related to COVID-19, but the extent of this coverage can vary based on your insurer and plan type.\n\nHere are a few key points you might want to check:\n\n1. **Diagnostic Testing**: Most plans now cover COVID-19 tests without cost-sharing when provided by in-network providers.\n2. **Treatment Costs**: If you need hospitalization or other medical treatments for COVID-19, these should be covered under your plan’s medical benefits.\n3. **Preventive Services**: Vaccinations are typically covered at no cost to you as preventive services.\n\nTo get precise information about your coverage:\n- Log in to your insurance provider's website and review your policy details.\n- Contact your insurer directly for a detailed explanation of what is included under your plan regarding COVID-19 treatments.\n- Check if there are any specific networks or providers you need to use for full coverage.\n\nIf you have any specific questions about your coverage, feel free to ask!",
#             additional_kwargs={},
#             response_metadata={
#                 "model": "qwen2.5: 32b",
#                 "created_at": "2025-03-23T16: 57: 37.860977Z",
#                 "done": True,
#                 "done_reason": "stop",
#                 "total_duration": 11937557750,
#                 "load_duration": 8861417,
#                 "prompt_eval_count": 59,
#                 "prompt_eval_duration": 379000000,
#                 "eval_count": 230,
#                 "eval_duration": 11547000000,
#                 "message": Message(
#                     role="assistant", content="", images=None, tool_calls=None
#                 ),
#             },
#             id="run-0ec06531-4688-4397-8c09-dd3ba8e307ca-0",
#             usage_metadata={
#                 "input_tokens": 59,
#                 "output_tokens": 230,
#                 "total_tokens": 289,
#             },
#         ),
#     }
# }
