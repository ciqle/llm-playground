from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage

# 替换 OpenAI 为 Ollama
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages


# 定义状态类型，messages 字段将通过 add_messages 注解进行合并
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 创建状态图构建器
builder = StateGraph(State)

# 初始化 Ollama 模型，与 ch4_1_with_memory_simple.py 保持一致
model = ChatOllama(
    model="qwen2.5:32b",
    base_url="http://localhost:11434",
)


# 定义聊天机器人节点函数，处理消息并返回回复
def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


if __name__ == "__main__":

    # 添加节点和边缘，构建图
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # 添加持久化功能，使用 MemorySaver 保存对话状态
    graph = builder.compile(checkpointer=MemorySaver())

    # 配置会话线程，可以支持多个不同的对话
    thread1 = {"configurable": {"thread_id": "1"}}

    # 运行对话并持久化存储状态
    result_1 = graph.invoke(
        input={"messages": [HumanMessage("hi, my name is Jack!")]}, config=thread1
    )
    print(result_1)
    print()

    # 提问第二个问题，系统应能够记住之前的对话内容
    result_2 = graph.invoke(
        input={"messages": [HumanMessage("what is my name?")]}, config=thread1
    )
    print(result_2)
    print()

    # 获取并打印当前的对话状态
    print(graph.get_state(thread1))

# Output of result_1:
# {
#     "messages": [
#         HumanMessage(
#             content="hi, my name is Jack!",
#             additional_kwargs={},
#             response_metadata={},
#             id="39c34624-e80d-4fdc-ae46-67bfe968b791",
#         ),
#         AIMessage(
#             content="Hello Jack! It's nice to meet you. How can I assist you today?",
#             additional_kwargs={},
#             response_metadata={
#                 "model": "qwen2.5:32b",
#                 "created_at": "2025-03-19T17:25:28.65214Z",
#                 "done": True,
#                 "done_reason": "stop",
#                 "total_duration": 1338717375,
#                 "load_duration": 31170042,
#                 "prompt_eval_count": 36,
#                 "prompt_eval_duration": 473000000,
#                 "eval_count": 18,
#                 "eval_duration": 833000000,
#                 "message": Message(
#                     role="assistant", content="", images=None, tool_calls=None
#                 ),
#             },
#             id="run-70019dc0-1a1c-41ce-8450-6bed1630f114-0",
#             usage_metadata={
#                 "input_tokens": 36,
#                 "output_tokens": 18,
#                 "total_tokens": 54,
#             },
#         ),
#     ]
# }

# Output of result_2:
# {
#     "messages": [
#         HumanMessage(
#             content="hi, my name is Jack!",
#             additional_kwargs={},
#             response_metadata={},
#             id="39c34624-e80d-4fdc-ae46-67bfe968b791",
#         ),
#         AIMessage(
#             content="Hello Jack! It's nice to meet you. How can I assist you today?",
#             additional_kwargs={},
#             response_metadata={
#                 "model": "qwen2.5:32b",
#                 "created_at": "2025-03-19T17:25:28.65214Z",
#                 "done": True,
#                 "done_reason": "stop",
#                 "total_duration": 1338717375,
#                 "load_duration": 31170042,
#                 "prompt_eval_count": 36,
#                 "prompt_eval_duration": 473000000,
#                 "eval_count": 18,
#                 "eval_duration": 833000000,
#                 "message": {
#                     "role": "assistant",
#                     "content": "",
#                     "images": None,
#                     "tool_calls": None,
#                 },
#             },
#             id="run-70019dc0-1a1c-41ce-8450-6bed1630f114-0",
#             usage_metadata={
#                 "input_tokens": 36,
#                 "output_tokens": 18,
#                 "total_tokens": 54,
#             },
#         ),
#         HumanMessage(
#             content="what is my name?",
#             additional_kwargs={},
#             response_metadata={},
#             id="d330113d-d404-48c2-9ef0-0ebfff0fd135",
#         ),
#         AIMessage(
#             content="Your name is Jack.",
#             additional_kwargs={},
#             response_metadata={
#                 "model": "qwen2.5:32b",
#                 "created_at": "2025-03-19T17:25:29.170531Z",
#                 "done": True,
#                 "done_reason": "stop",
#                 "total_duration": 515241000,
#                 "load_duration": 7739583,
#                 "prompt_eval_count": 68,
#                 "prompt_eval_duration": 261000000,
#                 "eval_count": 6,
#                 "eval_duration": 245000000,
#                 "message": Message(
#                     role="assistant", content="", images=None, tool_calls=None
#                 ),
#             },
#             id="run-05fbcda1-84e5-48cd-a0f4-45a34fb0065d-0",
#             usage_metadata={"input_tokens": 68, "output_tokens": 6, "total_tokens": 74},
#         ),
#     ]
# }

# Output of graph.get_state(thread1):
# StateSnapshot(
#     values={
#         "messages": [
#             HumanMessage(
#                 content="hi, my name is Jack!",
#                 additional_kwargs={},
#                 response_metadata={},
#                 id="39c34624-e80d-4fdc-ae46-67bfe968b791",
#             ),
#             AIMessage(
#                 content="Hello Jack! It's nice to meet you. How can I assist you today?",
#                 additional_kwargs={},
#                 response_metadata={
#                     "model": "qwen2.5:32b",
#                     "created_at": "2025-03-19T17:25:28.65214Z",
#                     "done": True,
#                     "done_reason": "stop",
#                     "total_duration": 1338717375,
#                     "load_duration": 31170042,
#                     "prompt_eval_count": 36,
#                     "prompt_eval_duration": 473000000,
#                     "eval_count": 18,
#                     "eval_duration": 833000000,
#                     "message": {
#                         "role": "assistant",
#                         "content": "",
#                         "images": None,
#                         "tool_calls": None,
#                     },
#                 },
#                 id="run-70019dc0-1a1c-41ce-8450-6bed1630f114-0",
#                 usage_metadata={
#                     "input_tokens": 36,
#                     "output_tokens": 18,
#                     "total_tokens": 54,
#                 },
#             ),
#             HumanMessage(
#                 content="what is my name?",
#                 additional_kwargs={},
#                 response_metadata={},
#                 id="d330113d-d404-48c2-9ef0-0ebfff0fd135",
#             ),
#             AIMessage(
#                 content="Your name is Jack.",
#                 additional_kwargs={},
#                 response_metadata={
#                     "model": "qwen2.5:32b",
#                     "created_at": "2025-03-19T17:25:29.170531Z",
#                     "done": True,
#                     "done_reason": "stop",
#                     "total_duration": 515241000,
#                     "load_duration": 7739583,
#                     "prompt_eval_count": 68,
#                     "prompt_eval_duration": 261000000,
#                     "eval_count": 6,
#                     "eval_duration": 245000000,
#                     "message": {
#                         "role": "assistant",
#                         "content": "",
#                         "images": None,
#                         "tool_calls": None,
#                     },
#                 },
#                 id="run-05fbcda1-84e5-48cd-a0f4-45a34fb0065d-0",
#                 usage_metadata={
#                     "input_tokens": 68,
#                     "output_tokens": 6,
#                     "total_tokens": 74,
#                 },
#             ),
#         ]
#     },
#     next=(),
#     config={
#         "configurable": {
#             "thread_id": "1",
#             "checkpoint_ns": "",
#             "checkpoint_id": "1f004e72-74ba-6ca8-8004-b25e136bf3ad",
#         }
#     },
#     metadata={
#         "source": "loop",
#         "writes": {
#             "chatbot": {
#                 "messages": [
#                     AIMessage(
#                         content="Your name is Jack.",
#                         additional_kwargs={},
#                         response_metadata={
#                             "model": "qwen2.5:32b",
#                             "created_at": "2025-03-19T17:25:29.170531Z",
#                             "done": True,
#                             "done_reason": "stop",
#                             "total_duration": 515241000,
#                             "load_duration": 7739583,
#                             "prompt_eval_count": 68,
#                             "prompt_eval_duration": 261000000,
#                             "eval_count": 6,
#                             "eval_duration": 245000000,
#                             "message": {
#                                 "role": "assistant",
#                                 "content": "",
#                                 "images": None,
#                                 "tool_calls": None,
#                             },
#                         },
#                         id="run-05fbcda1-84e5-48cd-a0f4-45a34fb0065d-0",
#                         usage_metadata={
#                             "input_tokens": 68,
#                             "output_tokens": 6,
#                             "total_tokens": 74,
#                         },
#                     )
#                 ]
#             }
#         },
#         "thread_id": "1",
#         "step": 4,
#         "parents": {},
#     },
#     created_at="2025-03-19T17:25:29.171252+00:00",
#     parent_config={
#         "configurable": {
#             "thread_id": "1",
#             "checkpoint_ns": "",
#             "checkpoint_id": "1f004e72-6fcc-694e-8003-8a7ba7eaf111",
#         }
#     },
#     tasks=(),
# )
