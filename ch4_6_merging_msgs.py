from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

if __name__ == "__main__":
    # 创建一系列消息，包括系统指令、用户问题和AI回复
    messages = [
        SystemMessage("you're a good assistant."),  # 系统消息：你是一个好助手
        SystemMessage("you always respond with a joke."),  # 系统消息：你总是用笑话回应
        HumanMessage(
            [{"type": "text", "text": "i wonder why it's called langchain"}]
        ),  # 用户消息（结构化格式）
        HumanMessage("and who is harrison chasing anyway"),  # 用户消息（文本格式）
        AIMessage(
            """Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!"""
        ),  # AI回复第一条
        AIMessage(
            """Why, he's probably chasing after the last cup of coffee in the office!"""
        ),  # AI回复第二条
    ]

    # 使用merge_message_runs函数合并连续的相同类型消息
    res = merge_message_runs(messages)
    print(res)

# 输出结果:
# [
#     SystemMessage(
#         content="you're a good assistant.\nyou always respond with a joke.",  # 两条系统消息被合并，内容用换行符连接
#         additional_kwargs={},
#         response_metadata={},
#     ),
#     HumanMessage(
#         content=[
#             {"type": "text", "text": "i wonder why it's called langchain"},
#             "and who is harrison chasing anyway",  # 两条用户消息被合并为一个列表
#         ],
#         additional_kwargs={},
#         response_metadata={},
#     ),
#     AIMessage(
#         content='Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!\nWhy, he\'s probably chasing after the last cup of coffee in the office!',  # 两条AI消息被合并，内容用换行符连接
#         additional_kwargs={},
#         response_metadata={},
#     ),
# ]
