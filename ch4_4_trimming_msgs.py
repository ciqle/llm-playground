import os

from langchain_core.messages import trim_messages  # 用于裁剪消息历史的工具函数
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    os.environ["HTTP_PROXY"] = "http://localhost:10086"
    os.environ["HTTPS_PROXY"] = "http://localhost:10086"

    # 创建消息裁剪器
    # trim_messages 用于控制发送给模型的消息长度，防止超出模型的上下文窗口限制
    trimmer = trim_messages(
        max_tokens=60,  # 设置最大token数量为60
        strategy="last",  # 使用"最后"策略，保留最近的消息
        token_counter=ChatOllama(  # 使用ChatOllama作为token计数器
            model="qwen2.5:32b", base_url="http://localhost:11434"
        ),
        include_system=True,  # 包含系统消息
        allow_partial=False,  # 不允许部分消息（即消息要么完整保留，要么完全删除）
        start_on="human",  # 从人类消息开始，确保每轮对话的完整性
    )

    # 创建一个消息列表，模拟对话历史
    messages = [
        SystemMessage(content="you're a good assistant"),  # 系统指令
        HumanMessage(content="hi! I'm bob"),  # 用户消息
        AIMessage(content="hi!"),  # AI回复
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="what's 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    # 调用trimmer处理消息列表
    # 根据上面的配置，trimmer会保留最后几条消息，总token数不超过65
    res = trimmer.invoke(messages)
    print(res)  # 打印裁剪后的结果

# Output:
# [
#   HumanMessage(
#     content = "hi! I'm bob",
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   AIMessage(
#     content = 'hi!',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   HumanMessage(
#     content = 'I like vanilla ice cream',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   AIMessage(
#     content = 'nice',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   HumanMessage(
#     content = "what's 2 + 2",
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   AIMessage(
#     content = '4',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   HumanMessage(
#     content = 'thanks',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   AIMessage(
#     content = 'no problem!',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   HumanMessage(
#     content = 'having fun?',
#     additional_kwargs = {},
#     response_metadata = {}
#   ),
#   AIMessage(
#     content = 'yes!',
#     additional_kwargs = {},
#     response_metadata = {}
#   )
# ]
