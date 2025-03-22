from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)
from langchain_ollama import ChatOllama

# 创建一个消息列表，包含不同类型的消息：
# - SystemMessage: 系统指令消息
# - HumanMessage: 用户输入的消息
# - AIMessage: AI助手回复的消息
# 每个消息都有唯一ID和可选的名称属性
messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

if __name__ == "__main__":
    # 使用filter_messages函数过滤消息列表
    # 这里设置include_types="human"表示只保留类型为HumanMessage的消息
    res1 = filter_messages(messages, include_types="human")
    print(res1)

    # 输出结果说明:
    # 过滤后只保留了两条HumanMessage类型的消息
    # res1 =
    # [
    #     HumanMessage(
    #         content="example input",
    #         additional_kwargs={},
    #         response_metadata={},
    #         name="example_user",
    #         id="2",
    #     ),
    #     HumanMessage(
    #         content="real input",
    #         additional_kwargs={},
    #         response_metadata={},
    #         name="bob",
    #         id="4",
    #     ),
    # ]

    res2 = filter_messages(
        messages, exclude_names=["example_user", "example_assistant"]
    )
    print(res2)

    # res2
    # messages = [
    #     SystemMessage(
    #         content='you are a good assistant',
    #         additional_kwargs={},
    #         response_metadata={},
    #         id='1'
    #     ),
    #     HumanMessage(
    #         content='real input',
    #         additional_kwargs={},
    #         response_metadata={},
    #         name='bob',
    #         id='4'
    #     ),
    #     AIMessage(
    #         content='real output',
    #         additional_kwargs={},
    #         response_metadata={},
    #         name='alice',
    #         id='5'
    #     )
    # ]

    res3 = filter_messages(
        messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"]
    )
    print(res3)

    model = ChatOllama(model="qwen2.5:32b", base_url="http://localhost:11434")

    # filter_messages can be chained with other message processors
    chain = filter_messages(exclude_names=["example_user", "example_assistant"]) | model
    res4 = chain.invoke(messages)
    print(res4)
