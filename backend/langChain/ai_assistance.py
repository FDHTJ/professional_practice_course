import os

from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings

class StreamPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
# ================= 配置区域 =================
# 替换你的 Key
os.environ["OPENAI_API_KEY"] = "495336b5c3a44a85b7b97b64da809573.Hbr3XNqsR3KKNxWO"
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

DB_PATH = "./database_faiss_pytorch_base"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
MODEL_CACHE = "./models"


# ===========================================

def build_agent():
    print("🛠️ 正在组装 PyTorch 智能体...")

    # 1. 准备 RAG 工具 (你的知识库)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=MODEL_CACHE
    )
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    def search_pytorch_docs(query: str) -> str:
        docs_with_scores = db.similarity_search_with_score(query, k=5)

        results = []
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source_type", "unknown")
            entry = (
                f"📄 来源: {source}\n"
                f"🔍 相似度: {score:.4f}\n"
                f"------ 内容 ------\n"
                f"{doc.page_content}"
            )
            results.append(entry)

        return "\n\n==========================\n\n".join(results)
        # docs = db.similarity_search(query, k=5)
        # return "\n\n".join([d.page_content for d in docs])

    import torch


    def safe_python_exec(code: str) -> str:
        """执行 Python 代码并返回输出，专门用于 shape 检查。"""
        try:
            # 限制可用变量，避免危险操作
            local_env = {"torch": torch}
            code=code.replace("Observation","")
            exec(code, {}, local_env)
            return str(local_env.get("out", "执行成功"))
        except Exception as e:
            return f"Python 错误: {e}"

    # Python 代码执行工具
    python_tool = Tool(
        func=safe_python_exec,
        description="当你需要运行 Python 代码验证张量 shape 或运行 PyTorch 代码时使用此工具。输入必须是纯 Python/PyTorch 代码。",
        name="python_exec"
    )

    rag_tool = Tool(
        func=search_pytorch_docs,
        name="search_pytorch_docs",
        description="遇到PyTorch概念、API 用法或报错信息 或者其他pytorch相关的知识时，必须先用这个工具查阅官方文档。"
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True  # 让 Agent 看到多轮对话
    )
    # 4. 构造 Agent
    llm = ChatOpenAI(model="glm-4.5-flash",
                     temperature=0,streaming=True,
                     callbacks=[StreamPrintHandler()])

    def make_learning_plans(information:str):
        prompt=f'''你是一个学习计划制定专家,请根据下面所给出的信息,一步一步为用户制定出切实可行的学习计划:
                    信息:{information}
                指令：请重点关注用户的学习目标、时间期限等信息,而且要尽可能的通过对话历史或者询问以了解用户所处水平,为用户量身定制出合适的计划.
                并按照以下格式返回规划后的结果：
                1.总体目标(总体需要达到的水平以及所用时间及用户所处水平)
                2.分阶段计划
                    -阶段一(阶段预计用时):[阶段主题]
                        *学习内容:[具体学习内容列表]
                        *推荐资源:[书籍、网站、课程等]
                        *评估方式:[测试、练习、项目等]
                    -阶段二...
                3.评估和调整:[如何评估进度和调整计划]
                '''
        return llm.invoke(prompt).content
    planning_tool=Tool(
        func=make_learning_plans,
        name="make_learning_plans",
        description="当需要为用户制定学习计划时调用此工具帮助你为其制定更完善合理的学习计划",
    return_direct = True
    )

    agent = initialize_agent(
        tools=[rag_tool,python_tool,planning_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory
    )



    return agent


if __name__ == "__main__":
    agent = build_agent()
    print("\n✅ 全能 PyTorch 助手已就绪！")

    while True:
        user_input = input("\n🙋 请提问 (q退出): ")
        if user_input.lower() == 'q':
            break

        try:
            # Agent 开始自动规划和执行
            response = agent.invoke({"input": user_input})
            # print("\n🤖 最终回答:", response['output'])
        except Exception as e:
            print(f"❌ 出错了: {e}")