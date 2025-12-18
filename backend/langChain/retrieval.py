import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ================= 配置区域 =================
# 1. 向量库路径（必须和 build_rag.py 里生成的一致）
DB_PATH = "./database_faiss_pytorch_base"

# 2. 模型名称（必须和构建时完全一致！）
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# 3. 指定模型保存/加载的本地路径（解决你的第二个问题）
# 这样下次运行就不会去联网下载，而是直接读这个文件夹
MODEL_CACHE_DIR = "./models"



def retrieval():
    print(f"🧠 正在加载模型: {EMBEDDING_MODEL_NAME} ...")

    # 强制使用 CPU，并指定本地缓存路径
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=MODEL_CACHE_DIR  # <--- 关键：指定本地保存路径
    )

    print(f"📂 正在加载向量数据库: {DB_PATH} ...")
    try:
        # allow_dangerous_deserialization=True 是必须的，因为我们要加载本地生成的 pickle 文件
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请检查 DB_PATH 是否正确，或者 build_rag.py 是否执行成功。")
        return

    # === 测试环节 ===
    while True:
        query = input("\n🔎 请输入关于 PyTorch 的问题 (输入 'q' 退出): ")
        if query.lower() == 'q':
            break

        print(f"   正在检索: '{query}' ...")

        # 搜索最相似的 3 个片段
        results = db.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(results):
            # score 越小越相似 (L2距离)
            print(f"\n--- [结果 {i + 1}] (相关度: {score:.4f}) ---")
            print(f"📄 来源文件: {doc.metadata.get('source', '未知')}")

            # 打印内容预览（去掉换行符，防止刷屏）
            content_preview = doc.page_content.replace("\n", " ")[:300]
            print(f"📝 内容摘要: {content_preview}...")


if __name__ == "__main__":
    retrieval()