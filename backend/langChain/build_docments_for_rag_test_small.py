import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================= 配置区域 =================
# 1. 设置你下载的 source 文件夹路径 (请修改这里！)
DOCS_PATH = "./documents"

# 2. 指定我们要保存向量数据库的路径
DB_SAVE_PATH = "./database_faiss_pytorch_base"

# 3. 选定 Embedding 模型 (关键！这里选 BGE-M3 支持中英互搜)
# 第一次运行会自动下载模型 (约 500MB+)，请保持网络通畅
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"


# ===========================================

def load_documents(base_path):
    """
    只加载高质量的文件夹：user_guide 和 notes
    """
    documents = []
    # 我们只关心这两个含金量最高的文件夹
    target_folders = ["user_guide", "notes"]

    print(f"🔍 开始扫描路径: {base_path}")

    for folder in target_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 警告: 找不到文件夹 {folder_path}，跳过。")
            continue

        print(f"📂 正在加载文件夹: {folder} ...")

        # 加载 Markdown 文件
        loader_md = DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
        docs_md = loader_md.load()

        # 加载 RST 文件 (简单作为文本加载)
        loader_rst = DirectoryLoader(folder_path, glob="**/*.rst", loader_cls=TextLoader, show_progress=True)
        docs_rst = loader_rst.load()

        # 给文档打标签，方便以后追踪来源
        for doc in docs_md + docs_rst:
            doc.metadata["source_type"] = "guide" if folder == "user_guide" else "technical_note"

        documents.extend(docs_md + docs_rst)
        print(f"   -> 文件夹 {folder} 加载了 {len(docs_md) + len(docs_rst)} 个文件")

    print(f"🎉 所有文档加载完毕，共 {len(documents)} 个文件。")
    return documents


def split_documents(documents):
    """
    把长文档切成小块，方便检索
    """
    print("✂️ 正在切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的大小
        chunk_overlap=200,  # 重叠部分，防止切断上下文
        separators=["\n\n", "\n", " ", ""]  # 优先按段落切分
    )
    splits = text_splitter.split_documents(documents)
    print(f"   -> 切分完成，共生成 {len(splits)} 个知识片段。")
    return splits


def build_vector_db(splits):
    """
    向量化并保存
    """
    print(f"🧠 正在加载 Embedding 模型 ({EMBEDDING_MODEL_NAME})... (初次运行需下载)")
    # 使用 BGE-M3，这是目前最强的开源多语言模型之一
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("🚀 正在构建向量数据库 (这一步可能需要几分钟)...")
    db = FAISS.from_documents(splits, embeddings)

    print(f"💾 正在保存到本地: {DB_SAVE_PATH}")
    db.save_local(DB_SAVE_PATH)
    print("✅ 知识库构建成功！")


if __name__ == "__main__":
    # 1. 加载
    raw_docs = load_documents(DOCS_PATH)

    if raw_docs:
        # 2. 切分
        chunks = split_documents(raw_docs)

        # 3. 建库
        build_vector_db(chunks)
    else:
        print("❌ 没有加载到任何文档，请检查路径是否正确。")