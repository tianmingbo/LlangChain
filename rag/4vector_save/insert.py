import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient

# 1. 加载 PDF 文档
# PyMuPDFLoader 对大文件（如 300 页）的解析速度非常快
file_path = "./rabbit.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()
print(f"文档加载完成，共 {len(docs)} 页")

# 2. 文本分段 (Chunking)
# 300 页的书籍必须分段，否则无法有效检索且会超过模型 Token 限制
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True
)
chunks = text_splitter.split_documents(docs)
print(f"切分完成，生成了 {len(chunks)} 个文本块")

# 3. 初始化 Embedding 模型
# 这里使用 HuggingFace 的本地模型，适合 Intel Mac 运行
# 如果有 API Key，也可以换成 OpenAIEmbeddings()
embed_model = HuggingFaceEmbeddings(
    model_name=os.path.expanduser("~/models/bge-base-zh-v1.5")
)

# 计算嵌入向量
embeddings = embed_model.embed_documents([chunk.page_content for chunk in chunks])
print("计算嵌入向量完成!")

# 转换数据格式
data = [
    {
        "vector": embedding,
        "text": chunk.page_content,
        "metadata": chunk.metadata,
    }
    for chunk, embedding in zip(chunks, embeddings)
]

# 实例化向量数据库客户端
client = MilvusClient(
    uri="./milvus_demo.db",  # 数据存储在本地当前目录下
)

# 插入实体
res = client.insert(collection_name="demo_collection", data=data)

print("插入成功的 ID 列表：", res['ids'])
print("插入数量：", res['insert_count'])
