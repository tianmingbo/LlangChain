import os
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings

# 实例化向量数据库客户端
client = MilvusClient(
    uri="./milvus_demo.db",  # 数据存储在本地当前目录下
)

embed_model = HuggingFaceEmbeddings(
    model_name=os.path.expanduser("~/models/bge-base-zh-v1.5")
)


# ========== 检索 ==========
def retrieval(query, embed_model, client):
    """检索并返回上下文"""
    query_embedding = embed_model.embed_query(query)  # 查询嵌入
    context = client.search(
        collection_name="demo_collection",  # collection 名称
        data=[query_embedding],  # 搜索的向量
        anns_field="vector",  # 进行向量搜索的字段
        # 度量方式：L2 欧氏距离/IP 内积/COSINE 余弦相似度
        search_params={"metric_type": "L2"},
        output_fields=["text", "metadata"],  # 输出字段
        limit=3,  # 搜索结果数量
    )
    return context


question = input("输入问题: ")
context = retrieval(question, embed_model, client)
print(context)

# 通过主键查询实体
res = client.get(
    collection_name="demo_collection",
    ids=[465900799019188638, 465900799019188639],
    output_fields=["text", "metadata"],
)
print(res)

# 通过过滤条件(https://milvus.io/docs/zh/boolean.md)查询实体
res = client.query(
    collection_name="demo_collection",
    filter='metadata["source"] == "./rabbit.pdf"',  # 使用 metadata["source"] 进行过滤
    output_fields=["text", "metadata"],
    limit=10,
)
print(res)
