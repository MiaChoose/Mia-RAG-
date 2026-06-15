import requests

AUTH = ("elastic", "6R*dRWKOBS7pO-oFtPII")

# 查看文档1的 embedding 字段
resp = requests.get(
    "http://localhost:9200/knowledge_base/_doc/1?_source_includes=embedding",
    auth=AUTH
)

source = resp.json().get("_source", {})
print("文档1 的字段:", list(source.keys()))

if "embedding" in source:
    embedding = source["embedding"]
    print(f"embedding 存在, 维度: {len(embedding)}")
    print(f"前10个值: {embedding[:10]}")
else:
    print("embedding 不存在")