"""
演示如何在 Pinecone 中搜索相似郵件

這個腳本展示完整的查詢流程：
1. 加載相同的 embedding 模型
2. 將查詢文本轉換為向量
3. 在 Pinecone 中搜索
4. 顯示結果
"""

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# 加載環境變量
load_dotenv()

# 配置
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
INDEX_NAME = "email-rag-search"

print("=" * 60)
print("郵件語義搜索演示")
print("=" * 60)

# 1. 加載 Embedding 模型（與摄取時相同）
print("\n[1/4] 加載 embedding 模型...")
model = SentenceTransformer(MODEL_NAME)
print(f"✓ 模型加載完成: {MODEL_NAME}")

# 2. 連接 Pinecone
print("\n[2/4] 連接到 Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)
print(f"✓ 已連接到索引: {INDEX_NAME}")

# 顯示索引統計
stats = index.describe_index_stats()
print(f"   總向量數: {stats['total_vector_count']:,}")

# 3. 示例查詢
queries = [
    "找到關於面試的郵件",
    "scholarship opportunities",
    "工作機會相關的信件",
]

print(f"\n[3/4] 搜索演示")
print("-" * 60)

for query_text in queries:
    print(f"\n📧 查詢: \"{query_text}\"")

    # 將查詢轉換為向量
    query_embedding = model.encode(query_text).tolist()
    print(f"   ✓ 查詢已轉換為 {len(query_embedding)} 維向量")

    # 在 Pinecone 中搜索
    results = index.query(
        vector=query_embedding,
        top_k=5,  # 返回前 5 個最相關的結果
        include_metadata=True  # 包含元數據
    )

    print(f"   ✓ 找到 {len(results['matches'])} 個結果\n")

    # 顯示結果
    for i, match in enumerate(results['matches'], 1):
        score = match['score']
        metadata = match['metadata']

        print(f"   [{i}] 相似度: {score:.4f} ({score*100:.1f}%)")
        print(f"       主題: {metadata.get('subject', 'N/A')[:60]}...")
        print(f"       發件人: {metadata.get('from', 'N/A')}")
        print(f"       日期: {metadata.get('date', 'N/A')}")
        print(f"       帳號: {metadata.get('account', 'N/A')}")

        # 顯示內容預覽
        content = metadata.get('content', '')
        if content:
            preview = content[:150].replace('\n', ' ')
            print(f"       內容: {preview}...")
        print()

print("=" * 60)
print("[4/4] 演示完成！")
print("=" * 60)

print("\n💡 提示:")
print("   - 相似度越接近 1.0 表示越相關")
print("   - 支持中文和英文查詢")
print("   - 可以使用自然語言描述你要找的內容")
