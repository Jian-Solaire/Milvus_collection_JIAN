# Milvus RAG API

基于 Milvus + BGE-M3 的 RAG (检索增强生成) API 服务。

## 功能特性

- 📄 **文档处理**: 支持 PDF、Word、Excel、PPT、TXT、Markdown 等多格式
- 🔍 **向量检索**: 混合检索 (Dense + Sparse)，支持 MQE/HyDE 查询扩展
- ⚡ **高性能**: 批量处理、异步索引
- 🏷️ **命名空间**: 多知识库隔离

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env)
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=your_user
MILVUS_PASSWORD=your_password
MILVUS_COLLECTION_NAME=rag_vectors
MODEL_DEVICE=cuda  # 或 cpu

```
记得拉docker，https://milvus.io/docs/zh/install_standalone-windows.md


## 启动服务

```bash
python -m api.rag_api
# 默认端口: 8000
```

## API 接口

### 健康检查

```bash
GET /health
```

### 文档摄取 (文本)

```bash
POST /knowledge/ingest
Content-Type: application/json

{
  "texts": ["要索引的文本内容"],
  "metadata": [{"source": "doc1"}],
  "options": {
    "chunking": {"size": 800, "overlap": 100},
    "embedding": {"namespace": "default"}
  }
}
```

### 文档摄取 (文件)

```bash
POST /knowledge/ingest/files
Content-Type: application/json

{
  "file_paths": ["/path/to/file.pdf"],
  "options": {
    "chunking": {"size": 800, "overlap": 100},
    "namespace": "default"
  }
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "files_processed": 1,
    "chunks_generated": 50,
    "indexed_count": 50,
    "duration_seconds": 2.345
  },
  "meta": {
    "namespace": "default",
    "timing": {
      "document_processing_seconds": 1.2,
      "chunking_seconds": 0.3,
      "indexing_seconds": 0.845
    }
  }
}
```

### 知识库搜索

```bash
POST /query
Content-Type: application/json

{
  "query": "你的查询问题",
  "options": {
    "search": {
      "namespace": "default",
      "top_k": 5,
      "type": "hybrid",
      "score_threshold": 0.5
    },
    "enhance": {
      "mqe": false,
      "hyde": false
    }
  }
}
```

**响应示例:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "xxx",
        "text": "相关文档片段...",
        "score": 0.85,
        "source_path": "/path/to/doc.pdf"
      }
    ],
    "count": 5,
    "query": "你的查询问题",
    "search_type": "hybrid",
    "duration_seconds": 0.125
  }
}
```

### 知识库统计

```bash
GET /knowledge/stats
```

### 删除文档

```bash
DELETE /knowledge/delete
Content-Type: application/json

{
  "chunk_ids": ["chunk_id_1", "chunk_id_2"]
}
```

### 清空知识库

```bash
DELETE /knowledge/clear
```

## 代码结构

```
script/
├── api/                  # FastAPI 接口
│   └── rag_api.py
├── service/              # 业务服务层
│   ├── ingestion_service.py
│   └── retrieval_service.py
├── pipeline/             # 处理管道
│   ├── offline_pipeline.py   # 文档处理、索引
│   └── online_pipeline.py    # 查询检索
├── core/                 # 核心组件
│   └── embedding.py      # BGE-M3 向量化
├── stores/               # 存储层
│   └── milvus_store.py
└── config/
    └── settings.py       # 配置管理
```

## 使用示例

```python
import requests

# 1. 摄取文档
resp = requests.post("http://localhost:8000/knowledge/ingest/files", json={
    "file_paths": ["document.pdf"]
})
print(resp.json())

# 2. 搜索查询
resp = requests.post("http://localhost:8000/query", json={
    "query": "什么是机器学习？"
})
print(resp.json())
```
