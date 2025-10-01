# Image Retrieval Engine

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites) 
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [API Usage](#api-usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Monitoring & Logging](#monitoring--logging)
- [Deployment](#deployment)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## � Overview
Image Retrieval Engine với MLOps integration - tìm kiếm hình ảnh dựa trên vector similarity.

## 📋 Prerequisites

## 🚀 Installation

## ⚙️ Configuration

### Milvus Vector Database
#### Run Milvus Container
```bash
# Start Milvus
bash standalone_embed.sh start
# Stop Milvus  
bash standalone_embed.sh stop
# Delete container + volumes
bash standalone_embed.sh delete
```

#### Check Milvus Status
```bash
# Verify Milvus is running
python milvus_e2e_qa/check_milvus.py

# Test connection
python milvus_e2e_qa/hello_milvus.py
```

## 📊 Data Processing

## 🤖 Model Training

## 📚 API Usage

## � MLOps Pipeline

## 📊 Monitoring & Logging

## 🚀 Deployment

## 🧪 Testing
### 1. Test milvus
```bash
# Chạy tất cả tests
pytest tests/test_milvus.py -v
# Chạy với output chi tiết
pytest tests/test_milvus.py -v -s
# Chạy và dừng ở test đầu tiên fail
pytest tests/test_milvus.py -x
# Chạy parallel (nếu có pytest-xdist)
pytest tests/test_milvus.py -n auto
```

## ⚡ Performance Optimization

## 🔧 Troubleshooting

## 🤝 Contributing