# Evaluation – Image Retrieval Engine

## Tổng quan

Module đánh giá nằm tại `source/validator/metrics.py`, gồm hai class chính:

| Class | Nhiệm vụ |
|---|---|
| `ImageRetrievalMetrics` | Tính các metrics (Precision@K, Recall@K, AP, mAP) |
| `Evaluator` | Orchestrate toàn bộ pipeline: query → retrieve → tính metrics → lưu kết quả |

---

## Pipeline hoạt động

```
[Val images trên disk]
        │
        │  (1) Đọc từng ảnh val
        ▼
[Extractor: ResNet / ViT]
        │
        │  (2) Trích xuất embedding vector
        ▼
[Milvus Vector DB]
        │
        │  (3) ANN search → trả về top_k ảnh giống nhất (mặc định top_k=50)
        ▼
[Retrieved IDs]  ◄──────────────────────────────────────────────────┐
        │                                                             │
        │  (4) So sánh với Ground Truth                              │
        ▼                                                             │
[MinIO Object Storage]                                               │
        │                                                             │
        │  Ground truth = toàn bộ ảnh cùng class trong MinIO         │
        ▼                                                             │
[Tính Precision@K, Recall@K, AP cho từng query] ─────────────────────┘
        │
        │  (5) Trung bình qua toàn bộ queries
        ▼
[mAP, mean Precision@K, mean Recall@K]
        │
        ▼
[Lưu JSON tại source/validator/results/metrics_results.json]
```

**Lưu ý quan trọng**: `top_k=50` không phải là số lần query vào DB. Đây là số ảnh tối đa được lấy ra **trong một lần query**. Mục đích là để tính đồng thời nhiều giá trị K (1, 5, 10, 20) mà chỉ cần query DB một lần duy nhất mỗi ảnh – tiết kiệm chi phí I/O.

---

## Ground Truth

Ground truth được xây dựng từ **MinIO Object Storage** (không phải từ folder local):

```python
ground_truth = {
    "cat":  ["img_001", "img_002", "img_003", ...],  # toàn bộ ảnh class cat
    "dog":  ["img_101", "img_102", ...],
    ...
}
```

- Mỗi key là tên class (= tên folder cha của ảnh val).
- Value là list basename (không có extension) của **tất cả** ảnh trong class đó trên MinIO, bao gồm cả ảnh train.
- Ground truth phản ánh đúng định nghĩa relevance: *"ảnh relevant là ảnh cùng class"*.

---

## Metrics

### Precision@K

Tỉ lệ ảnh đúng class trong top-K kết quả trả về:

$$\text{Precision@K} = \frac{|\text{retrieved}_{1..K} \cap \text{relevant}|}{K}$$

- Đo lường **độ chính xác** của K kết quả đầu tiên.
- Không quan tâm đến ranking bên trong K.

### Recall@K

Tỉ lệ ảnh relevant được tìm thấy trong top-K:

$$\text{Recall@K} = \frac{|\text{retrieved}_{1..K} \cap \text{relevant}|}{|\text{relevant}|}$$

- Đo lường **độ bao phủ**: hệ thống tìm được bao nhiêu % ảnh cùng class.
- Tự nhiên tăng khi K tăng.

### Average Precision (AP)

Đo lường chất lượng ranking cho một query, có tính đến thứ tự:

$$\text{AP} = \frac{\sum_{i=1}^{N} P(i) \cdot \mathbb{1}[\text{rel}(i)]}{\min(|\text{relevant}|,\ \text{top\_k})}$$

Trong đó:
- $P(i)$ = Precision tại vị trí $i$
- $\mathbb{1}[\text{rel}(i)] = 1$ nếu ảnh thứ $i$ đúng class, ngược lại = 0
- Mẫu số dùng $\min(|\text{relevant}|, \text{top\_k})$ thay vì $|\text{relevant}|$

**Lý do chọn mẫu số $\min(|\text{relevant}|, \text{top\_k})$**:

Retriever chỉ trả về tối đa `top_k` kết quả. Nếu một class có 200 ảnh nhưng `top_k=50`, về mặt lý thuyết AP tốt nhất có thể đạt là $50/200 = 0.25$ dù hệ thống hoạt động hoàn hảo. Điều này penalize hệ thống một cách không công bằng. Dùng $\min(200, 50) = 50$ làm mẫu số đảm bảo AP có thể đạt 1.0 khi tất cả 50 kết quả đều đúng class.

### Mean Average Precision (mAP)

Trung bình AP qua toàn bộ queries:

$$\text{mAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}(q)$$

- $|Q|$ = tổng số query images (tất cả ảnh trong val set, không phân biệt class).
- Metric tổng hợp nhất để so sánh các model.

---

## Kết quả theo K

Với mỗi giá trị K trong `eval_k_values = [1, 5, 10, 20]`, code tính mean trên tất cả queries:

```
mean Precision@K = mean([P@K(q1), P@K(q2), ..., P@K(qN)])
mean Recall@K    = mean([R@K(q1), R@K(q2), ..., R@K(qN)])
```

Chỉ cần query DB một lần với `top_k=max(eval_k_values)` rồi cắt tại mỗi K → không query lại DB nhiều lần. Do đó `top_k` truyền vào `evaluate_on_dataset` phải **≥ max(eval_k_values)**.

---

## Cách chạy

```bash
# Từ root của project
cd /path/to/Image-Retrieval-Engine
python source/validator/metrics.py
```

Hoặc import vào script khác:

```python
from source.retriever.retriever import ImageRetriever
from source.validator.metrics import ImageRetrievalMetrics, Evaluator

retriever = ImageRetriever(
    extractor_type='resnet',
    vdb_type='milvus',
    extractor_config={'model_name': 'resnet34', 'device': 'cpu'},
    vdb_config={'collection_name': 'resnet_embedding'}
)
retriever.connect_and_load()

metrics = ImageRetrievalMetrics()
evaluator = Evaluator(retriever, metrics)

ground_truth = metrics.create_ground_truth_from_folders()

results = evaluator.evaluate_on_dataset(
    query_images=query_images,   # list đường dẫn ảnh val
    ground_truth=ground_truth,
    top_k=50,                    # phải >= max(eval_k_values)
    eval_k_values=[1, 5, 10, 20]
)
```

---

## Output

Kết quả được append vào `source/validator/results/metrics_results.json`:

```json
{
  "model_name": "resnet",
  "vdb_type": "milvus",
  "metrics": {
    "mAP": 0.812,
    "Precision@1":  0.934,
    "Recall@1":     0.021,
    "Precision@5":  0.891,
    "Recall@5":     0.098,
    "Precision@10": 0.874,
    "Recall@10":    0.183,
    "Precision@20": 0.851,
    "Recall@20":    0.340
  }
}
```

---

## Lưu ý & Giới hạn

| Vấn đề | Trạng thái |
|---|---|
| Key overwrite khi nhiều ảnh val cùng class | **Đã fix** – key là đường dẫn đầy đủ (`query_img`), không phải tên class |
| mAP underestimate khi `class size > top_k` | **Đã fix** – mẫu số dùng `min(\|relevant\|, top_k)` |
| `top_k` phải ≥ `max(eval_k_values)` | Cần đảm bảo khi gọi `evaluate_on_dataset` |
| Ground truth lấy từ **toàn bộ** MinIO (gồm cả train) | Intentional – mô phỏng real-world retrieval trên toàn bộ corpus |
| File kết quả dùng `append` mode | Nhiều lần chạy sẽ ghi nối vào file – cần xóa/quản lý thủ công |
