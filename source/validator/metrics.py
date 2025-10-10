import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from urllib.parse import urlparse
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import glob
import json

from retriever.retriever import ImageRetriever
from utils.helpers import create_logger
from configs.helper import DataConfig
from data_processer.minio_client import MinioClient


RESULT_PATH = "./source/validator/results/metrics_results.json"

minio_client = MinioClient(
  minio_endpoint=DataConfig.minio_endpoint,
  minio_access_key=DataConfig.minio_access_key,
  minio_secret_key=DataConfig.minio_secret_key,
  bucket_name=DataConfig.bucket_name
)


class ImageRetrievalMetrics:
    def __init__(self):
      self.logger = create_logger()
        
    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Tính Precision@K
        
        Args:
            retrieved_ids: List các image IDs được retrieve (theo thứ tự ranking)
            relevant_ids: Set các image IDs relevant với query
            k: Top-K results để tính precision
            
        Returns:
            Precision@K score (0.0 - 1.0)
        """
        if k <= 0 or len(retrieved_ids) == 0:
            return 0.0
            
        # Chỉ lấy top-k results
        top_k_retrieved = retrieved_ids[:k]
        
        # Đếm số relevant items trong top-k
        relevant_count = sum(1 for item_id in top_k_retrieved if item_id in relevant_ids)
        
        return relevant_count / len(top_k_retrieved)
    
    def recall_at_k(self, retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Tính Recall@K
        
        Args:
            retrieved_ids: List các image IDs được retrieve
            relevant_ids: Set các image IDs relevant với query  
            k: Top-K results để tính recall
            
        Returns:
            Recall@K score (0.0 - 1.0)
        """
        if len(relevant_ids) == 0:
            return 0.0
            
        if k <= 0 or len(retrieved_ids) == 0:
            return 0.0
            
        # Chỉ lấy top-k results
        top_k_retrieved = retrieved_ids[:k]
        
        # Đếm số relevant items trong top-k
        relevant_count = sum(1 for item_id in top_k_retrieved if item_id in relevant_ids)
        
        return relevant_count / len(relevant_ids)
    
    def average_precision(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Tính Average Precision (AP) cho một query
        
        Args:
            retrieved_ids: List các image IDs được retrieve (theo thứ tự ranking)
            relevant_ids: Set các image IDs relevant với query
            
        Returns:
            Average Precision score (0.0 - 1.0)
        """
        if len(relevant_ids) == 0:
            return 0.0
            
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item_id in enumerate(retrieved_ids):
            if item_id in relevant_ids:
                relevant_count += 1
                # Tính precision tại vị trí i+1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
            
        return precision_sum / len(relevant_ids)
    
    def mean_average_precision(self, query_results: Dict[str, Tuple[List[str], Set[str]]]) -> float:
        """
        Tính Mean Average Precision (mAP) cho tất cả queries
        
        Args:
            query_results: Dict với:
                - key: query_id  
                - value: tuple (retrieved_ids, relevant_ids)
                
        Returns:
            mAP score (0.0 - 1.0)
        """
        if len(query_results) == 0:
            return 0.0
            
        ap_sum = 0.0
        for query_id, (retrieved_ids, relevant_ids) in query_results.items():
            ap = self.average_precision(retrieved_ids, relevant_ids)
            ap_sum += ap
            
        return ap_sum / len(query_results)
    
    def evaluate_batch(self, 
                      query_results: Dict[str, Tuple[List[str], Set[str]]], 
                      k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Đánh giá batch queries với nhiều metrics
        
        Args:
            query_results: Dict kết quả retrieval cho từng query
            k_values: List các giá trị K để tính precision/recall@K
            
        Returns:
            Dict chứa tất cả metrics
        """
        results = {}
        
        # Tính mAP
        results['mAP'] = self.mean_average_precision(query_results)
        
        # Tính precision và recall cho từng K
        for k in k_values:
            precision_scores = []
            recall_scores = []
            
            for query_id, (retrieved_ids, relevant_ids) in query_results.items():
                # print(f"Top {k}: {query_id}")
                # print(f"Retrieved IDs: {retrieved_ids[:k]}")
                # print(f"Relevant IDs: {relevant_ids}")
                # print("-----" * 50)
                prec_k = self.precision_at_k(retrieved_ids, relevant_ids, k)
                rec_k = self.recall_at_k(retrieved_ids, relevant_ids, k)
                
                precision_scores.append(prec_k)
                recall_scores.append(rec_k)
            
            # Trung bình tất cả queries
            results[f'Precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'Recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
        
        return results
    
    def create_ground_truth_from_folders(self) -> Dict[str, Set[str]]:
        """
        Tạo ground truth từ cấu trúc folder (mỗi folder = một class)
        
        Args:
            data_dir: Path tới folder chứa data (có subfolder cho từng class)
            
        Returns:
            Dict mapping từ image_id đến set of relevant image_ids (cùng class)
        """
        categories = sorted(minio_client.get_categories())
        ground_truth = defaultdict(list)
        
        for category in categories:
          ground_truth[category] = minio_client.get_images_in_category(category=category)
          ground_truth[category] = [os.path.splitext(os.path.basename(img_link))[0] for img_link in ground_truth[category]]
        
        return ground_truth


class Evaluator:
    """
    Class đơn giản để chạy evaluation hoàn chỉnh
    """
    
    def __init__(self, 
                 retriever: ImageRetriever,
                 metrics_calculator: ImageRetrievalMetrics):
        self.retriever = retriever
        self.metrics = metrics_calculator
        self.logger = create_logger("evaluator")
        
    def evaluate_on_dataset(self, 
                           query_images: List[str], 
                           ground_truth: Dict[str, Set[str]],
                           top_k: int = 50,
                           eval_k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Chạy evaluation trên dataset
        
        Args:
            query_images: List path của query images
            ground_truth: Ground truth mapping 
            top_k: Số lượng images retrieve cho mỗi query
            eval_k_values: Các giá trị K để evaluate
            
        Returns:
            Dict kết quả metrics
        """
        query_results = {}
        
        self.logger.info(f"Starting evaluation on {len(query_images)} queries...")
        
        for i, query_img in enumerate(query_images):
            try:
                # Search với retriever
                search_results = self.retriever.search_similar_images(
                    query_input=query_img,
                    top_k=top_k,
                )
                
                # Extract retrieved IDs (bỏ query image nếu có)
                retrieved_basename = []
                for result in search_results:
                    img_link = os.path.basename(result.get('image_path'))
                    retrieved_basename.append(os.path.splitext(os.path.basename(urlparse(img_link).path))[0]) # get basename from url

                # Get ground truth cho query này
                basename_query = query_img.split("/")[-2]  # Lấy tên folder cha của ảnh
                relevant_basename = ground_truth.get(basename_query, set())

                query_results[basename_query] = (retrieved_basename, relevant_basename)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(query_images)} queries")
                    
            except Exception as e:
                self.logger.error(f"Error processing query {query_img}: {str(e)}")
                continue
        
        results = {"model_name": self.retriever.extractor_type,
                   "vdb_type": self.retriever.vdb_type,
                   "metrics": {}}
        # Calculate metrics
        metrics = self.metrics.evaluate_batch(query_results, eval_k_values)
        results['metrics'] = metrics
        
        with open(RESULT_PATH, mode="a") as f:
            f.write(json.dumps(results, indent=2))

        self.logger.info("Evaluation completed!")
        return results
    

def main():
    # Initialize components
    retriever = ImageRetriever(
        extractor_type='resnet',
        vdb_type='milvus',
        extractor_config={'model_name': 'resnet34', 'device': 'cpu'},
        vdb_config={'collection_name': 'resnet_embedding'}
    )
    
    metrics = ImageRetrievalMetrics()
    evaluator = Evaluator(retriever, metrics)
    
    # Setup retriever
    if not retriever.connect_and_load():
        print("Failed to setup retriever!")
        return
    
    # Tạo ground truth từ folder structure
    ground_truth = metrics.create_ground_truth_from_folders()
    
    # Collect query images (có thể random sample)
    data_dir = "images/evaluation/"
    query_images = glob.glob(pathname=data_dir + "/*/*.png", recursive=True)

    results = evaluator.evaluate_on_dataset(
        query_images=query_images,
        ground_truth=ground_truth,
        top_k=50,
        eval_k_values=[1, 5, 10, 20]
    )
    
    return results


if __name__ == "__main__":
    main()