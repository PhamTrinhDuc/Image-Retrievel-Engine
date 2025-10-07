import time
import os
import sys
import numpy as np
from PIL import Image

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.helpers import create_logger
from embedder.extractor_factory import EmbedderFactory
from vector_db.milvus_client import MilvusClient


logger = create_logger(job_name="benchmark_timer")
RESULT_PATH = "./source/validator/results/benchmark_results.json"

class ModelBenchmark:
    """Simple benchmark for embedding models and search performance"""
    
    def __init__(self):
        self.extractor_types = ["resnet"]
        self.test_images = []
        self.results = {}
    
    def load_test_images(self, image_dir="images/evaluation", max_images=10):
        """Load test images for benchmarking"""
        if not os.path.exists(image_dir):
            logger.warning(f"Warning: {image_dir} not found, creating dummy images")
            return self._create_dummy_images(max_images)
        
        images = []
        count = 0
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')) and count < max_images:
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error loading {file}: {e}")
                        continue
        
        return images
    
    def _create_dummy_images(self, count=10):
        """Create dummy images for testing"""
        images = []
        for i in range(count):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            images.append(img)
        return images
    
    def benchmark_embedding_models(self, images, num_runs=3):
        """Benchmark embedding extraction time for different models"""
        logger.info("Benchmarking Embedding Models...")
        
        results = {}
        
        for model_name in self.extractor_types:
            print(f"\nTesting {model_name}...")
            try:
                # Initialize extractor
                extractor = EmbedderFactory.create_extractor(extractor_type=model_name)
                
                times = []
                for run in range(num_runs):
                    start_time = time.time()
                    
                    # Extract features for all images
                    for img in images:
                        features = extractor.get_embedding(img)
                    
                    end_time = time.time()
                    run_time = end_time - start_time
                    times.append(run_time)
                    
                    print(f"  Run {run+1}: {run_time:.3f}s")
                
                # Calculate statistics
                avg_time = np.mean(times)
                avg_per_image = avg_time / len(images)
                
                results[model_name] = {
                    'avg_total_time': avg_time,
                    'avg_per_image': avg_per_image,
                    'feature_dim': extractor.feature_dim,
                    'all_times': times
                }

                logger.info(f"  Average: {avg_time:.3f}s total, {avg_per_image:.3f}s per image")

            except Exception as e:
                logger.error(f"  Error with {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results

    def benchmark_milvus_search(self, collection_name: str, num_searches=10, top_k=5):
        """Benchmark Milvus search performance"""
        logger.info("Benchmarking Milvus Search...")

        try:
            # Initialize Milvus client
            milvus_client = MilvusClient(host="localhost", port="19530", collection_name=collection_name)
            milvus_client.connect()
            milvus_client.load_collection()
            
            # Check if collection exists
            collections = milvus_client.list_collections()
            if collection_name not in collections:
                logger.error(f"Collection {collection_name} not found")
                return {'error': 'Collection not found'}
            
            # Create random query vectors (assuming 512-dim features)
            search_times = []
            
            for i in range(num_searches):
                # Random query vector
                query_vector = np.random.random(512)
                
                start_time = time.time()
                
                # Perform search
                results = milvus_client.search_similar(
                    query_embedding=query_vector,
                    top_k=top_k
                )
                
                end_time = time.time()
                search_time = end_time - start_time
                search_times.append(search_time)

                logger.info(f"Search {i+1}: {search_time:.4f}s, found {len(results[0]) if results else 0} results")

            # Calculate statistics
            avg_search_time = np.mean(search_times)
            min_search_time = np.min(search_times)
            max_search_time = np.max(search_times)
            
            search_results = {
                'avg_search_time': avg_search_time,
                'min_search_time': min_search_time,
                'max_search_time': max_search_time,
                'all_times': search_times,
                'top_k': top_k
            }
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error benchmarking Milvus: {e}")
            return {'error': str(e)}
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        logger.info("Starting Image Retrieval Benchmark")

        # Load test images
        logger.info("Loading test images...")
        self.test_images = self.load_test_images()
        logger.info(f"Loaded {len(self.test_images)} test images")

        # Benchmark embedding models
        embedding_results = self.benchmark_embedding_models(self.test_images)
        
        # Benchmark Milvus search
        search_results = self.benchmark_milvus_search(collection_name="resnet_embedding")
        
        # Store results
        self.results = {
            'embedding_models': embedding_results,
            'milvus_search': search_results,
            'test_info': {
                'num_test_images': len(self.test_images),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
        with open(RESULT_PATH, mode="a") as f:
            import json
            f.write(json.dumps(self.results, indent=2))
            
        return self.results
    

def main():
    """Main benchmark function"""
    benchmark = ModelBenchmark()
    
    # Run benchmark
    results = benchmark.run_full_benchmark()
    
    return results


if __name__ == "__main__":
    main()
