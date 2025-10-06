import time
import os
import sys
import numpy as np
from PIL import Image

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from embedder.extractor_factory import ExtractorFactory
from vector_db.milvus_client import MilvusClient


class ModelBenchmark:
    """Simple benchmark for embedding models and search performance"""
    
    def __init__(self):
        self.extractor_types = ["resnet", "vgg", "vit", "dinov2", "clip"]
        self.test_images = []
        self.results = {}
    
    def load_test_images(self, image_dir="../images/evaluation", max_images=10):
        """Load test images for benchmarking"""
        if not os.path.exists(image_dir):
            print(f"Warning: {image_dir} not found, creating dummy images")
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
                        print(f"Error loading {file}: {e}")
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
        print("Benchmarking Embedding Models...")
        print("=" * 50)
        
        results = {}
        
        for model_name in self.extractor_types:
            print(f"\nTesting {model_name}...")
            try:
                # Initialize extractor
                extractor = ExtractorFactory.get_extractor(model_name)
                
                times = []
                for run in range(num_runs):
                    start_time = time.time()
                    
                    # Extract features for all images
                    for img in images:
                        features = extractor.extract_features(img)
                    
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
                
                print(f"  Average: {avg_time:.3f}s total, {avg_per_image:.3f}s per image")
                print(f"  Feature dim: {extractor.feature_dim}")
                
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def benchmark_milvus_search(self, num_searches=10, top_k=5):
        """Benchmark Milvus search performance"""
        print("\nBenchmarking Milvus Search...")
        print("=" * 50)
        
        try:
            # Initialize Milvus client
            milvus_client = MilvusClient()
            collection_name = "image_vectors"  # Default collection
            
            # Check if collection exists
            collections = milvus_client.client.list_collections()
            if collection_name not in [col.name for col in collections]:
                print(f"Collection {collection_name} not found")
                return {'error': 'Collection not found'}
            
            # Create random query vectors (assuming 512-dim features)
            search_times = []
            
            for i in range(num_searches):
                # Random query vector
                query_vector = np.random.random(512).tolist()
                
                start_time = time.time()
                
                # Perform search
                results = milvus_client.search(
                    collection_name=collection_name,
                    query_vectors=[query_vector],
                    limit=top_k
                )
                
                end_time = time.time()
                search_time = end_time - start_time
                search_times.append(search_time)
                
                print(f"Search {i+1}: {search_time:.4f}s, found {len(results[0]) if results else 0} results")
            
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
            
            print(f"\nSearch Statistics:")
            print(f"  Average: {avg_search_time:.4f}s")
            print(f"  Min: {min_search_time:.4f}s")
            print(f"  Max: {max_search_time:.4f}s")
            
            return search_results
            
        except Exception as e:
            print(f"Error benchmarking Milvus: {e}")
            return {'error': str(e)}
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting Image Retrieval Benchmark")
        print("=" * 50)
        
        # Load test images
        print("Loading test images...")
        self.test_images = self.load_test_images()
        print(f"Loaded {len(self.test_images)} test images")
        
        # Benchmark embedding models
        embedding_results = self.benchmark_embedding_models(self.test_images)
        
        # Benchmark Milvus search
        search_results = self.benchmark_milvus_search()
        
        # Store results
        self.results = {
            'embedding_models': embedding_results,
            'milvus_search': search_results,
            'test_info': {
                'num_test_images': len(self.test_images),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No results to display. Run benchmark first.")
            return
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Embedding models summary
        print("\nEmbedding Models Performance:")
        print("-" * 40)
        
        embedding_results = self.results['embedding_models']
        for model, data in embedding_results.items():
            if 'error' in data:
                print(f"{model:10}: ERROR - {data['error']}")
            else:
                print(f"{model:10}: {data['avg_per_image']:.3f}s per image, dim: {data['feature_dim']}")
        
        # Search performance summary
        print("\nMilvus Search Performance:")
        print("-" * 40)
        
        search_results = self.results['milvus_search']
        if 'error' in search_results:
            print(f"Search: ERROR - {search_results['error']}")
        else:
            print(f"Average search time: {search_results['avg_search_time']:.4f}s")
            print(f"Search range: {search_results['min_search_time']:.4f}s - {search_results['max_search_time']:.4f}s")


def main():
    """Main benchmark function"""
    benchmark = ModelBenchmark()
    
    # Run benchmark
    results = benchmark.run_full_benchmark()
    
    # Print summary
    benchmark.print_summary()
    
    return results


if __name__ == "__main__":
    main()
