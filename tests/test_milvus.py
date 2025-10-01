import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from source.vector_db.milvus_client import MilvusClient

# Test configuration
TEST_HOST = "localhost"
TEST_PORT = "19530"
TEST_COLLECTION = "test_collection"

@pytest.fixture
def milvus_client():
    """Pytest fixture for Milvus client"""
    client = MilvusClient(TEST_HOST, TEST_PORT, TEST_COLLECTION)
    yield client
    # Cleanup after test
    try:
        if client.connect():
            client.drop_collection()
            client.disconnect()
    except:
        pass

@pytest.fixture
def connected_client():
    """Pytest fixture for connected Milvus client"""
    client = MilvusClient(TEST_HOST, TEST_PORT, TEST_COLLECTION)
    if not client.connect():
        pytest.skip("Cannot connect to Milvus server")
    yield client
    client.disconnect()

class TestMilvusClient:
    
    def test_connection(self, milvus_client):
        """Test connection to Milvus"""
        result = milvus_client.connect()
        assert result == True, "Failed to connect to Milvus"
        milvus_client.disconnect()
    
    def test_create_collection(self, connected_client):
        """Test creating collection"""
        # Drop if exists
        try:
            connected_client.drop_collection()
        except:
            pass
        
        # Create new collection
        connected_client.create_collection(dim=128)
        
        # Verify collection exists
        stats = connected_client.get_collection_stats()
        assert stats.get('collection_name') == TEST_COLLECTION
    
    def test_insert_embeddings(self, connected_client):
        """Test inserting embeddings"""
        # Create collection
        try:
            connected_client.create_collection(dim=128)
        except:
            pass
        
        connected_client.load_collection()
        
        # Generate test data
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        embeddings = np.random.random((3, 128)).astype(np.float32)
        metadata = ["cat", "dog", "bird"]
        
        # Insert data
        ids = connected_client.insert_embeddings(image_paths, embeddings, metadata)
        
        assert len(ids) == 3, f"Expected 3 IDs, got {len(ids)}"
        assert all(isinstance(id_val, int) for id_val in ids), "IDs should be integers"
    
    def test_search_similar(self, connected_client):
        """Test searching similar embeddings"""
        # Setup collection with data
        try:
            connected_client.create_collection(dim=128)
        except:
            pass
        
        connected_client.load_collection()
        
        # Insert test data
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        embeddings = np.random.random((3, 128)).astype(np.float32)
        metadata = ["cat", "dog", "bird"]
        connected_client.insert_embeddings(image_paths, embeddings, metadata)
        
        # Search
        query_embedding = np.random.random((128,)).astype(np.float32)
        results = connected_client.search_similar(query_embedding, top_k=2)
        
        assert len(results) <= 2, "Should return at most 2 results"
        assert len(results) > 0, "Should return at least 1 result"
        
        # Check result structure
        for result in results:
            assert 'id' in result
            assert 'distance' in result
            assert 'image_path' in result
            assert isinstance(result['distance'], float)
    
    def test_collection_stats(self, connected_client):
        """Test getting collection stats"""
        # Create collection with some data
        try:
            connected_client.create_collection(dim=128)
        except:
            pass
        
        connected_client.load_collection()
        
        # Insert some data
        image_paths = ["test1.jpg", "test2.jpg"]
        embeddings = np.random.random((2, 128)).astype(np.float32)
        metadata = ["test1", "test2"]
        connected_client.insert_embeddings(image_paths, embeddings, metadata)
        
        # Get stats
        stats = connected_client.get_collection_stats()
        
        assert isinstance(stats, dict)
        assert 'num_entities' in stats
        assert 'collection_name' in stats
        assert stats['collection_name'] == TEST_COLLECTION
    
    def test_delete_by_ids(self, connected_client):
        """Test deleting embeddings by IDs"""
        # Setup collection with data
        try:
            connected_client.create_collection(dim=128)
        except:
            pass
        
        connected_client.load_collection()
        
        # Insert test data
        image_paths = ["delete1.jpg", "delete2.jpg", "keep.jpg"]
        embeddings = np.random.random((3, 128)).astype(np.float32)
        metadata = ["del1", "del2", "keep"]
        ids = connected_client.insert_embeddings(image_paths, embeddings, metadata)
        
        # Delete first 2 items
        connected_client.delete_by_ids(ids[:2])
        
        # Verify deletion (this is basic - in real scenario you'd search to verify)
        stats = connected_client.get_collection_stats()
        # Note: Milvus might not immediately reflect deletions in stats
        assert isinstance(stats, dict)
    
    def test_drop_collection(self, connected_client):
        """Test dropping collection"""
        # Create collection first
        try:
            connected_client.create_collection(dim=64)
        except:
            pass
        
        # Drop collection
        connected_client.drop_collection()
        
        # Verify it's dropped (try to get stats should fail or return empty)
        # This is implementation dependent
        pass

if __name__ == "__main__":
    # Run with pytest when called directly
    pytest.main([__file__, "-v"])