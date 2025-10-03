#!/usr/bin/env python3
"""
Unit tests for the essential functions of VGGExtractor.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
from PIL import Image
from source.models import VGGExtractor

class TestVGGExtractor:
    """Test suite for essential VGGExtractor functions."""

    @pytest.fixture
    def extractor(self):
        """Create a VGGExtractor instance for testing."""
        return VGGExtractor(
            model_name="vgg16",  # Use VGG16 for testing
            device="cpu",
            batch_size=4
        )

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        return Image.new('RGB', (224, 224), color='blue')

    def test_initialization(self, extractor):
        """Test proper initialization of the extractor."""
        assert extractor.model_name == "vgg16"
        assert extractor.device == "cpu"
        assert extractor.feature_dim['model_dim'] == 4096
        assert extractor.feature_dim['projection_dim'] == 768

    def test_single_image_processing(self, extractor, sample_image):
        """Test single image feature extraction."""
        features = extractor.get_embedding(sample_image)

        assert isinstance(features, np.ndarray)
        assert features.shape == (768,)  # Projected dimension
        assert np.all(np.isfinite(features))  # No NaN or inf values

        # Test normalization (L2 norm should be approximately 1)
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 1e-6

    def test_batch_processing(self, extractor, sample_image):
        """Test batch image processing."""
        sample_images = [sample_image] * 4  # Create a batch of 4 identical images
        batch_features = extractor.get_batch_embeddings(sample_images)

        assert isinstance(batch_features, np.ndarray)
        assert batch_features.shape == (4, 768)  # Batch size x Projected dimension
        assert np.all(np.isfinite(batch_features))

        # Test that each row is normalized
        norms = np.linalg.norm(batch_features, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-6)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])