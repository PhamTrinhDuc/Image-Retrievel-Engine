#!/usr/bin/env python3
"""
Simplified unit tests for the ViTExtractor.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytest
import numpy as np
from PIL import Image
from source.models import ViTExtractor

class TestViTExtractor:
    """Simplified test suite for ViTExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a ViTExtractor instance for testing."""
        return ViTExtractor(
            model_name="vit_base_patch16_224",  # Use smaller model for faster tests
            device="cpu",
            embed_dim=768,
            batch_size=4
        )

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        return Image.new('RGB', (224, 224), color='blue')

    @pytest.fixture
    def sample_images(self, sample_image):
        """Create multiple sample images for batch testing."""
        images = [sample_image]
        for i in range(4):
            # Create slightly different images
            color = (i * 50, (i * 30) % 255, (i * 80) % 255)
            images.append(Image.new('RGB', (224, 224), color=color))
        return images

    def test_initialization(self, extractor):
        """Test proper initialization of the extractor."""
        assert extractor.model_name == "resnet18"
        assert extractor.device == "cpu"
        assert extractor.batch_size == 4
        assert extractor.feature_dim['model_dim'] == 512
        assert extractor.feature_dim['projection_dim'] == 768

    def test_single_image_processing(self, extractor, sample_image):
        """Test single image feature extraction."""
        features = extractor.get_embedding(sample_image)

        assert isinstance(features, np.ndarray)
        assert features.shape == (768,)  # resnet18 feature dimension
        assert np.all(np.isfinite(features))  # No NaN or inf values

        # Test normalization (L2 norm should be approximately 1)
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 1e-6

    def test_batch_processing(self, extractor, sample_images):
        """Test batch image processing."""
        batch_features = extractor.get_batch_embeddings(sample_images)

        assert isinstance(batch_features, np.ndarray)
        assert batch_features.shape == (len(sample_images), 768)
        assert np.all(np.isfinite(batch_features))

        # Test that each row is normalized
        norms = np.linalg.norm(batch_features, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-6)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])