import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import timm
import torch
from base.base_embedder import ImageFeatureExtractor
from configs.helper import DataConfig 


class ResNetExtractor(ImageFeatureExtractor): 
    """ResNet feature extractor with batch processing."""
    
    # Supported ResNet models with their feature dimensions
    SUPPORTED_MODELS = {
        'resnet18': 512,
        'resnet34': 512, 
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
        'resnext50_32x4d': 2048,
        'resnext101_32x8d': 2048,
        'wide_resnet50_2': 2048,
        'wide_resnet101_2': 2048
    }
    def __init__(self, 
                 model_name: str = "resnet34", 
                 device: str = "cpu",
                 batch_size: int = 16,
                 enable_mixed_precision: bool = False): 
        """
        Initialize ResNet feature extractor with enhanced capabilities.
        
        Args:
            model_name: ResNet model variant to use
            device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.)
            enable_caching: Whether to cache preprocessed images
            cache_size: Maximum number of cached items
            batch_size: Default batch size for batch processing
            enable_mixed_precision: Use automatic mixed precision (AMP)
        """
        super().__init__(model_name=model_name, 
                         device=device, 
                         batch_size=batch_size, 
                         enable_mixed_precision=enable_mixed_precision)
        
    def load_model(self) -> torch.nn.Module:
      """Load pretrained ResNet model with validation."""
      try:
        self.logger.info(f"Loading model: {self.model_name}")
        model = timm.create_model(
            model_name=self.model_name, 
            cache_dir=os.path.join(DataConfig.model_cache_dir, self.model_name),
            pretrained=True, 
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global average pooling
            drop_rate=0.0  # Disable dropout for inference
        )
        
        # Validate model architecture
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            test_output = model(test_input)
        
        expected_dim = self.SUPPORTED_MODELS.get(self.model_name, 512)
        actual_dim = test_output.shape[-1]
        
        if actual_dim != expected_dim:
            self.logger.warning(f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        self.logger.info(f"Model loaded successfully. Feature dim: {actual_dim}")
        return model.eval()
        
      except Exception as e:
        self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")
    