import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import timm
import torch
from base.base_embedder import ImageFeatureExtractor
from configs.helper import DataConfig

class DINOv2Extractor(ImageFeatureExtractor): 
    """DINOv2 feature extractor with batch processing."""
    
    # Supported DINOv2 models with their feature dimensions
    SUPPORTED_MODELS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
        'dinov2_vits14_reg': 384,
        'dinov2_vitb14_reg': 768,
        'dinov2_vitl14_reg': 1024,
        'dinov2_vitg14_reg': 1536
    }

    JOB_NAME = "dinov2_extractor"
    
    def __init__(self, 
                 model_name: str = "dinov2_vitb14", 
                 device: str = "cpu",
                 batch_size: int = 16,
                 enable_mixed_precision: bool = False): 
        """
        Initialize DINOv2 feature extractor with enhanced capabilities.
        
        Args:
            model_name: DINOv2 model variant to use
            device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.)
            batch_size: Default batch size for batch processing
            enable_mixed_precision: Use automatic mixed precision (AMP)
        """
        super().__init__(model_name=model_name, 
                         batch_size=batch_size, 
                         device=device, 
                         enable_mixed_precision=enable_mixed_precision)
        
       
    def load_model(self) -> torch.nn.Module:
      """Load pretrained DINOv2 model with validation."""
      try:
        self.logger.info(f"Loading model: {self.model_name}")
        model = timm.create_model(
            model_name=self.model_name, 
            cache_dir=os.path.join(DataConfig.model_cache_dir, self.model_name),
            pretrained=True, 
            num_classes=0,  # Remove classification head
            global_pool="token",  # Use CLS token for DINOv2
            drop_rate=0.0  # Disable dropout for inference
        )
        
        # Validate model architecture
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            test_output = model(test_input)
        
        expected_dim = self.SUPPORTED_MODELS.get(self.model_name, 768)
        actual_dim = test_output.shape[-1]
        
        if actual_dim != expected_dim:
            self.logger.warning(f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        self.logger.info(f"Model loaded successfully. Feature dim: {actual_dim}")
        return model.eval()
        
      except Exception as e:
        self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")