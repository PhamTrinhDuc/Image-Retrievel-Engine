import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import timm
import torch
from base.base_embedder import ImageFeatureExtractor


class ViTExtractor(ImageFeatureExtractor): 
    """Vision Transformer (ViT) feature extractor with batch processing."""
    
    # Supported ViT models with their feature dimensions
    SUPPORTED_MODELS = {
        'vit_tiny_patch16_224': 192,
        'vit_small_patch16_224': 384,
        'vit_base_patch16_224': 768,
        'vit_base_patch32_224': 768,
        'vit_large_patch16_224': 1024,
        'vit_large_patch32_224': 1024,
        'deit_tiny_patch16_224': 192,
        'deit_small_patch16_224': 384,
        'deit_base_patch16_224': 768
    }
    
    def __init__(self, 
                 model_name: str = "vit_base_patch16_224", 
                 device: str = "cpu",
                 batch_size: int = 16,
                 enable_mixed_precision: bool = False): 
        """
        Initialize ViT feature extractor with enhanced capabilities.
        
        Args:
            model_name: ViT model variant to use
            device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.)
            batch_size: Default batch size for batch processing
            enable_mixed_precision: Use automatic mixed precision (AMP)
        """
        super().__init__(model_name=model_name, 
                         device=device, 
                         batch_size=batch_size, 
                         enable_mixed_precision=enable_mixed_precision)
        
    def load_model(self) -> torch.nn.Module:
      """Load pretrained ViT model with validation."""
      try:
        self.logger.info(f"Loading model: {self.model_name}")
        model = timm.create_model(
            model_name=self.model_name, 
            pretrained=True, 
            num_classes=0,  # Remove classification head
            global_pool="token",  # Use CLS token for ViT
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