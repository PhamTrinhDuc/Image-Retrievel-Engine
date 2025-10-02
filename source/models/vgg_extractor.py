import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import timm
import torch
import numpy as np
from typing import Union, Any, List, Optional, Dict
from timm.data.transforms_factory import create_transform
from timm.data.config import resolve_data_config
import warnings
from base import ImageFeatureExtractor
from utils.helpers import create_logger


class VGGExtractor(ImageFeatureExtractor): 
    """VGG feature extractor with batch processing and projection layer."""
    
    # Supported VGG models with their feature dimensions
    SUPPORTED_MODELS = {
        'vgg11': 4096,
        'vgg13': 4096,
        'vgg16': 4096,
        'vgg19': 4096,
        'vgg11_bn': 4096,
        'vgg13_bn': 4096,
        'vgg16_bn': 4096,
        'vgg19_bn': 4096
    }
    
    def __init__(self, 
                 model_name: str = "vgg16", 
                 device: str = "cpu",
                 batch_size: int = 16,
                 embed_dim: int = 768,
                 enable_mixed_precision: bool = False): 
        """
        Initialize VGG feature extractor with enhanced capabilities.
        
        Args:
            model_name: VGG model variant to use (vgg11, vgg13, vgg16, vgg19, with/without bn)
            device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.)
            batch_size: Default batch size for batch processing
            embed_dim: Output embedding dimension after projection
            enable_mixed_precision: Use automatic mixed precision (AMP)
        """
        super().__init__(model_name=model_name, device=device)
        
        # Validate model name
        if model_name not in self.SUPPORTED_MODELS:
            warnings.warn(f"Model {model_name} not in validated list. Supported: {list(self.SUPPORTED_MODELS.keys())}")
        
        # Configuration
        self.batch_size = batch_size
        self.enable_mixed_precision = enable_mixed_precision and device.startswith('cuda')
        
        # Setup logging
        self.logger = create_logger(job_name="vgg_extractor")

        # Load model and move to specified device
        self.model = self.load_model().to(device=device)

        # Setup layer projection
        self.embed_dim = embed_dim
        input_dim = self.SUPPORTED_MODELS.get(model_name, 4096)
        self.use_projection = embed_dim != self.SUPPORTED_MODELS.get(model_name, 4096)
        if self.use_projection:
          self.projection = torch.nn.Linear(in_features=input_dim, 
                                            out_features=self.embed_dim).to(device=device)
          torch.nn.init.xavier_uniform_(self.projection.weight)
          torch.nn.init.zeros_(self.projection.bias)
        
        # Setup preprocessing pipeline
        self.config = resolve_data_config(args={}, model=self.model)
        self.processor = create_transform(**self.config)
        
        # Setup mixed precision if enabled
        if self.enable_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info(f"Initialized {model_name} on {device}, mixed_precision: {self.enable_mixed_precision})")

    def load_model(self) -> torch.nn.Module:
      """Load pretrained VGG model with validation."""
      try:
        self.logger.info(f"Loading VGG model: {self.model_name}")
        model = timm.create_model(
            model_name=self.model_name, 
            pretrained=True, 
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global average pooling
            drop_rate=0.0  # Disable dropout for inference
        )
        
        # Validate model architecture
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            test_output = model(test_input)
        
        expected_dim = self.SUPPORTED_MODELS.get(self.model_name, 4096)
        actual_dim = test_output.shape[-1]
        
        if actual_dim != expected_dim:
            self.logger.warning(f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        self.logger.info(f"Model loaded successfully. Feature dim: {actual_dim}")
        return model.eval()
        
      except Exception as e:
        self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _preprocess_single_impl(self, image_path: str) -> torch.Tensor:
        """Implementation of single image preprocessing (cacheable)."""
        image = self._load_image(image_input=image_path)
        image_processed = self.processor(image)
        return image_processed.to(self.device)
    
    def preprocess_batch(self, data_list: List[Any]) -> torch.Tensor:
        """Preprocess a batch of images efficiently."""
        try:
            batch_tensors = []
            
            for data in data_list:
                # Direct preprocessing for PIL images or numpy arrays
                image = self._load_image(image_input=data)
                processed = self.processor(image).to(self.device)
                batch_tensors.append(processed)
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensors)
            self.logger.debug(f"Preprocessed batch of {len(data_list)} images")
            return batch_tensor
            
        except Exception as e:
            raise ValueError(f"Error preprocessing batch: {str(e)}")
    
    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess single image data for VGG model.
        
        Args:
            data: Image file path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor ready for model inference
        """
        try:
            # Direct preprocessing
            image = self._load_image(image_input=data)
            image_processed = self.processor(image).to(self.device)
            
            # Add batch dimension
            return image_processed.unsqueeze(0)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {data}")
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {data}: {str(e)}")
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def extract_features(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from preprocessed image tensor.
        
        Args:
            input_tensor: Preprocessed image tensor (single image or batch)
            
        Returns:
            Feature vector(s) as numpy array
        """
        try:
            self.model.eval()
            
            with torch.no_grad():
              if self.enable_mixed_precision:
                  with torch.cuda.amp.autocast():
                    output = self.model(input_tensor)
              else:
                  output = self.model(input_tensor)
            
            if self.use_projection: 
                output = self.projection(output)
              
            # Handle both single image and batch
            if output.shape[0] == 1:  # Single image in batch format
                features = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
            else:  # Batch result
                features = output.detach().cpu().numpy()  # Keep batch dimension
            
            self.logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction error: {str(e)}")
    
    def get_batch_embeddings(self, data_list: List[Any], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Extract features from a batch of images with optimized processing.
        
        Args:
            data_list: List of image data (paths, PIL Images, or numpy arrays)
            batch_size: Batch size for processing (uses default if None)
            
        Returns:
            Array of feature embeddings
        """
        if not data_list:
            self.logger.warning(f"No input image to embedding")
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(data_list), batch_size):
          batch_data = data_list[i:i + batch_size]
          
          try:
            # Preprocess batch
            batch_tensor = self.preprocess_batch(batch_data)
            
            # Extract features
            batch_features = self.extract_features(batch_tensor)
            
            # Normalize features
            if batch_features.ndim == 1:
                batch_features = batch_features.reshape(1, -1)
            
            normalized_features = self._normalize_features(batch_features)
            all_embeddings.append(normalized_features)
            
            self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(data_list) + batch_size - 1)//batch_size}")
              
          except Exception as e:
              self.logger.error(f"Batch processing failed for batch starting at {i}: {str(e)}")
              # Continue with next batch
              continue
        
        if not all_embeddings:
            raise RuntimeError("No batches were processed successfully")
        
        return np.vstack(all_embeddings)
    
    def get_embedding(self, data: Any) -> np.ndarray:
      if not data:
          self.logger.warning(f"No input image to embedding")
          return np.array([])
      try: 
        # Preprocessing
        preprocessed_data = self.preprocess(data)
        # Extract feature from image
        features = self.extract_features(preprocessed_data)
        self.logger.info(f"Processed image for embedding")
        return self._normalize_features(features)
      except Exception as e: 
          self.logger.error(f"Failed to embedding single image: {str(e)}")
    
    @property
    def feature_dim(self) -> dict:
        """Get the dimension of extracted features."""
        return  {
            "model_dim": self.SUPPORTED_MODELS.get(self.model_name, 4096), 
            "projection_dim": self.embed_dim 
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        return {
            **base_info,
            'batch_size': self.batch_size,
            'mixed_precision': self.enable_mixed_precision,
            'supported_models': list(self.SUPPORTED_MODELS.keys()),
            'memory_usage': torch.cuda.memory_allocated(self.device) if self.device.startswith('cuda') else 'N/A'
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'logger'):
            self.logger.info("VGGExtractor instance destroyed")