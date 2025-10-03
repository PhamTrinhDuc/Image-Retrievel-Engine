import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Any, Optional
import torch
import warnings
from PIL import Image
from timm.data.transforms_factory import create_transform
from timm.data.config import resolve_data_config
from utils.helpers import create_logger

class BaseFeatureExtractor(ABC): 
  """
  Abstract base class for feature extraction models.
  This class defines the interface for extracting features from different types of data
  to create embeddings for vector database storage.
  """
  def __init__(self, model_name: str, device: str="cpu"):
    """
    Initialize the feature extractor.
    
    Args:
        model_name (str): Name/identifier of the model
        device (str): Device to run the model on ('cpu' or 'cuda')
    """
    self.model_name = model_name
    self.device = device
    self.model = None
  
  @abstractmethod
  def load_model(self) -> None:
    """
    Load the pre-trained model.
    This method should be implemented by concrete classes.
    """ 
    pass

  @abstractmethod
  def preprocess(self, data: Any) -> Any: 
    """
    Preprocess input data before feature extraction.
    
    Args:
        data: Raw input data (image, text, etc.)
        
    Returns:
        Preprocessed data ready for the model
    """
    pass  

  @abstractmethod
  def extract_features(self, data: Any) -> np.ndarray: 
    """
    Extract features from preprocessed data.
    
    Args:
        data: Preprocessed input data
        
    Returns:
        Feature vector as numpy array
    """
    pass 

  def get_embedding(self, data: Any) -> np.ndarray:
        """
        Complete pipeline: preprocess data and extract features.
        
        Args:
            data: Raw input data
            
        Returns:
            Feature embedding as numpy array
        """
            
        preprocessed_data = self.preprocess(data)
        features = self.extract_features(preprocessed_data)
        
        return self._normalize_features(features)
    
  def get_batch_embeddings(self, data_list: List[Any]) -> np.ndarray:
      """
      Extract features from a batch of data.
      Args:
          data_list: List of raw input data
      Returns:
          Array of feature embeddings
      """
      embeddings = []
      for data in data_list:
          embedding = self.get_embedding(data)
          embeddings.append(embedding)
      
      return np.array(embeddings)
  
  def _normalize_features(self, features: np.ndarray) -> np.ndarray:
      """
      Normalize feature vectors (L2 normalization by default).
      Args:
          features: Raw feature vector
      Returns:
          Normalized feature vector
      """
      if len(features.shape) == 1:
          # Single vector
          norm = np.linalg.norm(features)
          return features / (norm + 1e-8)
      else:
          # Batch of vectors
          norms = np.linalg.norm(features, axis=1, keepdims=True)
          return features / (norms + 1e-8)
  
  @property
  def feature_dim(self) -> int:
      """
      Get the dimension of extracted features.
      Should be implemented by concrete classes.
      """
      raise NotImplementedError("Subclasses must implement feature_dim property")
  
  def get_model_info(self) -> dict:
      """
      Get information about the model.
      
      Returns:
          Dictionary containing model information
      """
      return {
          "model_name": self.model_name,
          "device": self.device,
          "feature_dim": self.feature_dim,
      }
  
  def __repr__(self) -> str:
      return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"


class ImageFeatureExtractor(BaseFeatureExtractor):
  """
  Abstract base class specifically for image feature extraction.
  """

  # Supported DINOv2 models with their feature dimensions
  SUPPORTED_MODELS = {}
  JOB_NAME = ""

  def __init__(self, 
                model_name: str, 
                device: str,
                batch_size: int,
                embed_dim: int,
                enable_mixed_precision: bool): 
    """
    Initialize pretrained model feature extractor with enhanced capabilities.
    
    Args:
        model_name: pretrained model variant to use
        device: Device for computation ('cpu', 'cuda', 'cuda:0', etc.)
        batch_size: Default batch size for batch processing
        embed_dim: Output embedding dimension
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
    self.logger = create_logger(job_name=self.JOB_NAME)

    # Load model and move to specified device
    self.model = self.load_model().to(device=device)

    # Setup layer projection
    self.embed_dim = embed_dim
    input_dim = self.SUPPORTED_MODELS.get(model_name, 768)
    self.use_projection = embed_dim != self.SUPPORTED_MODELS.get(model_name, 768)
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

  @abstractmethod
  def load_model(self) -> torch.nn.Module:
      """Load pretrained model. Must be implemented by subclasses."""
      pass
  
  def _load_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
      """
      Helper method to load image from different input types.
      
      Args:
          image_input: Image file path, PIL Image, or numpy array
          
      Returns:
          PIL Image object
      """
      if isinstance(image_input, str):
          return Image.open(image_input).convert('RGB')
      elif isinstance(image_input, Image.Image):
          return image_input.convert('RGB')
      elif isinstance(image_input, np.ndarray):
          return Image.fromarray(image_input).convert('RGB')
      else:
          raise ValueError(f"Unsupported image input type: {type(image_input)}")

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
      Preprocess single image data for DINOv2 model.
      
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
          if output.shape[0] == 1:  # Single image result
              features = output.squeeze().detach().cpu().numpy()
          else:  # Batch result
              features = output.squeeze().detach().cpu().numpy()
          
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
          "model_dim": self.SUPPORTED_MODELS.get(self.model_name, 768), 
          "projection_dim": self.embed_dim 
      }

  def get_model_info(self) -> dict[str, Any]:
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
          self.logger.info(f"{self.JOB_NAME} instance destroyed")


class TextFeatureExtractor(BaseFeatureExtractor):
  """
  Abstract base class specifically for text feature extraction.
  """
  
  @abstractmethod
  def preprocess(self, text: Union[str, List[str]]) -> Any:
      """
      Preprocess text data.
      
      Args:
          text: Text string or list of text strings
          
      Returns:
          Preprocessed text tokens/embeddings
      """
      pass