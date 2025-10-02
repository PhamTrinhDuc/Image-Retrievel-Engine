from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Any, Optional
import torch
from PIL import Image

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
  
  @abstractmethod
  def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> Any:
      """
      Preprocess image data.
      
      Args:
          image: Image data (file path, PIL Image, or numpy array)
          
      Returns:
          Preprocessed image tensor
      """
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