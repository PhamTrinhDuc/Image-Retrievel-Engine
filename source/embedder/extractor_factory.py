import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Union
from embedder.resnet_extractor import ResNetExtractor
from embedder.dino_extractor import DINOv2Extractor
from embedder.vgg_extractor import VGGExtractor
from embedder.vit_extractor import ViTExtractor
from base.base_embedder import ImageFeatureExtractor


class EmbedderFactory:
    """Factory class to create image feature extractors"""
    
    @staticmethod
    def create_extractor(extractor_type: str, **kwargs) -> ImageFeatureExtractor:
        """
        Create a feature extractor
        
        Args:
            extractor_type: Type of feature extractor ('resnet', 'vgg', 'vit', 'dinov2', 'clip')
            **kwargs: Configuration parameters for the specific extractor
            
        Returns:
            ImageFeatureExtractor instance
        """
        if extractor_type.lower() == "resnet":
            return ResNetExtractor(**kwargs)
        elif extractor_type.lower() == "vgg":
            return VGGExtractor(**kwargs)
        elif extractor_type.lower() == "vit":
            return ViTExtractor(**kwargs)
        elif extractor_type.lower() == "dinov2":
            return DINOv2Extractor(**kwargs)
        else:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")