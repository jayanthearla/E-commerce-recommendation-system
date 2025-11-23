"""
Feature extraction using Vision Transformer (ViT-B/16)
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from typing import List, Tuple, Callable, Optional

class FeatureExtractor:
    """Extract image features using ViT-B/16"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the feature extractor
        
        Args:
            device: 'cuda' or 'cpu'. If None, auto-detect.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading ViT-B/16 model on {self.device}...")
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 768  # ViT-B/16 hidden size
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array (768-dim)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize features
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            return features[0]
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_from_directory(
        self,
        directory_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, List[str], dict]:
        """
        Extract features from all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            progress_callback: Function(current, total, current_image_path)
            
        Returns:
            Tuple of (features array, image_paths list, metadata dict)
        """
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        # Find all image files
        image_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if len(image_paths) == 0:
            return np.array([]), [], {}
        
        print(f"Found {len(image_paths)} images. Extracting features...")
        
        features_list = []
        valid_paths = []
        
        for idx, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(idx, len(image_paths), image_path)
            
            feature = self.extract_features(image_path)
            if feature is not None:
                features_list.append(feature)
                valid_paths.append(image_path)
        
        if len(features_list) == 0:
            return np.array([]), [], {}
        
        features_array = np.vstack(features_list).astype('float32')
        
        metadata = {
            'num_images': len(valid_paths),
            'feature_dim': self.feature_dim,
            'device': self.device
        }
        
        print(f"Extracted features from {len(valid_paths)} images")
        
        return features_array, valid_paths, metadata

