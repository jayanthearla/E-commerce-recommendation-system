"""
FAISS IVF index building for efficient similarity search
"""
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple

class IndexBuilder:
    """Build and manage FAISS IVF index"""
    
    def __init__(self, n_regions: int = 100, nprobe: int = 10):
        """
        Initialize index builder
        
        Args:
            n_regions: Number of IVF regions (clusters)
            nprobe: Number of regions to probe during search
        """
        self.n_regions = n_regions
        self.nprobe = nprobe
    
    def build_index(
        self,
        features: np.ndarray,
        image_paths: List[str],
        metadata: Dict,
        index_path: str,
        metadata_path: str
    ) -> Dict:
        """
        Build FAISS IVF index from features
        
        Args:
            features: Feature matrix (N x D)
            image_paths: List of image paths
            metadata: Additional metadata dict
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata JSON
            
        Returns:
            Dictionary with index statistics
        """
        num_images, feature_dim = features.shape
        
        print(f"Building FAISS IVF index for {num_images} images...")
        
        # Adjust n_regions if we have fewer images
        actual_n_regions = min(self.n_regions, num_images // 10)
        if actual_n_regions < 1:
            actual_n_regions = 1
        
        # Create IVF index with L2 distance
        quantizer = faiss.IndexFlatL2(feature_dim)
        index = faiss.IndexIVFFlat(quantizer, feature_dim, actual_n_regions)
        
        # Train the index
        print("Training index...")
        index.train(features)
        
        # Add vectors to index
        print("Adding vectors to index...")
        index.add(features)
        
        # Set nprobe
        index.nprobe = min(self.nprobe, actual_n_regions)
        
        # Save index
        faiss.write_index(index, index_path)
        print(f"Index saved to {index_path}")
        
        # Prepare metadata
        full_metadata = {
            'num_images': num_images,
            'feature_dim': feature_dim,
            'n_regions': actual_n_regions,
            'nprobe': index.nprobe,
            'device': metadata.get('device', 'unknown'),
            'image_paths': image_paths
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
        
        stats = {
            'num_images': num_images,
            'feature_dim': feature_dim,
            'n_regions': actual_n_regions,
            'nprobe': index.nprobe
        }
        
        return stats

