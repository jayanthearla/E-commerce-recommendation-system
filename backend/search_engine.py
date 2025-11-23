"""
Search engine for finding similar images using FAISS
"""
import faiss
import numpy as np
import json
import os
from typing import List, Dict
from backend.feature_extractor import FeatureExtractor

class SearchEngine:
    """Search for similar images using FAISS index"""
    
    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize search engine
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
        """
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.image_paths = self.metadata['image_paths']
        self.feature_extractor = FeatureExtractor()
        
        print(f"Loaded index with {len(self.image_paths)} images")
    
    def search(self, query_image_path: str, k: int = 10) -> List[Dict]:
        """
        Search for similar images
        
        Args:
            query_image_path: Path to query image
            k: Number of results to return
            
        Returns:
            List of result dictionaries with image_path, similarity, distance, etc.
        """
        # Extract features from query image
        query_features = self.feature_extractor.extract_features(query_image_path)
        
        if query_features is None:
            raise ValueError("Failed to extract features from query image")
        
        # Reshape for FAISS (1 x D)
        query_features = query_features.reshape(1, -1).astype('float32')
        
        # Search
        k = min(k, len(self.image_paths))
        distances, indices = self.index.search(query_features, k)
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.image_paths):
                continue
            
            image_path = self.image_paths[idx]
            
            # Convert L2 distance to similarity (using cosine similarity since features are normalized)
            # For normalized vectors: cosine_sim = 1 - (L2^2 / 2)
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            result = {
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'similarity': float(similarity),
                'distance': float(distance),
                'rank': i + 1,
                'feature_stats': {
                    'feature_dim': self.metadata['feature_dim'],
                    'index_type': 'IVF'
                }
            }
            
            results.append(result)
        
        return results

