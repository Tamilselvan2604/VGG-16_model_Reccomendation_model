import numpy as np
import joblib
import faiss
import os
from model import FashionRecommender

def test_similarity(query_path):
    """Test similarity search from command line"""
    # Load assets
    index = faiss.read_index("fashion_index.faiss")
    image_paths = joblib.load("image_paths.pkl")
    recommender = FashionRecommender()
    
    # Process query
    features = recommender.extract_features(query_path)
    if features is None:
        print("Failed to process query image")
        return
    
    # Search
    distances, indices = index.search(np.array([features.astype('float32')]), 3)
    
    print("\nQuery image:", query_path)
    print("Most similar items:")
    for i, idx in enumerate(indices[0]):
        if idx < len(image_paths):
            print(f"{i+1}. {image_paths[idx]} (Distance: {distances[0][i]:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Fashion Recommender')
    
    # Use raw string and proper path handling
    default_path = os.path.normpath(r"Fashion_update\static\images\1164.jpg")
    
    parser.add_argument('--image', 
                      type=str, 
                      default=default_path,
                      help='Path to test image (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Verify path exists
    if not os.path.exists(args.image):
        print(f"Error: File not found - {args.image}")
    else:
        test_similarity(args.image)