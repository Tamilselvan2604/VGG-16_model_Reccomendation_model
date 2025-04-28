import os
import numpy as np
import faiss
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class FashionRecommender:
    def __init__(self, dataset_dir="static/images/"):
        self.dataset_dir = dataset_dir
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.index = None
        self.image_paths = []

    def extract_features(self, img_path):
        """Extract CNN features from an image"""
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            features = self.model.predict(x.reshape(1, 224, 224, 3))
            return features.flatten()
        except Exception as e:
            print(f"Skipped {img_path}: {str(e)}")
            return None

    def build_index(self):
        """Create FAISS index from dataset"""
        valid_images = [f for f in os.listdir(self.dataset_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        features = []
        self.image_paths = []
        
        for idx, img_file in enumerate(valid_images):
            img_path = os.path.join(self.dataset_dir, img_file)
            feat = self.extract_features(img_path)
            
            if feat is not None:
                features.append(feat)
                self.image_paths.append(img_file)
                
            if (idx+1) % 100 == 0:
                print(f"Processed {idx+1}/{len(valid_images)} images")

        features = np.array(features, dtype='float32')
        self.index = faiss.IndexFlatL2(features.shape[1])
        self.index.add(features)
        
        # Save assets
        faiss.write_index(self.index, "fashion_index.faiss")
        joblib.dump(self.image_paths, "image_paths.pkl")
        print(f"Index built with {len(features)} images")

if __name__ == "__main__":
    recommender = FashionRecommender()
    recommender.build_index()