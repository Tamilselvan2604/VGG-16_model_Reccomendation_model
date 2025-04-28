from flask import Flask, request, render_template, send_from_directory  # Add send_from_directory
import os
import numpy as np
import joblib
import faiss
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load AI components
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
index = faiss.read_index("fashion_index.faiss")
image_paths = joblib.load("image_paths.pkl")

def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    features = model.predict(x.reshape(1, 224, 224, 3))
    return features.flatten().astype('float32')

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/uploads/<filename>')  # New route to serve uploaded files
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = f"temp_{os.urandom(8).hex()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        query_feat = process_image(filepath)
        distances, indices = index.search(np.array([query_feat]), 5)
        
        results = [image_paths[i] for i in indices[0] if i < len(image_paths)]
        
        return render_template('results.html', 
                             query_image=filename,
                             results=results)
    
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)