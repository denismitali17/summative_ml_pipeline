import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import threading
import shutil
from pathlib import Path

# Import our custom modules
from src.model import PneumoniaDetector as WildlifeClassifier
from src.data_loader import DataLoader

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates')
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload and model folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global variables for model and training status
MODEL = None
CLASS_NAMES = []
MODEL_VERSION = None
IS_TRAINING = False
TRAINING_STATUS = {
    'status': 'idle',  # 'idle', 'training', 'completed', 'error'
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None,
    'metrics': None
}

# Model metrics
model_metrics = {
    'version': None,
    'last_trained': None,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1_score': 0.0,
    'num_classes': 0,
    'class_names': []
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_latest_model():
    """Load the most recent model from the models directory."""
    global MODEL, CLASS_NAMES, MODEL_VERSION, model_metrics
    
    # Find all model files
    model_files = list(Path(app.config['MODEL_FOLDER']).glob('pneumonia_detector_*.pkl'))
    
    if not model_files:
        print("No trained models found. Please train a model first.")
        return False
    
    # Get the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    
    try:
        # Load the model
        MODEL = WildlifeClassifier.load(str(latest_model))
        MODEL_VERSION = latest_model.stem.replace('pneumonia_detector_', '')
        
        # Update model metrics
        model_metrics.update({
            'version': MODEL_VERSION,
            'num_classes': len(MODEL.class_names) if hasattr(MODEL, 'class_names') else 0,
            'class_names': MODEL.class_names if hasattr(MODEL, 'class_names') else []
        })
        
        # Try to load metrics if they exist
        metrics_file = latest_model.with_name(f"training_metrics_{MODEL_VERSION}.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                model_metrics.update({
                    'accuracy': metrics.get('accuracy', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'f1_score': metrics.get('f1_score', 0.0),
                    'last_trained': os.path.getmtime(str(latest_model))
                })
        
        print(f"Loaded model: {latest_model}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def train_model_async(data_dir, epochs=10, learning_rate=1e-4):
    """Train the model in a separate thread."""
    global MODEL, CLASS_NAMES, MODEL_VERSION, TRAINING_STATUS, model_metrics, IS_TRAINING
    
    try:
        TRAINING_STATUS.update({
            'status': 'training',
            'progress': 0,
            'message': 'Preparing data...',
            'start_time': datetime.now().isoformat(),
            'metrics': None
        })
        
        # Initialize data loader
        data_loader = DataLoader(
            data_dir=data_dir,
            img_size=(224, 224),
            test_size=0.2,
            val_size=0.2
        )
        
        # Load the data
        X_train, X_val, X_test, y_train, y_val, y_test, class_indices = data_loader.load_data()
        CLASS_NAMES = [k for k in sorted(class_indices, key=class_indices.get)]
        
        # Flatten the images for Random Forest
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Initialize the model
        TRAINING_STATUS['message'] = 'Initializing model...'
        MODEL = WildlifeClassifier(
            model_type='random_forest',
            n_estimators=100
        )
        
        # Train the model
        TRAINING_STATUS['message'] = 'Training model...'
        history = MODEL.train(X_train, y_train, X_val, y_val)
        
        # Evaluate the model
        TRAINING_STATUS['message'] = 'Evaluating model...'
        metrics = MODEL.evaluate(X_test, y_test)
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"pneumonia_detector_{timestamp}.pkl")
        MODEL.save(model_path)
        
        # Save class indices
        class_indices_path = os.path.join(app.config['MODEL_FOLDER'], f"class_indices_{timestamp}.json")
        with open(class_indices_path, 'w') as f:
            json.dump(class_indices, f)
        
        # Save metrics
        metrics_path = os.path.join(app.config['MODEL_FOLDER'], f"training_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }, f)
        
        # Update model metrics
        model_metrics.update({
            'version': timestamp,
            'last_trained': datetime.now().isoformat(),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES
        })
        
        TRAINING_STATUS.update({
            'status': 'completed',
            'progress': 100,
            'message': 'Training completed successfully',
            'end_time': datetime.now().isoformat(),
            'metrics': model_metrics
        })
        
    except Exception as e:
        TRAINING_STATUS.update({
            'status': 'error',
            'message': str(e),
            'end_time': datetime.now().isoformat()
        })
        print(f"Error during training: {e}")
    finally:
        IS_TRAINING = False

# Load the latest model on startup
load_latest_model()

# Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', 
                         classes=CLASS_NAMES,
                         metrics=model_metrics,
                         training_status=TRAINING_STATUS)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read and preprocess the image
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, -1)  # Flatten the image
            
            # Make prediction
            if MODEL is not None:
                class_idx, confidence, _ = MODEL.predict(img_array)
                predicted_class = CLASS_NAMES[class_idx] if CLASS_NAMES and class_idx < len(CLASS_NAMES) else str(class_idx)
                
                return jsonify({
                    'status': 'success',
                    'prediction': {
                        'class': predicted_class,
                        'confidence': float(confidence),
                        'class_id': int(class_idx),
                        'timestamp': datetime.now().isoformat()
                    },
                    'model_version': MODEL_VERSION
                })
            else:
                return jsonify({'error': 'Model not loaded'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """API endpoint for retraining the model with new data."""
    global IS_TRAINING
    
    if IS_TRAINING:
        return jsonify({
            'status': 'busy',
            'message': 'Model is already being trained',
            'training_status': TRAINING_STATUS
        }), 409  # Conflict status code
    
    # Check if training data is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No training data provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if it's a zip file
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Training data must be a zip file'}), 400
    
    try:
        # Create a temporary directory for the training data
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Get parameters from request
        epochs = int(request.form.get('epochs', 10))
        learning_rate = float(request.form.get('learning_rate', 1e-4))
        
        # Start training in a separate thread
        IS_TRAINING = True
        thread = threading.Thread(
            target=train_model_async,
            args=(temp_dir, epochs, learning_rate)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Model retraining started',
            'training_id': id(thread)
        })
        
    except Exception as e:
        IS_TRAINING = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """API endpoint for checking training status."""
    return jsonify({
        'model_status': {
            'loaded': MODEL is not None,
            'version': MODEL_VERSION,
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES
        },
        'training_status': TRAINING_STATUS,
        'metrics': model_metrics
    })

@app.route('/api/classes')
def get_classes():
    """API endpoint for getting available classes."""
    return jsonify({
        'classes': CLASS_NAMES,
        'count': len(CLASS_NAMES)
    })

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)