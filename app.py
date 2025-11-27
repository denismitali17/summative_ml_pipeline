import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance
import io
import threading
import shutil
from pathlib import Path
import joblib

from src.model import PneumoniaDetector as WildlifeClassifier
from src.data_loader import DataLoader

# Feature extraction utilities (same as notebook)
class FeatureExtractor:
    @staticmethod
    def extract_histogram(image, bins=32):
        """Extract color histogram features using PIL."""
        hist = np.histogram(np.array(image).flatten(), bins=bins, range=(0, 256))[0]
        return hist / hist.sum()
    
    @staticmethod
    def extract_edges(image):
        """Extract edge features using PIL's edge detection."""
        edges = image.filter(ImageFilter.FIND_EDGES)
        return np.array(edges).flatten() / 255.0
    
    @staticmethod
    def extract_texture(image):
        """Extract texture features using PIL's edge enhancement."""
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(2.0)
        return np.array(enhanced).flatten() / 255.0
    
    @staticmethod
    def extract_all_features_from_image(image, target_size=(224, 224)):
        """Extract all features from a PIL Image object."""
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('L')
            else:
                image = image.convert('L') if hasattr(image, 'convert') else image
            
            if image.size != target_size:
                image = image.resize(target_size)
            
            hist = FeatureExtractor.extract_histogram(image)
            edges = FeatureExtractor.extract_edges(image)
            texture = FeatureExtractor.extract_texture(image)
            
            features = np.hstack([
                hist,
                edges[:1000],
                texture[:1000],
                np.array(image).flatten() / 255.0
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise

app = Flask(__name__, template_folder='app/templates', static_folder='app/static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'zip'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

MODEL = None
CLASS_NAMES = []
MODEL_VERSION = None
IS_TRAINING = False
TRAINING_STATUS = {
    'status': 'idle',  
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None,
    'metrics': None
}

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
    # Gather candidate model files (new and legacy)
    candidate_models = []
    model_folder = Path(app.config['MODEL_FOLDER'])
    for p in model_folder.glob('pneumonia_detector_*.pkl'):
        candidate_models.append(p)
    legacy_model = model_folder / 'pneumonia_rf_model.pkl'
    if legacy_model.exists():
        candidate_models.append(legacy_model)

    if not candidate_models:
        print("No trained models found. Please train a model first.")
        return False

    # Sort candidates by creation time (newest first)
    candidate_models = sorted(candidate_models, key=os.path.getctime, reverse=True)

    # Try to pick the newest model that has >=2 classes
    selected_model = None
    selected_num_classes = 0
    for candidate in candidate_models:
        try:
            # For legacy model, inspect joblib contents
            if candidate.name == 'pneumonia_rf_model.pkl':
                md = joblib.load(str(candidate))
                classes = md.get('classes') or md.get('classes_') or md.get('label_encoder')
                if isinstance(classes, (list, tuple)):
                    num = len(classes)
                elif hasattr(classes, 'classes_'):
                    num = len(getattr(classes, 'classes_', []))
                else:
                    num = 0
            else:
                # For new models, check for class_indices JSON first
                version = candidate.stem.replace('pneumonia_detector_', '')
                class_indices_file = candidate.with_name(f"class_indices_{version}.json")
                if class_indices_file.exists():
                    with open(class_indices_file, 'r') as f:
                        ci = json.load(f)
                        num = len(ci.keys())
                else:
                    # Fallback: attempt to load the model and check attribute
                    try:
                        temp_model = WildlifeClassifier.load(str(candidate))
                        cn = getattr(temp_model, 'class_names', None)
                        num = len(cn) if cn else 0
                    except Exception:
                        num = 0

            if num >= 2:
                selected_model = candidate
                selected_num_classes = num
                break
        except Exception:
            # If anything fails for this candidate, skip it
            continue

    # If no multi-class model found, fall back to newest model
    if selected_model is None:
        selected_model = candidate_models[0]
        print(f"No multi-class model found; falling back to newest model: {selected_model.name}")
    else:
        print(f"Selected multi-class model: {selected_model.name} with {selected_num_classes} classes")

    latest_model = selected_model

    try:
        # Handle legacy model format
        if latest_model.name == 'pneumonia_rf_model.pkl':
            print(f"Loading legacy model format: {latest_model}")
            model_data = joblib.load(str(latest_model))
            
            # Create a wrapper to handle the legacy model
            class LegacyModelWrapper:
                def __init__(self, model_data):
                    self.model = model_data.get('model')
                    self.label_encoder = model_data.get('label_encoder')
                    self.classes_ = model_data.get('classes')
                    self.feature_importances_ = model_data.get('feature_importances')
                    self.class_names = list(self.classes_) if self.classes_ is not None else ['NORMAL', 'PNEUMONIA']
                    self.scaler = None
                
                def predict(self, X):
                    predictions = self.model.predict(X)
                    probabilities = self.model.predict_proba(X)
                    predicted_class = predictions
                    confidence = np.max(probabilities, axis=1)
                    return predicted_class, confidence, probabilities
                
                def save(self, path):
                    joblib.dump({
                        'model': self.model,
                        'label_encoder': self.label_encoder,
                        'classes': self.classes_,
                        'feature_importances': self.feature_importances_
                    }, path)
                
                @staticmethod
                def load(path):
                    model_data = joblib.load(path)
                    return LegacyModelWrapper(model_data)
            
            MODEL = LegacyModelWrapper(model_data)
            MODEL_VERSION = 'legacy_' + datetime.fromtimestamp(os.path.getctime(str(latest_model))).strftime("%Y%m%d_%H%M%S")
            CLASS_NAMES = MODEL.class_names
        else:
            # Handle new model format — but first check file contents in case a legacy dict
            try:
                maybe_data = joblib.load(str(latest_model))
            except Exception:
                maybe_data = None

            # If the loaded artifact looks like the legacy dict, wrap it
            if isinstance(maybe_data, dict) and 'model' in maybe_data:
                print(f"Detected legacy-format contents in {latest_model.name}, wrapping as legacy model")
                model_data = maybe_data
                class LegacyModelWrapper:
                    def __init__(self, model_data):
                        self.model = model_data.get('model')
                        self.label_encoder = model_data.get('label_encoder')
                        self.classes_ = model_data.get('classes')
                        self.feature_importances_ = model_data.get('feature_importances')
                        self.class_names = list(self.classes_) if self.classes_ is not None else ['NORMAL', 'PNEUMONIA']
                        self.scaler = None

                    def predict(self, X):
                        predictions = self.model.predict(X)
                        probabilities = self.model.predict_proba(X)
                        predicted_class = predictions
                        confidence = np.max(probabilities, axis=1)
                        return predicted_class, confidence, probabilities

                    def save(self, path):
                        joblib.dump({
                            'model': self.model,
                            'label_encoder': self.label_encoder,
                            'classes': self.classes_,
                            'feature_importances': self.feature_importances_
                        }, path)

                    @staticmethod
                    def load(path):
                        model_data = joblib.load(path)
                        return LegacyModelWrapper(model_data)

                MODEL = LegacyModelWrapper(model_data)
                MODEL_VERSION = latest_model.stem.replace('pneumonia_detector_', '')
                CLASS_NAMES = MODEL.class_names
            else:
                # Normal new-model loading
                MODEL = WildlifeClassifier.load(str(latest_model))
                MODEL_VERSION = latest_model.stem.replace('pneumonia_detector_', '')
                CLASS_NAMES = getattr(MODEL, 'class_names', [])
            if CLASS_NAMES is None or len(CLASS_NAMES) == 0:
                # Try to load class names from class_indices file
                class_indices_file = latest_model.with_name(f"class_indices_{MODEL_VERSION}.json")
                if class_indices_file.exists():
                    try:
                        with open(class_indices_file, 'r') as f:
                            class_indices = json.load(f)
                            CLASS_NAMES = [k for k in sorted(class_indices, key=class_indices.get)]
                            MODEL.class_names = CLASS_NAMES
                    except Exception as e:
                        print(f"Error loading class_indices: {e}")
                        CLASS_NAMES = []
                else:
                    # Fallback to default classes
                    CLASS_NAMES = []
        
        # Update model metrics - ensure CLASS_NAMES is a list
        if CLASS_NAMES is None:
            CLASS_NAMES = []

        # Normalize class name capitalization for display (e.g., 'NORMAL' -> 'Normal')
        try:
            CLASS_NAMES = [str(x).capitalize() for x in CLASS_NAMES]
        except Exception:
            pass
        
        model_metrics.update({
            'version': MODEL_VERSION,
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES
        })
        
        # Try to load metrics file if it exists
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
        print(f"Model classes: {CLASS_NAMES}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model_async(data_dir, epochs=10, learning_rate=1e-4):
    """Train the model in a separate thread."""
    global MODEL, CLASS_NAMES, MODEL_VERSION, TRAINING_STATUS, model_metrics, IS_TRAINING
    
    try:
        print(f"[DEBUG] Starting async training with data_dir: {data_dir}")
        TRAINING_STATUS.update({
            'status': 'training',
            'progress': 10,
            'message': 'Preparing data...',
            'start_time': datetime.now().isoformat(),
            'metrics': None
        })
        
        print(f"[DEBUG] Loading data from {data_dir}")
        data_loader = DataLoader(
            data_dir=data_dir,
            img_size=(224, 224),
            test_size=0.2,
            val_size=0.2
        )
        
        print(f"[DEBUG] Loading dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test, class_indices = data_loader.load_data()
        print(f"[DEBUG] Data loaded - X_train shape: {X_train.shape}, class_indices: {class_indices}")
        
        # Extract class names from class_indices
        class_names_list = [k for k in sorted(class_indices, key=class_indices.get)]
        print(f"[DEBUG] Class names: {class_names_list}")
        
        TRAINING_STATUS['progress'] = 25
        TRAINING_STATUS['message'] = 'Flattening features...'
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        print(f"[DEBUG] Features flattened - X_train shape: {X_train.shape}")
        
        TRAINING_STATUS['progress'] = 40
        TRAINING_STATUS['message'] = 'Initializing model...'
        print(f"[DEBUG] Initializing model...")
        
        MODEL = WildlifeClassifier(
            model_type='random_forest',
            n_estimators=100
        )
        
        # Set class names on the model
        MODEL.class_names = class_names_list
        
        TRAINING_STATUS['progress'] = 50
        TRAINING_STATUS['message'] = 'Training model...'
        print(f"[DEBUG] Training model...")
        history = MODEL.train(X_train, y_train, X_val, y_val)
        print(f"[DEBUG] Model trained - history: {history}")
        
        TRAINING_STATUS['progress'] = 75
        TRAINING_STATUS['message'] = 'Evaluating model...'
        print(f"[DEBUG] Evaluating model...")
        metrics = MODEL.evaluate(X_test, y_test)
        print(f"[DEBUG] Metrics: {metrics}")
        
        # Save the model
        TRAINING_STATUS['progress'] = 85
        TRAINING_STATUS['message'] = 'Saving model...'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"pneumonia_detector_{timestamp}.pkl")
        print(f"[DEBUG] Saving model to {model_path}")
        MODEL.save(model_path)
        
        # Save class indices
        class_indices_path = os.path.join(app.config['MODEL_FOLDER'], f"class_indices_{timestamp}.json")
        with open(class_indices_path, 'w') as f:
            json.dump(class_indices, f)
        print(f"[DEBUG] Class indices saved")
        
        # Save metrics
        metrics_path = os.path.join(app.config['MODEL_FOLDER'], f"training_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'accuracy': float(metrics['accuracy']) if metrics['accuracy'] is not None else 0.0,
                'precision': float(metrics['precision']) if metrics['precision'] is not None else 0.0,
                'recall': float(metrics['recall']) if metrics['recall'] is not None else 0.0,
                'f1_score': float(metrics['f1_score']) if metrics['f1_score'] is not None else 0.0
            }, f)
        print(f"[DEBUG] Metrics saved")
        
        # Update model metrics
        model_metrics.update({
            'version': timestamp,
            'last_trained': datetime.now().isoformat(),
            'accuracy': float(metrics['accuracy']) if metrics['accuracy'] is not None else 0.0,
            'precision': float(metrics['precision']) if metrics['precision'] is not None else 0.0,
            'recall': float(metrics['recall']) if metrics['recall'] is not None else 0.0,
            'f1_score': float(metrics['f1_score']) if metrics['f1_score'] is not None else 0.0,
            'num_classes': len(class_names_list),
            'class_names': class_names_list
        })
        
        # Update global CLASS_NAMES
        CLASS_NAMES = class_names_list
        
        TRAINING_STATUS.update({
            'status': 'completed',
            'progress': 100,
            'message': 'Training completed successfully',
            'end_time': datetime.now().isoformat(),
            'metrics': model_metrics
        })
        print(f"[DEBUG] Training completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        TRAINING_STATUS.update({
            'status': 'error',
            'message': str(e),
            'end_time': datetime.now().isoformat()
        })
    finally:
        IS_TRAINING = False
        print(f"[DEBUG] Training thread finished")

# Load the latest model on startup
load_latest_model()

# Routes
@app.route('/')
def index():
    """Render the main page."""
    # Ensure CLASS_NAMES is not None
    class_names = CLASS_NAMES if CLASS_NAMES is not None else []
    return render_template('index.html', 
                         classes=class_names,
                         metrics=model_metrics,
                         training_status=TRAINING_STATUS)

@app.route('/test')
def test():
    """Simple test endpoint to verify routing."""
    return jsonify({'status': 'ok', 'message': 'Flask app is working'})

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    print(f"[DEBUG] Predict endpoint called")
    print(f"[DEBUG] Request files: {request.files}")
    
    if 'file' not in request.files:
        print(f"[DEBUG] No file in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    print(f"[DEBUG] File name: {file.filename}")
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            print(f"[DEBUG] Processing file: {file.filename}")
            # Read the image - use RGB like in training
            img = Image.open(file.stream).convert('RGB')
            print(f"[DEBUG] Image opened, size: {img.size}")
            
            # Resize to 224x224 like in training
            img = img.resize((224, 224))

            # Prepare two candidate feature vectors:
            # 1) Grayscale feature vector from FeatureExtractor (used by older pipeline)
            gray_img = img.convert('L')
            try:
                gray_feats = FeatureExtractor.extract_all_features_from_image(gray_img, target_size=(224, 224))
                gray_feats = np.array(gray_feats).reshape(1, -1)
                print(f"[DEBUG] Grayscale features shape: {gray_feats.shape}")
            except Exception as e:
                print(f"[DEBUG] Error extracting grayscale features: {e}")
                gray_feats = None

            # 2) RGB flattened features (used by other trainings)
            try:
                rgb_array = np.array(img) / 255.0
                rgb_feats = rgb_array.reshape(1, -1)
                print(f"[DEBUG] RGB flattened features shape: {rgb_feats.shape}")
            except Exception as e:
                print(f"[DEBUG] Error extracting RGB features: {e}")
                rgb_feats = None

            # Determine expected number of features from the loaded model (if available)
            expected_features = None
            if MODEL is not None:
                if hasattr(MODEL, 'model') and hasattr(MODEL.model, 'n_features_in_'):
                    expected_features = getattr(MODEL.model, 'n_features_in_', None)
                elif hasattr(MODEL, 'n_features_in_'):
                    expected_features = getattr(MODEL, 'n_features_in_', None)
            print(f"[DEBUG] Model expected features: {expected_features}")

            # Choose which feature vector to use
            chosen = None
            if expected_features is not None:
                if gray_feats is not None and gray_feats.shape[1] == expected_features:
                    chosen = gray_feats
                    print("[DEBUG] Using grayscale features for prediction (matched expected size)")
                elif rgb_feats is not None and rgb_feats.shape[1] == expected_features:
                    chosen = rgb_feats
                    print("[DEBUG] Using RGB flattened features for prediction (matched expected size)")
            else:
                # No expected_features info — prefer grayscale extraction first
                if gray_feats is not None:
                    chosen = gray_feats
                    print("[DEBUG] No expected size; trying grayscale features first")
                elif rgb_feats is not None:
                    chosen = rgb_feats
                    print("[DEBUG] No expected size; falling back to RGB flattened features")

            # If still None, try to set chosen to whichever is available
            if chosen is None:
                chosen = gray_feats if gray_feats is not None else rgb_feats

            # Make prediction
            if MODEL is not None and chosen is not None:
                try:
                    print(f"[DEBUG] Making prediction with features shape: {chosen.shape}")
                    class_idx, confidence, _ = MODEL.predict(chosen)
                    print(f"[DEBUG] Prediction result - class_idx: {class_idx}, confidence: {confidence}")

                    predicted_class = CLASS_NAMES[class_idx[0]] if CLASS_NAMES and class_idx[0] < len(CLASS_NAMES) else str(class_idx[0])

                    result = {
                        'status': 'success',
                        'prediction': {
                            'class': predicted_class,
                            'confidence': float(confidence[0]),
                            'class_id': int(class_idx[0]),
                            'timestamp': datetime.now().isoformat()
                        },
                        'model_version': MODEL_VERSION
                    }
                    print(f"[DEBUG] Returning: {result}")
                    return jsonify(result)
                except ValueError as ve:
                    print(f"[ERROR] Prediction error (feature mismatch): {ve}")
                    return jsonify({'error': 'Feature size mismatch with model. Try retraining or using a different model.'}), 500
                except Exception as e:
                    print(f"[ERROR] Prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': str(e)}), 500
            else:
                if MODEL is None:
                    print(f"[DEBUG] Model is None")
                    return jsonify({'error': 'Model not loaded'}), 500
                else:
                    print(f"[DEBUG] No valid features extracted for prediction")
                    return jsonify({'error': 'Could not extract features for prediction'}), 400
                
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    print(f"[DEBUG] File type not allowed: {file.filename}")
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """API endpoint for retraining the model with new data."""
    global IS_TRAINING
    
    print(f"[DEBUG] Retrain endpoint called")
    print(f"[DEBUG] IS_TRAINING: {IS_TRAINING}")
    
    if IS_TRAINING:
        return jsonify({
            'status': 'busy',
            'message': 'Model is already being trained',
            'training_status': TRAINING_STATUS
        }), 409  
    
    # Check if training data is provided
    if 'file' not in request.files:
        print(f"[DEBUG] No file in request")
        return jsonify({'error': 'No training data provided'}), 400
    
    file = request.files['file']
    print(f"[DEBUG] File received: {file.filename}, size: {len(file.read())} bytes")
    file.seek(0)  # Reset file pointer
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if it's a zip file
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Training data must be a zip file'}), 400
    
    try:
        print(f"[DEBUG] Processing zip file: {file.filename}")
        # Create a temporary directory for the training data
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        print(f"[DEBUG] Created temp directory: {temp_dir}")
        
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        print(f"[DEBUG] Saved zip file to: {zip_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            print(f"[DEBUG] Extracted zip file")
        
        # Get parameters from request
        epochs = int(request.form.get('epochs', 10))
        learning_rate = float(request.form.get('learning_rate', 1e-4))
        print(f"[DEBUG] Training parameters - epochs: {epochs}, learning_rate: {learning_rate}")
        
        # Start training in a separate thread
        IS_TRAINING = True
        thread = threading.Thread(
            target=train_model_async,
            args=(temp_dir, epochs, learning_rate)
        )
        thread.daemon = True
        thread.start()
        print(f"[DEBUG] Training thread started")
        
        return jsonify({
            'status': 'started',
            'message': 'Model retraining started',
            'training_id': id(thread)
        })
        
    except Exception as e:
        print(f"[ERROR] Retrain error: {e}")
        import traceback
        traceback.print_exc()
        IS_TRAINING = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """API endpoint for checking training status."""
    print(f"[DEBUG] Status endpoint called - Training status: {TRAINING_STATUS}")
    # Ensure CLASS_NAMES is not None
    class_names = CLASS_NAMES if CLASS_NAMES is not None else []
    return jsonify({
        'model_status': {
            'loaded': MODEL is not None,
            'version': MODEL_VERSION,
            'num_classes': len(class_names),
            'class_names': class_names
        },
        'training_status': TRAINING_STATUS,
        'metrics': model_metrics
    })

@app.route('/api/classes')
def get_classes():
    """API endpoint for getting available classes."""
    # Ensure CLASS_NAMES is not None
    class_names = CLASS_NAMES if CLASS_NAMES is not None else []
    return jsonify({
        'classes': class_names,
        'count': len(class_names)
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