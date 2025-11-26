import os
import json
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

class PneumoniaDetector:
    """
    A scikit-learn based model for detecting pneumonia from chest X-ray images.
    """
    
    def __init__(self, model_type='random_forest', n_estimators=100):
        """
        Initialize the PneumoniaDetector.
        
        Args:
            model_type (str): Type of model to use ('random_forest' or 'svm')
            n_estimators (int): Number of trees in the random forest
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.class_names = None
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.model)
        ])
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            dict: Training history
        """
        self.pipeline.fit(X_train, y_train)
        
        history = {}
        if X_val is not None and y_val is not None:
            val_score = self.pipeline.score(X_val, y_val)
            history['val_accuracy'] = val_score
            print(f"Validation accuracy: {val_score:.4f}")
            
        train_score = self.pipeline.score(X_train, y_train)
        history['train_accuracy'] = train_score
        print(f"Training accuracy: {train_score:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            tuple: (predicted_class, confidence, probabilities)
        """
        probabilities = self.pipeline.predict_proba(X)
        predicted_class = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        return predicted_class, confidence, probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)
        
        # Classification report
        report = classification_report(
            y_test, 
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save the model to a file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'class_names': self.class_names,
            'model_data': {
                'pipeline': self.pipeline
            }
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model.
        """
        model_data = joblib.load(filepath)
        model = cls(
            model_type=model_data['model_type'],
            n_estimators=model_data['n_estimators']
        )
        model.class_names = model_data['class_names']
        model.pipeline = model_data['model_data']['pipeline']
        return model