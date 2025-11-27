#!/usr/bin/env python
"""
Script to retrain the Pneumonia Prediction with new data.
"""
import os
import argparse
import json
import shutil
from datetime import datetime
import numpy as np
from model import WildlifeClassifier
from data_loader import DataLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Retrain Pneumonia Prediction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the existing model file (.h5)')
    parser.add_argument('--new_data_dir', type=str, required=True,
                        help='Path to the directory containing new training data')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the retrained model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for fine-tuning')
    return parser.parse_args()

def main():
    """Main function to retrain the model."""
    # Parse command line arguments
    args = parse_args()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the existing model
    print(f"Loading model from {args.model_path}...")
    model = WildlifeClassifier.load(args.model_path)
    
    # Initialize data loader with the new data
    print(f"Loading new training data from {args.new_data_dir}...")
    data_loader = DataLoader(
        data_dir=args.new_data_dir,
        img_size=(224, 224),
        test_size=0.1,  
        val_size=0.2    
    )
    
    # Load the data
    train_gen, val_gen, test_gen, class_indices = data_loader.load_data()
    
    
    new_class_names = [k for k in sorted(class_indices, key=class_indices.get)]
    if set(new_class_names) != set(model.class_names):
        print("Warning: New classes detected in the training data.")
        print(f"Previous classes: {model.class_names}")
        print(f"New classes: {new_class_names}")
        model.class_names = new_class_names
    
    # Print dataset information
    print("\nDataset information:")
    print(f"  - Number of classes: {len(model.class_names)}")
    print(f"  - Classes: {', '.join(model.class_names)}")
    print(f"  - Training samples: {train_gen.samples}")
    print(f"  - Validation samples: {val_gen.samples}")
    if test_gen is not None:
        print(f"  - Test samples: {test_gen.samples}")
    
    # Fine-tune the model with the new data
    print(f"\nFine-tuning model for {args.epochs} epochs...")
    history = model.fine_tune(
        train_gen,
        val_gen,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    
    if test_gen is not None:
        print("\nEvaluating on test data...")
        metrics = model.evaluate(test_gen)
        
        # Print evaluation metrics
        print("\nTest Metrics:")
        print(f"  - Loss: {metrics['loss']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1-Score: {metrics['f1_score']:.4f}")
    
    # Save the retrained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"wildlife_classifier_retrained_{timestamp}.h5")
    model.save(model_path)
    print(f"\nRetrained model saved to {model_path}")
    
    # Save the updated class indices
    class_indices_path = os.path.join(args.output_dir, f"class_indices_retrained_{timestamp}.json")
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Updated class indices saved to {class_indices_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"retraining_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Retraining history saved to {history_path}")
    
    # Return the paths to the saved files
    return {
        'model_path': model_path,
        'class_indices_path': class_indices_path,
        'history_path': history_path,
        'metrics': metrics if test_gen is not None else None
    }

if __name__ == '__main__':
    main()