#!/usr/bin/env python
"""
Script to train the Pneumonia Detection model from chest X-ray images.
"""
import os
import argparse
import json
from datetime import datetime
import tensorflow as tf
from data_loader import load_dataset
from model import PneumoniaDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--model_name', type=str, default='efficientnetb0',
                        choices=['efficientnetb0', 'resnet50'],
                        help='Base model architecture')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (width and height)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained model')
    return parser.parse_args()

def main():
    """Main function to train the model."""
    # Parse command line arguments
    args = parse_args()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(args.random_state)
    
    # Load and preprocess the dataset
    print(f"Loading dataset from {args.data_dir}...")
    train_gen, val_gen, test_gen, class_indices, num_classes, class_names = load_dataset(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Print dataset information
    print(f"\nDataset information:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Classes: {', '.join(class_names)}")
    print(f"  - Training samples: {train_gen.samples}")
    print(f"  - Validation samples: {val_gen.samples}")
    print(f"  - Test samples: {test_gen.samples}")
    
    # Initialize the model
    print(f"\nInitializing {args.model_name.upper()} model for Pneumonia Detection...")
    model = PneumoniaDetector(
        num_classes=2,  # Binary classification: Normal vs Pneumonia
        input_shape=(args.img_size, args.img_size, 3),
        model_name=args.model_name
    )
    model.class_names = class_names
    
    # Print model summary
    model.model.summary()
    
    # Train the model
    print("\nStarting model training...")
    history = model.train(
        train_gen,
        val_gen,
        epochs=args.epochs
    )
    
    # Evaluate the model on test data
    print("\nEvaluating on test data...")
    metrics = model.evaluate(test_gen)
    
    # Print evaluation metrics
    print("\nTest Metrics:")
    print(f"  - Loss: {metrics['loss']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-Score: {metrics['f1_score']:.4f}")
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"pneumonia_detector_{timestamp}.h5")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save the class indices
    class_indices_path = os.path.join(args.output_dir, f"class_indices_{timestamp}.json")
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved to {class_indices_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    main()
