#!/usr/bin/env python
"""
Script to make predictions using the trained Pneumonia Prediction.
"""
import os
import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from model import WildlifeClassifier

def load_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Target size for the image.
        
    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Resize the image
    img = img.resize(img_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def load_class_indices(class_indices_path):
    """
    Load class indices from a JSON file.
    
    Args:
        class_indices_path (str): Path to the JSON file containing class indices.
        
    Returns:
        dict: Dictionary mapping class indices to class names.
    """
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Invert the dictionary to get index to class name mapping
    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class

def main():
    """Main function to make predictions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using the African Wildlife Classifier')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.h5)')
    parser.add_argument('--class_indices', type=str, required=True,
                        help='Path to the JSON file containing class indices')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image file for prediction')
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = WildlifeClassifier.load(args.model_path)
    
    # Load class indices
    print(f"Loading class indices from {args.class_indices}...")
    index_to_class = load_class_indices(args.class_indices)
    
    # Set class names in the model
    model.class_names = [index_to_class[i] for i in range(len(index_to_class))]
    
    # Load and preprocess the image
    print(f"Processing image: {args.image_path}")
    img_array = load_image(args.image_path)
    
    # Make prediction
    print("Making prediction...")
    predicted_class_index, confidence, probabilities = model.predict(img_array)
    
    # Get the predicted class name
    predicted_class = model.class_names[predicted_class_index]
    
    # Print the results
    print("\nPrediction Results:")
    print(f"  - Predicted class: {predicted_class} (index: {predicted_class_index})")
    print(f"  - Confidence: {confidence * 100:.2f}%")
    
    # Print top 3 predictions
    print("\nTop 3 predictions:")
    top_indices = np.argsort(probabilities)[::-1][:3]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {model.class_names[idx]}: {probabilities[idx] * 100:.2f}%")

if __name__ == '__main__':
    main()
