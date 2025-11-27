#!/usr/bin/env python
"""
Script to make predictions using the trained Pneumonia Detection model.
"""
import os
import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from model import WildlifeClassifier

def load_xray_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess a chest X-ray image for prediction.
    
    Args:
        image_path (str): Path to the X-ray image file.
        img_size (tuple): Target size for the image (height, width).
        
    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    try:
        img = Image.open(image_path).convert('L')
        
        img_array = np.array(img)
        if np.mean(img_array) < 10 or np.mean(img_array) > 240:
            raise ValueError("Image appears to be blank or overexposed")
        
        img = img.resize(img_size)
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Error processing X-ray image: {str(e)}")

def load_class_indices(class_indices_path):
    """
    Load class indices from a JSON file.
    
    Args:
        class_indices_path (str): Path to the JSON file containing class indices.
        
    Returns:
        dict: Dictionary mapping class indices to class names.
    """
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        index_to_class = {v: k for k, v in class_indices.items()}
        return index_to_class
    except Exception as e:
        raise ValueError(f"Error loading class indices: {str(e)}")

def main():
    """Main function to make predictions."""
    parser = argparse.ArgumentParser(description='Make predictions using the Pneumonia Detection model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file (.h5)')
    parser.add_argument('--class_indices', type=str, required=True,
                      help='Path to the JSON file containing class indices')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the X-ray image file for prediction')
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image file not found: {args.image_path}")
            
        if not args.image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("Only PNG, JPG, and JPEG image formats are supported")
        
        print(f"Loading model from {args.model_path}...")
        model = WildlifeClassifier.load(args.model_path)
        
        print(f"Loading class indices from {args.class_indices}...")
        index_to_class = load_class_indices(args.class_indices)
        model.class_names = [index_to_class[i] for i in range(len(index_to_class))]

        print(f"Processing X-ray image: {args.image_path}")
        try:
            img_array = load_xray_image(args.image_path)
        except Exception as e:
            raise ValueError(f"Invalid X-ray image: {str(e)}")
        
        print("Making prediction...")
        predicted_class_index, confidence, probabilities = model.predict(img_array)
        
        predicted_class = model.class_names[predicted_class_index]
        
        print("\nPrediction Results:")
        print(f"  - Predicted class: {predicted_class}")
        print(f"  - Confidence: {confidence * 100:.2f}%")
        print("\nClass Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  - {model.class_names[i]}: {prob * 100:.2f}%")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())