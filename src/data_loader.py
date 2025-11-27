"""
Data loading and preprocessing utilities for the Pneumonia Detector.
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    """Handles loading and preprocessing of image data."""
    
    def __init__(self, data_dir, img_size=(224, 224), test_size=0.2, val_size=0.2, random_state=42):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Path to the directory containing the dataset.
            img_size (tuple): Target size for the images (height, width).
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training set to include in the validation split.
            random_state (int): Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.class_names = []
        self.num_classes = 0
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, class_indices)
        """
        images = []
        labels = []
        
        # Get class names from subdirectories
        self.class_names = sorted([d for d in os.listdir(self.data_dir) 
                                 if os.path.isdir(os.path.join(self.data_dir, d))])
        self.num_classes = len(self.class_names)
        
        if self.num_classes == 0:
            raise ValueError(f"No subdirectories found in {self.data_dir}. Each class should be in its own subdirectory.")
        
        # Encode labels
        self.label_encoder.fit(self.class_names)
        
        # Load images and labels
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Recursively search for image files in class directory
            for root, dirs, files in os.walk(class_dir):
                for img_name in files:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, img_name)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize(self.img_size)
                            img_array = np.array(img) / 255.0
                            images.append(img_array)
                            labels.append(class_name)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        if not images:
            raise ValueError(f"No valid images found in {self.data_dir}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = self.label_encoder.transform(labels)
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=self.val_size/(1-self.test_size), 
            random_state=self.random_state, 
            stratify=y_train_val
        )
        
        # Create class indices mapping
        class_indices = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_indices

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            numpy.ndarray: Preprocessed image array.
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        return img_array.reshape(1, -1)  # Flatten for scikit-learn

    def get_class_name(self, class_index):
        """
        Get the class name from a class index.
        
        Args:
            class_index (int): Index of the class.
            
        Returns:
            str: The name of the class.
        """
        if 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return None

def load_dataset(data_dir, img_size=(224, 224), test_size=0.2, val_size=0.2, random_state=42):
    """
    Convenience function to load and preprocess the dataset.
    
    Args:
        data_dir (str): Path to the directory containing the dataset.
        img_size (tuple): Target size for the images (height, width).
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, class_indices)
    """
    loader = DataLoader(
        data_dir=data_dir,
        img_size=img_size,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    return loader.load_data()