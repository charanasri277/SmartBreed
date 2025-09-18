import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import regularizers
import os
import json

def create_fresh_model(num_classes):
    """Create a fresh model with ImageNet weights as emergency fallback"""
    print("Creating fresh model architecture...")
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def load_trained_model(model_path="final_indian_bovine_breed_classifier_mobilenetv2.h5"):
    """
    Load model with emergency fallback
    """
    # First, get number of classes
    try:
        with open("class_indices.json", "r") as f:
            class_indices = json.load(f)
        num_classes = len(class_indices)
    except:
        print("Warning: Could not load class_indices.json, using default 41 classes")
        num_classes = 41
    
    # Try to load the saved model
    if os.path.exists(model_path):
        try:
            print(f"Attempting to load model from {model_path}...")
            model = load_model(model_path, compile=False)
            
            # Recompile without problematic metrics
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            print("Existing model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            print("Creating fresh model instead...")
    
    # Fallback: Create fresh model
    print("WARNING: Using fresh model with ImageNet weights!")
    print("This model is NOT trained on your cattle data!")
    print("Please retrain using the fixed training script in Colab.")
    
    model = create_fresh_model(num_classes)
    return model

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess uploaded image for prediction
    """
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise e

def predict_breed(model, img_array, class_indices):
    """
    Predict breed from preprocessed image
    """
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions, axis=1)[0])
        
        idx_to_class = {v: k for k, v in class_indices.items()}
        breed_name = idx_to_class.get(predicted_class_idx, f"Unknown_Class_{predicted_class_idx}")
        
        confidence = float(np.max(predictions)) * 100
        
        return breed_name, confidence, predicted_class_idx
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e
