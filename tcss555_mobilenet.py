#!/usr/bin/python3
"""
TCSS 555 Prediction Script with MobileNetV2 Gender Classification
Generates XML prediction files using trained MobileNetV2 model
"""

import csv
import sys
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
MODEL_PATH = "C:/Users/rusla/Desktop/TCSS555/mobilenet_gender_model.pth"
PROFILE_CSV = "C:/Users/rusla/Desktop/TCSS555/data/training/profile/profile.csv"

# Hardcoded baseline values for age and personality (from training data)
BASELINE_AGE_GROUP = "xx-24"
BASELINE_EXTROVERT = "3.49"
BASELINE_NEUROTIC = "2.73"
BASELINE_AGREEABLE = "3.58"
BASELINE_CONSCIENTIOUS = "3.45"
BASELINE_OPEN = "3.91"

# Image preprocessing settings (must match training)
IMG_SIZE = 128


# ============================================================================
# MODEL CLASS (Must match training script)
# ============================================================================
class MobileNetV2Gender(nn.Module):
    """MobileNetV2 model for binary gender classification"""
    
    def __init__(self):
        super(MobileNetV2Gender, self).__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(pretrained=False)
        
        # Get the number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Custom classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def load_model(device):
    """Load the trained MobileNetV2 model"""
    model = MobileNetV2Gender().to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Print model info if available
        if 'val_accuracy' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_accuracy']:.4f} ({checkpoint['val_accuracy']*100:.2f}%)")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        raise


def load_and_preprocess_image(img_path, transform):
    """Load and preprocess an image for the model"""
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        return image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}", file=sys.stderr)
        return None


def predict_gender(model, img_path, transform, device):
    """
    Predict gender from an image using the MobileNetV2 model.
    
    Returns:
        Gender prediction as string ('male' or 'female')
    """
    image = load_and_preprocess_image(img_path, transform)
    
    if image is None:
        # Return baseline if image loading fails
        return 'female'
    
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        prediction = output.item()
    
    # Threshold at 0.5: >0.5 = female (1), <=0.5 = male (0)
    gender = 'female' if prediction > 0.5 else 'male'
    
    return gender


# ============================================================================
# XML GENERATION
# ============================================================================
def save_to_XML_file(input_dir, output_dir, model, transform, device, calculate_accuracy=False):
    """
    Generate XML files with predictions for each user in the test dataset.
    
    Args:
        input_dir: Path to the directory containing user image files
        output_dir: Path to save the XML prediction files
        model: Loaded PyTorch model
        transform: Image transformation pipeline
        device: torch device (cpu or cuda)
        calculate_accuracy: Whether to calculate accuracy (requires profile.csv)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth if calculating accuracy
    ground_truth = None
    if calculate_accuracy:
        try:
            ground_truth = pd.read_csv(PROFILE_CSV)
            ground_truth['gender_str'] = ground_truth['gender'].map({0.0: 'male', 1.0: 'female'})
            ground_truth = ground_truth.set_index('userid')
            print(f"Loaded ground truth from {PROFILE_CSV}\n")
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")
            calculate_accuracy = False
    
    # Get all user IDs from image files in the input directory
    try:
        # List all files in the directory
        all_files = os.listdir(input_dir)
        
        # Extract user IDs from filenames (remove extension)
        user_data = []
        for filename in all_files:
            # Skip hidden files and non-image files
            if filename.startswith('.') or filename.startswith('._'):
                continue
            
            # Check if it's a file (not a directory)
            full_path = os.path.join(input_dir, filename)
            if os.path.isfile(full_path):
                # Check if it's an image file
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    userid = os.path.splitext(filename)[0]
                    user_data.append((userid, full_path))
        
    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not user_data:
        print(f"Warning: No user image files found in {input_dir}", file=sys.stderr)
        return
    
    print(f"Found {len(user_data)} users in dataset")
    print(f"Generating XML files with MobileNetV2 gender predictions...\n")
    
    # For accuracy calculation
    predictions = []
    true_labels = []
    
    # Generate XML file for each user
    count = 0
    for userid, img_path in user_data:
        # Predict gender using MobileNetV2 model
        gender = predict_gender(model, img_path, transform, device)
        
        # Store for accuracy calculation
        if calculate_accuracy and userid in ground_truth.index:
            predictions.append(gender)
            true_labels.append(ground_truth.loc[userid, 'gender_str'])
        
        # XML String with predictions
        xml_string = (
            f'<user id="{userid}" '
            f'age_group="{BASELINE_AGE_GROUP}" '
            f'gender="{gender}" '
            f'extrovert="{BASELINE_EXTROVERT}" '
            f'neurotic="{BASELINE_NEUROTIC}" '
            f'agreeable="{BASELINE_AGREEABLE}" '
            f'conscientious="{BASELINE_CONSCIENTIOUS}" '
            f'open="{BASELINE_OPEN}" />\n'
        )
        
        # File path to save the XML file
        file_path = os.path.join(output_dir, f"{userid}.xml")
        
        # Write the XML string to the file
        try:
            with open(file_path, "w", encoding="utf-8") as out_file:
                out_file.write(xml_string)
            count += 1
            
            # Print progress for first few and every 100th file
            if count <= 5 or count % 100 == 0:
                print(f"Generated: {userid}.xml (Gender: {gender})")
        except Exception as e:
            print(f"Error writing {userid}.xml: {e}", file=sys.stderr)
    
    print(f"\nSuccessfully generated {count} XML files")
    print(f"Output directory: {output_dir}")
    
    # Calculate and display accuracy
    if calculate_accuracy and len(predictions) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        print("\n" + "="*70)
        print("ACCURACY RESULTS")
        print("="*70)
        print(f"Gender Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Baseline Gender Accuracy: 0.59 (59%)")
        if accuracy > 0.59:
            print(f"✓ Improvement over baseline: +{(accuracy-0.59)*100:.2f}%")
        else:
            print(f"✗ Below baseline: {(accuracy-0.59)*100:.2f}%")
        print("="*70)


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Parse command line arguments and generate XML predictions."""
    parser = argparse.ArgumentParser(
        description='Generate XML predictions using MobileNetV2 gender model for TCSS 555 project'
    )
    
    parser.add_argument(
        '-i', '--input_dir', 
        required=True, 
        type=str,
        help='Path to the directory containing user image files'
    )
    
    parser.add_argument(
        '-o', '--output_dir', 
        required=True, 
        type=str,
        help='Path to the output directory for XML files'
    )
    
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Calculate accuracy using ground truth from profile.csv'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TCSS 555 - MobileNetV2 Gender Prediction")
    print("="*70)
    print(f"\nModel path: {MODEL_PATH}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.eval:
        print(f"Ground truth: {PROFILE_CSV}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the MobileNetV2 model
    print("\nLoading MobileNetV2 gender model...")
    try:
        model = load_model(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model from {MODEL_PATH}: {e}", file=sys.stderr)
        print("\nPlease ensure:")
        print("1. The model file exists at the specified path")
        print("2. You have trained the model using train_mobilenet.py")
        sys.exit(1)
    
    # Setup image transformation (must match training)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\nPrediction settings:")
    print(f"  Gender: MobileNetV2 model prediction")
    print(f"  Age Group: {BASELINE_AGE_GROUP} (baseline)")
    print(f"  Personality traits: baseline values")
    print()
    
    # Generate the XML files
    save_to_XML_file(args.input_dir, args.output_dir, model, transform, device, 
                     calculate_accuracy=args.eval)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
