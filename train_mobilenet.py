#!/usr/bin/python3
"""
MobileNetV2 Gender Classification Training Script
TCSS 555 Machine Learning Project
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
IMG_DIR = "C:/Users/rusla/Desktop/TCSS555/data/training/image"
CSV_PATH = "C:/Users/rusla/Desktop/TCSS555/data/training/profile/profile.csv"
MODEL_SAVE_PATH = "C:/Users/rusla/Desktop/TCSS555/mobilenet_gender_model.pth"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 128

# ============================================================================
# DATASET CLASS
# ============================================================================
class GenderDataset(Dataset):
    """Dataset class for gender classification"""
    
    def __init__(self, data, img_dir, transform=None):
        self.data = data.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['userid']}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        # Gender: 0 = male, 1 = female
        gender = torch.tensor(row['gender'], dtype=torch.float32)
        
        return image, gender


# ============================================================================
# MODEL CLASS
# ============================================================================
class MobileNetV2Gender(nn.Module):
    """MobileNetV2 model for binary gender classification"""
    
    def __init__(self):
        super(MobileNetV2Gender, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Get the number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom head
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
# TRAINING FUNCTION
# ============================================================================
def train_model():
    """Train the MobileNetV2 gender classification model"""
    
    print("="*70)
    print("MobileNetV2 Gender Classification Training")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {CSV_PATH}...")
    data = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(data)}")
    print(f"Gender distribution:\n{data['gender'].value_counts()}")
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        data, 
        test_size=0.2, 
        random_state=42,
        stratify=data['gender']
    )
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    print("\nCreating data loaders...")
    train_dataset = GenderDataset(train_data, IMG_DIR, train_transform)
    val_dataset = GenderDataset(val_data, IMG_DIR, val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nInitializing MobileNetV2 model...")
    model = MobileNetV2Gender().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    
    best_val_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (images, gender) in enumerate(train_loader):
            images = images.to(device)
            gender = gender.to(device).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gender)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for accuracy calculation
            preds = (outputs > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(gender.cpu().numpy())
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, gender in val_loader:
                images = images.to(device)
                gender = gender.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, gender)
                
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(gender.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_preds)
        
        # Update learning rate scheduler
        scheduler.step(val_accuracy)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved! (Val Acc: {val_accuracy:.4f})")
        
        print("-"*70)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total training time: {elapsed_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Baseline gender accuracy: 0.59 (59%)")
    if best_val_accuracy > 0.59:
        print(f"✓ Improvement over baseline: +{(best_val_accuracy-0.59)*100:.2f}%")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TCSS 555 - MobileNetV2 Gender Classification")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Image Directory: {IMG_DIR}")
    print(f"  CSV Path: {CSV_PATH}")
    print(f"  Model Save Path: {MODEL_SAVE_PATH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()