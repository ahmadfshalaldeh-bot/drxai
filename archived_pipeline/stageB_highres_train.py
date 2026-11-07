import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
import argparse

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DRDatasetHighRes(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpeg")
        
        # Images are already preprocessed via Ben's algo to 512x512 BGR
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to zero tensor if missing, but shouldn't happen
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, torch.tensor(row['label'], dtype=torch.long)

def get_transforms():
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform

def create_model(num_classes=5):
    # Using EfficientNet-B4 for a great balance of accuracy and memory at 512x512
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train_fold(fold, df, args):
    print(f"========== FOLD {fold} ==========")
    train_df = df[df['fold'] != fold].copy()
    val_df = df[df['fold'] == fold].copy()
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = DRDatasetHighRes(train_df, args.img_dir, transform=train_transform)
    val_dataset = DRDatasetHighRes(val_df, args.img_dir, transform=val_transform)
    
    # Weighted sampling for massive imbalance
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[t] for t in train_df['label']])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    model = create_model().to(args.device)
    
    # Focal Loss handles tough misclassifications
    criterion = FocalLoss(alpha=1, gamma=2).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler()
    
    best_macro_f1 = -1
    patience_counter = 0
    os.makedirs(os.path.join(args.run_dir, "models"), exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        # VALIDATE
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                with autocast():
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        scheduler.step(macro_f1)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.run_dir, "models", f"effnetb4_fold_{fold}_best.pth"))
            print(f"  -> Best model saved! (Macro-F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/train_images_512_ben", help="Path to Ben's preprocessed 512x512 images")
    parser.add_argument("--labels", default="diabetic-retinopathy-detection/trainLabels.csv", help="Original labels")
    parser.add_argument("--splits", default="splits/fold_assignments.csv", help="Cross-validation splits")
    parser.add_argument("--run_dir", default="efficientnet_highres_run_2026-09-30", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=40, help="Max epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {args.device}")
    
    df = pd.read_csv(args.splits)
    
    for fold in range(5):
        train_fold(fold, df, args)

if __name__ == "__main__":
    main()
