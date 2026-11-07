import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix, cohen_kappa_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse

class DRDatasetHighResEval(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpeg")
        
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, torch.tensor(row['label'], dtype=torch.long), row['image_id']

def create_model(num_classes=5):
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def evaluate_fold(fold, df, args):
    print(f"Evaluating Fold {fold}...")
    val_df = df[df['fold'] == fold].copy()
    val_transform = get_val_transform()
    val_dataset = DRDatasetHighResEval(val_df, args.img_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = create_model().to(args.device)
    model_path = os.path.join(args.run_dir, "models", f"effnetb4_fold_{fold}_best.pth")
    if not os.path.exists(model_path):
        print(f"Model path missing: {model_path}")
        return pd.DataFrame()
        
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    model.eval()
    
    all_preds, all_probs, all_ids, all_labels = [], [], [], []
    
    with torch.no_grad():
        for images, labels, img_ids in tqdm(val_loader, desc=f"Fold {fold}"):
            images = images.to(args.device)
            # Use standard precision for inference to avoid AMP overhead bugs if any
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_ids.extend(img_ids)
            
    res = pd.DataFrame({
        'image_id': all_ids,
        'label': all_labels,
        'oof_pred': all_preds
    })
    probs_df = pd.DataFrame(all_probs, columns=[f'prob_class_{i}' for i in range(5)])
    return pd.concat([res, probs_df], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/train_images_512_ben")
    parser.add_argument("--splits", default="splits/fold_assignments.csv")
    parser.add_argument("--run_dir", default="efficientnet_highres_run_2026-09-30")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {args.device}")
    
    df = pd.read_csv(args.splits)
    
    all_oof = []
    for fold in range(5):
        fold_res = evaluate_fold(fold, df, args)
        all_oof.append(fold_res)
        
    final_oof = pd.concat(all_oof, ignore_index=True)
    
    os.makedirs(os.path.join(args.run_dir, "predictions"), exist_ok=True)
    final_oof.to_csv(os.path.join(args.run_dir, "predictions", "oof_predictions.csv"), index=False)
    
    # Calculate metrics
    labels = final_oof['label'].values
    preds = final_oof['oof_pred'].values
    probs = final_oof[[f'prob_class_{i}' for i in range(5)]].values
    
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    ll = log_loss(labels, probs)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    
    print("\n" + "="*50)
    print("FINAL OUT-OF-FOLD METRICS (EFFICIENTNET-B4 | 512x512)")
    print("="*50)
    print(f"Total Evaluated Images : {len(final_oof)}")
    print(f"Raw Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro-F1               : {macro_f1:.4f}")
    print(f"Quadratic Kappa (QWK)  : {qwk:.4f}")
    print(f"Log Loss               : {ll:.4f}")
    print("="*50)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    print("\nPredictions saved successfully.")

if __name__ == "__main__":
    main()
