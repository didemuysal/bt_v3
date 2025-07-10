# train.py
# Final, corrected version for dissertation experiments.
# Fixes the tqdm AttributeError and includes all features.

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (auc, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)
from sklearn.preprocessing import label_binarize
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from data import BrainTumourDataset
from model import create_brain_tumour_model
from splits import get_patient_level_splits

# --- Helper Function ---
def run_one_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []
    
    # The loader is a tqdm object. We iterate over it.
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scores = torch.softmax(outputs, dim=1)
        preds = torch.argmax(scores, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.cpu().detach().numpy())
        total_loss += loss.item() * images.size(0)
    
    # --- BUG FIX ---
    # The error was here. We need to get the length from the underlying dataloader,
    # not the tqdm object itself.
    avg_loss = total_loss / len(loader.iterable.dataset)
    
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_scores)


def get_optimizer(model_params, optimizer_name, lr):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=0.01)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, alpha=0.99, eps=1e-8)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model_params, rho=0.9, eps=1e-6)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")


# --- Main Training Loop ---
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("="*50)
    print(f"Starting Experiment: {time.ctime()}")
    print(f"Running with arguments: {args}")
    print(f"Using device: {DEVICE}")
    print("="*50)

    results_log, n_classes = [], 3
    all_splits = get_patient_level_splits(args.data_folder, args.cvind_path)
    summed_cm = np.zeros((n_classes, n_classes), dtype=int)
    mean_fpr, tprs, aucs = np.linspace(0, 1, 100), [], []

    for i, (train_files, train_labels, val_files, val_labels, test_files, test_labels) in enumerate(all_splits):
        fold_num = i + 1
        print(f"\n--- FOLD {fold_num}/5 ---")
        train_dataset = BrainTumourDataset(args.data_folder, train_files, train_labels, is_train=True)
        val_dataset = BrainTumourDataset(args.data_folder, val_files, val_labels, is_train=False)
        test_dataset = BrainTumourDataset(args.data_folder, test_files, test_labels, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

        model = create_brain_tumour_model(model_name=args.model).to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        if args.strategy == 'finetune':
            print(f"\nStage 1: Training head for {args.head_epochs} epochs...")
            for param in model.parameters(): param.requires_grad = False
            for param in model.fc.parameters(): param.requires_grad = True
            head_optimizer = get_optimizer(model.fc.parameters(), args.optimizer, args.lr)
            for epoch in range(args.head_epochs):
                pbar = tqdm(train_loader, desc=f"Fold {fold_num} Head Epoch {epoch+1}/{args.head_epochs}")
                run_one_epoch(model, pbar, criterion, head_optimizer, device=DEVICE)
            fine_tune_lr = args.lr / 10.0
            print(f"\nStage 2: Fine-tuning the full network with LR={fine_tune_lr}...")
        else: # baseline
            fine_tune_lr = args.lr
            print(f"\nStage 2: Training full network with LR={fine_tune_lr}...")

        for param in model.parameters(): param.requires_grad = True
        optimizer = get_optimizer(model.parameters(), args.optimizer, fine_tune_lr)
        best_val_loss, epochs_without_improvement = float('inf'), 0
        best_model_state = None

        for epoch in range(args.max_epochs):
            train_pbar = tqdm(train_loader, desc=f"Fold {fold_num} FT Epoch {epoch+1}/{args.max_epochs} (Train)")
            run_one_epoch(model, train_pbar, criterion, optimizer, device=DEVICE)
            val_pbar = tqdm(val_loader, desc=f"Fold {fold_num} FT Epoch {epoch+1}/{args.max_epochs} (Val)", leave=False)
            val_loss, _, _, _ = run_one_epoch(model, val_pbar, criterion, device=DEVICE)
            print(f"  -> Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, best_model_state = val_loss, copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"  -> No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= args.patience:
                    print(f"  -> EARLY STOPPING triggered after {args.patience} epochs without improvement.")
                    break
       
       # Create the base filename
        file_prefix = f"{args.model}_{args.strategy}_{args.optimizer}"
        
        # Only add the learning rate if the optimizer uses it
        if args.optimizer != 'adadelta':
            file_prefix += f"_lr-{args.lr}"
            
        save_path = f"fold_{fold_num}_{file_prefix}_best_model.pth"
        
        torch.save(best_model_state, save_path)
        print(f"  -> Best model for fold {fold_num} saved to {save_path}")

        print("\nTesting the best model...")
        test_pbar = tqdm(test_loader, desc="Testing")
        test_loss, test_labels, test_preds, test_scores = run_one_epoch(model, test_pbar, criterion, device=DEVICE)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average=None, labels=list(range(n_classes)), zero_division=0)
        acc = (test_preds == test_labels).mean()
        summed_cm += confusion_matrix(test_labels, test_preds, labels=list(range(n_classes)))
        test_labels_binarized = label_binarize(test_labels, classes=list(range(n_classes)))
        fold_results = {'fold': fold_num, 'test_loss': test_loss, 'test_accuracy': acc}
        for class_idx, name in enumerate(['meningioma', 'glioma', 'pituitary']):
            fpr, tpr, _ = roc_curve(test_labels_binarized[:, class_idx], test_scores[:, class_idx])
            roc_auc = auc(fpr, tpr)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(roc_auc)
            for metric, value in zip(['precision', 'recall', 'f1', 'auc'], [precision[class_idx], recall[class_idx], f1[class_idx], roc_auc]):
                fold_results[f"{name}_{metric}"] = value
        results_log.append(fold_results)
        print(f"✅ Fold {fold_num} Test Accuracy: {acc:.3%}")

    df = pd.DataFrame(results_log)
    print("\n--- Cross-Validation Complete ---")
    print("Mean Performance Metrics Across 5 Folds:")
    print(df.drop(columns=['fold']).mean(axis=0))

    lr_str = "default" if args.optimizer == 'adadelta' else f"{args.lr}"
    file_prefix = f"{args.model}_{args.strategy}_{args.optimizer}_lr-{lr_str}"

    cm_normalized = summed_cm.astype('float') / summed_cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Meningioma', 'Glioma', 'Pituitary'], yticklabels=['Meningioma', 'Glioma', 'Pituitary'])
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix\n({file_prefix.replace("_", " ").title()})')
    plt.savefig(f"{file_prefix}_confusion_matrix.png")
    print(f"\nSaved confusion matrix to {file_prefix}_confusion_matrix.png")

    plt.figure(figsize=(8, 6))
    tprs_per_class = np.array(tprs).reshape(-1, n_classes, len(mean_fpr))
    aucs_per_class = np.array(aucs).reshape(-1, n_classes)
    for i, (color, name) in enumerate(zip(['aqua', 'darkorange', 'cornflowerblue'], ['Meningioma', 'Glioma', 'Pituitary'])):
        mean_tpr = np.mean(tprs_per_class[:, i, :], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs_per_class[:, i])
        std_auc = np.std(aucs_per_class[:, i])
        plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=f'ROC {name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curves\n({file_prefix.replace("_", " ").title()})'); plt.legend(loc="lower right")
    plt.savefig(f"{file_prefix}_roc_curve.png")
    print(f"Saved ROC curve to {file_prefix}_roc_curve.png")

    df.to_csv(f"{file_prefix}_results.csv", index=False)
    print(f"Saved detailed results to {file_prefix}_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Brain Tumour Classification Model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50'])
    parser.add_argument('--strategy', type=str, default='finetune', choices=['finetune', 'baseline'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adadelta'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_folder', type=str, default='data_raw')
    parser.add_argument('--cvind_path', type=str, default='cvind.mat')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--head_epochs', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)