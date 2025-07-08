# train.py
# Main script to train and evaluate the model using 5-fold cross-validation.

import copy
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from data import BrainTumourDataset
from splits import get_patient_level_splits
from model import create_brain_tumour_model

# --- Configuration ---
DATA_FOLDER = "data_raw"
CVIND_PATH  = "cvind.mat"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
LEARNING_RATE_HEAD = 1e-4  # Learning rate for the new classifier head
LEARNING_RATE_FT   = 1e-5  # A smaller learning rate for fine-tuning the whole model
HEAD_TRAIN_EPOCHS = 3      # Number of epochs to train only the head
MAX_FINETUNE_EPOCHS = 50   # Max number of epochs for fine-tuning
PATIENCE = 5               # How many epochs to wait for improvement before stopping

# --- Helper Function to run one epoch ---
def run_one_epoch(model, loader, criterion, optimizer=None):
    """
    Runs a single epoch of training or evaluation.
    - If an optimizer is provided, it runs in training mode.
    - Otherwise, it runs in evaluation mode (no gradients).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # For evaluation, we don't need to calculate gradients
    with torch.set_grad_enabled(is_training):
        # The 'loader' passed to this function will now be a tqdm object
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Get model predictions
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad() # Reset gradients
                loss.backward()       # Backpropagation
                optimizer.step()      # Update weights

            # Track performance
            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * images.size(0)
            correct_predictions += (preds == labels).sum().item()
            total_samples += images.size(0)
            
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# --- Main Training Loop ---
def main():
    print(f"Using device: {DEVICE}")
    total_start_time = time.time() # Start the total timer

    # Get the 5 data splits for cross-validation
    all_splits = get_patient_level_splits(DATA_FOLDER, CVIND_PATH)
    test_accuracies = []

    # Loop through each of the 5 folds
    for i, (train_files, train_labels, val_files, val_labels, test_files, test_labels) in enumerate(all_splits):
        fold_num = i + 1
        print(f"\n--- FOLD {fold_num}/5 ---")
        
        # --- Create DataLoaders ---
        train_dataset = BrainTumourDataset(DATA_FOLDER, train_files, train_labels, is_train=True)
        val_dataset   = BrainTumourDataset(DATA_FOLDER, val_files, val_labels, is_train=False)
        test_dataset  = BrainTumourDataset(DATA_FOLDER, test_files, test_labels, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # --- Create Model and Optimizer ---
        model = create_brain_tumour_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        # --- Stage 1: Train only the classifier head ---
        print("\nStage 1: Training the classifier head...")
        head_optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_HEAD)
        for epoch in range(HEAD_TRAIN_EPOCHS):
            # Wrap train_loader with tqdm for a progress bar
            progress_bar = tqdm(train_loader, desc=f"Fold {fold_num} Head Epoch {epoch+1}/{HEAD_TRAIN_EPOCHS}")
            _, train_acc = run_one_epoch(model, progress_bar, criterion, head_optimizer)
            # Update the progress bar with the final accuracy for the epoch
            progress_bar.set_postfix(accuracy=f"{train_acc:.4f}")
            
        # --- Stage 2: Fine-tune the entire network ---
        print("\nStage 2: Fine-tuning the full network...")
        for param in model.parameters():
            param.requires_grad = True # Unfreeze all layers
        
        finetune_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_FT)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(MAX_FINETUNE_EPOCHS):
            # Wrap train_loader with tqdm
            train_progress_bar = tqdm(train_loader, desc=f"Fold {fold_num} FT Epoch {epoch+1}/{MAX_FINETUNE_EPOCHS} (Train)")
            _, train_acc = run_one_epoch(model, train_progress_bar, criterion, finetune_optimizer)
            train_progress_bar.set_postfix(accuracy=f"{train_acc:.4f}")

            # Wrap val_loader with tqdm
            val_progress_bar = tqdm(val_loader, desc=f"Fold {fold_num} FT Epoch {epoch+1}/{MAX_FINETUNE_EPOCHS} (Val)")
            val_loss, val_acc = run_one_epoch(model, val_progress_bar, criterion)
            val_progress_bar.set_postfix(loss=f"{val_loss:.4f}", accuracy=f"{val_acc:.4f}")

            print(f"  -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping: if validation loss doesn't improve, stop training
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break
        
        # --- Stage 3: Test the best model on the held-out test set ---
        print("\nStage 3: Testing the best model...")
        model.load_state_dict(best_model_state) # Load the best performing weights
        
        # Wrap test_loader with tqdm
        test_progress_bar = tqdm(test_loader, desc=f"Fold {fold_num} Testing")
        _, test_acc = run_one_epoch(model, test_progress_bar, criterion)
        test_accuracies.append(test_acc)
        print(f"✅ Fold {fold_num} Test Accuracy: {test_acc:.3%}")
    
    # --- Final Summary ---
    print("\n--- Cross-Validation Complete ---")
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    print(f"Average Test Accuracy across all folds: {mean_acc:.3%} ± {std_acc:.3%}")

    # Calculate and print total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"\nTotal training time: {total_training_time // 60:.0f} minutes and {total_training_time % 60:.0f} seconds. ⏱️")

if __name__ == '__main__':
    main()