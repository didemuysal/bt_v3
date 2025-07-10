import argparse
import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# --- Imports from your project ---
from model import create_brain_tumour_model
from data import BrainTumourDataset # We use this to get the transformations

# --- Imports from the grad-cam library ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# The labels in the dataset are 1, 2, 3. We map them to 0, 1, 2.
# 1: meningioma, 2: glioma, 3: pituitary tumor
CLASS_INDEX_TO_NAME = {0: 'meningioma', 1: 'glioma', 2: 'pituitary'}

def get_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for the Brain Tumour Classification model.")
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model checkpoint (.pth), e.g., "fold_1_resnet50_finetune_adam_lr-0.0001_best_model.pth".')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the input image (.mat file).')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                        help="The model architecture used during training.")
    parser.add_argument('--target_class', type=int, default=None,
                        help='Optional: specify a class index (0: meningioma, 1: glioma, 2: pituitary). If None, the top predicted class is used.')
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the Model
    print(f"Loading model architecture: {args.model_name}")
    model = create_brain_tumour_model(model_name=args.model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Unfreeze all layers in the model to allow gradients to be computed
    for param in model.parameters():
        param.requires_grad = True
    
    model.eval()

    # 2. Select the Target Layer
    target_layers = [model.layer4]

    # 3. Prepare the Input Image
    dummy_dataset = BrainTumourDataset(data_folder="", filenames=[], labels=[], is_train=False)
    image_transform = dummy_dataset.transform

    import h5py
    with h5py.File(args.image_path, "r") as f:
        image_data = f["cjdata"]["image"][()]
    
    image_data = image_data.astype(np.float32)
    image_data /= image_data.max()

    input_tensor = image_transform(image_data).unsqueeze(0).to(device)
    
    vis_img = cv2.cvtColor(np.uint8(image_data * 255), cv2.COLOR_GRAY2RGB)
    vis_img = np.float32(vis_img) / 255

    # --- THIS IS THE FIX ---
    # Resize the visualization image to match the model's input size (224x224)
    vis_img = cv2.resize(vis_img, (224, 224))
    # ----------------------

    # 4. Instantiate and Run Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    if args.target_class is not None:
        targets = [ClassifierOutputTarget(args.target_class)]
        target_name = CLASS_INDEX_TO_NAME.get(args.target_class, "Unknown")
        print(f"Generating CAM for specified target class: {args.target_class} ({target_name})")
    else:
        targets = None
        print("No target class specified. Using the model's top prediction.")

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    output = torch.softmax(model(input_tensor), dim=1)
    top_pred_index = output.argmax().item()
    top_pred_score = output.max().item()
    predicted_class_name = CLASS_INDEX_TO_NAME.get(top_pred_index, "Unknown")
    print(f"Model Prediction: '{predicted_class_name}' (Class {top_pred_index}) with confidence {top_pred_score:.3f}")

    # 5. Visualize and Save the Result
    visualization = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
    
    img_basename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_filename = f"{img_basename}_gradcam_pred_{predicted_class_name}.png"

    cv2.imwrite(output_filename, visualization)
    print(f"✅ Grad-CAM visualization saved to: {output_filename}")

if __name__ == '__main__':
    main()