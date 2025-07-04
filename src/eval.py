# No need to touch anything here :D
import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import os

# Configuration
IGNORE_INDEX = 255
NUM_CLASSES = 8
CLASS_NAMES = [
    "Out of Bounds",
    "Water",
    "Trees",
    "Flooded Vegetation",
    "Built Structures",
    "Crops",
    "Bare Ground",
    "Rangeland",
]

LEGEND_INDICES = [1, 2, 4, 7]

# Reuse the same preprocessing functions from training
original_classes = [0, 1, 2, 5, 7, 8, 10, 11]
class_mapping = {old: new for new, old in enumerate(original_classes)}
max_class = max(original_classes)
lookup_table = np.full(max_class + 2, IGNORE_INDEX, dtype=np.int64)
for old, new in class_mapping.items():
    lookup_table[old] = new

def remap_classes(mask_array):
    clipped = np.clip(mask_array, 0, max_class + 1)
    return lookup_table[clipped]

def load_and_pad(image_path, mask_path):
    """Load and pad images to make dimensions divisible by 32"""
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0) / 10000.0  # Sentinel scaling
        h, w = image.shape[:2]
    
    with rasterio.open(mask_path) as src:
        mask = remap_classes(src.read(1))
    
    # Calculate padding needed for network compatibility
    divisor = 32  # For ResNet-based UNet
    pad_h = (divisor - (h % divisor)) % divisor
    pad_w = (divisor - (w % divisor)) % divisor
    
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=IGNORE_INDEX)
    
    return image_padded, mask_padded, (h, w)

def calculate_metrics(preds, labels, num_classes, ignore_index):
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    accuracy = np.diag(cm).sum() / cm.sum()
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    f1 = 2 * np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0))

    return {
        'confusion_matrix': cm,
        'overall_accuracy': accuracy,
        'iou': iou,
        'f1_score': f1
    }

def visualize_full_result(image, true_mask, pred_mask, save_path):
    plt.figure(figsize=(28, 8))
    
    plt.subplot(1, 3, 1)
    plt.imshow(true_mask, vmin=0, vmax=NUM_CLASSES-1, cmap='jet')
    plt.title('Ground Truth')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, vmin=0, vmax=NUM_CLASSES-1, cmap='jet')
    plt.title('Prediction')
    
    # Create unified legend
    patches = [
        mpatches.Patch(color=plt.cm.jet(i / (NUM_CLASSES - 1)), label=CLASS_NAMES[i])
        for i in LEGEND_INDICES
    ]

    # Add legend to the figure
    plt.figlegend(
        handles=patches,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize="medium",
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def evaluate_model(model_path, image_path, mask_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and pad data
    image_padded, mask_padded, orig_dims = load_and_pad(image_path, mask_path)
    orig_h, orig_w = orig_dims
    
    # Convert to tensor
    image_tensor = torch.tensor(image_padded.transpose(2, 0, 1), 
                              dtype=torch.float32).unsqueeze(0).to(device)
    
    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=image_padded.shape[-1],
        classes=NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Remove padding
    final_pred = pred_mask[:orig_h, :orig_w]
    final_mask = mask_padded[:orig_h, :orig_w]
    final_image = image_padded[:orig_h, :orig_w, :]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize
    visualize_full_result(final_image, final_mask, final_pred,
                        os.path.join(output_dir, 'full_prediction.png'))
    
    # Calculate metrics
    metrics = calculate_metrics(
        final_pred.flatten(),
        final_mask.flatten(),
        NUM_CLASSES,
        IGNORE_INDEX
    )
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n\n")
        f.write("Class-wise Metrics:\n")
        f.write("{:<20} {:<10} {:<10} {:<10}\n".format("Class", "IoU", "F1", "Support"))
        for i in LEGEND_INDICES:
            support = metrics['confusion_matrix'][i].sum()
            f.write("{:<20} {:<10.4f} {:<10.4f} {:<10}\n".format(
                CLASS_NAMES[i],
                metrics['iou'][i],
                metrics['f1_score'][i],
                support
            ))
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    evaluate_model(
        model_path="../models/model.pth",
        image_path="../stacked/blist_raw_2020.tif",
        mask_path="../truth/2020.tif",
        output_dir="../evaluation_results"
    )