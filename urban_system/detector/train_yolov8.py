"""
YOLOv8 Pothole Detection Training Pipeline
Incorporate techniques from:
  - Jiang (2024): Road Damage Detection using Deep Neural Networks (RDD2022 dataset format, multiscale features)
  - Zhong et al. (2025): Accurate 2D candidate localization for geometric quantification
"""

import os
import shutil
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(BASE_DIR, "models", ".yolo_config")

def train_pothole_detector(
    data_yaml_path=None,
    output_model_dir=None,
    epochs=10,
    imgsz=640,
    batch_size=8
):
    data_yaml_path = data_yaml_path or os.path.join(BASE_DIR, "data", "pothole_dataset", "data.yaml")
    output_model_dir = output_model_dir or os.path.join(BASE_DIR, "models")
    print("=" * 60)
    print("AI URBAN HOTSPOT SYSTEM: TRAINING YOLOV8 POTHOLE DETECTOR")
    print(f"Dataset YAML: {data_yaml_path}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Epochs: {epochs} | Image Size: {imgsz} | Batch: {batch_size}")
    print("=" * 60)
    
    os.makedirs(output_model_dir, exist_ok=True)
    base_weights = os.path.join(output_model_dir, "yolov8n.pt")
    if not os.path.exists(base_weights):
        base_weights = "yolov8n.pt"
    
    # Initialize YOLOv8 nano model from local weights
    model = YOLO(base_weights)
    
    # Train model with data augmentation to combat weather & lighting variations (Jiang 2024)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device="cpu",
        workers=0,  # avoid multiprocessing issues on Windows
        project=output_model_dir,
        name="pothole_yolo_run",
        exist_ok=True,
        # Augmentation hyperparameters for road distress & lighting shifts
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.8,
        verbose=True
    )
    
    # Validate the trained model
    print("\nEvaluating YOLOv8 validation metrics...")
    metrics = model.val(data=data_yaml_path, split="val")
    
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print("\n" + "=" * 50)
    print("YOLOv8 POTHOLE DETECTION EVALUATION METRICS:")
    print(f"  Precision:       {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:          {recall:.4f} ({recall*100:.2f}%)")
    print(f"  mAP@0.5:         {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95:    {map50_95:.4f} ({map50_95*100:.2f}%)")
    print("=" * 50 + "\n")
    
    # Copy best weights to standardized location
    best_weights = os.path.join(output_model_dir, "pothole_yolo_run", "weights", "best.pt")
    target_weights = os.path.join(output_model_dir, "pothole_yolov8_best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"Best model weights saved to: {target_weights}")
    else:
        last_weights = os.path.join(output_model_dir, "pothole_yolo_run", "weights", "last.pt")
        if os.path.exists(last_weights):
            shutil.copy(last_weights, target_weights)
            print(f"Last model weights saved to: {target_weights}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "map50": float(map50),
        "map50_95": float(map50_95),
        "model_path": target_weights
    }

if __name__ == "__main__":
    train_pothole_detector(epochs=8)
