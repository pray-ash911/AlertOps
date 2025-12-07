from ultralytics import YOLO
import torch


def download_yolov8m():
    """
    Download YOLOv8 medium model using ultralytics
    """
    print("Downloading YOLOv8m model...")

    # This will automatically download the model if not already present
    model = YOLO('yolov8m.pt')

    # Test the model to ensure it's loaded
    print("Model downloaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model device: {model.device}")

    return model


# Usage
if __name__ == "__main__":
    model = download_yolov8m()
    # Model is now ready for inference