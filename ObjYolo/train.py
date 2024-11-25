from ultralytics import YOLO
import torch

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())  # Should be True
print("CUDA Device Count:", torch.cuda.device_count())  # Number of GPUs
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))  # GPU name


if __name__ == '__main__':
    print("Initializing YOLO model...")
    """.pt is pre trained model or custom model can be used directly with.yaml extension"""

    # build a new model from scratch
    # model = YOLO("yolov8n.yaml")

    model = YOLO('yolo11n.pt')  # Load your model file
    # print(model.info)

    print("Starting training...")
    try:
        train_results = model.train(
            data="data.yaml",  # Dataset YAML file
            epochs=3,  # Training epochs
            # name='train_debug',  # Experiment name
            batch=16,  # Batch size
            imgsz=640,  # Image size
            device="cuda",  # Use GPU
            # workers=0,  # Disable multiprocessing for debugging
        )
        print("Training completed successfully.")

    except Exception as e:
        print("Training failed:", e)
