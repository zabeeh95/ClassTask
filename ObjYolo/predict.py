import os
from ultralytics import YOLO
import cv2

if __name__ == "__main__":

    # Use the exported model file
    model = YOLO("runs/detect/train/weights/best.pt")

    # Threshold for detection
    threshold = 0.2

    # Initialize webcam (default is index 0)
    cap = cv2.VideoCapture(0)

    # Check if webcam is accessible as it return a boolean
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    while True:
        # Capture a frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        results = model(frame)[0]

        # Loop through detections and draw bounding boxes
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Add label
                label = f"{results.names[int(class_id)]}: {score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Live Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
