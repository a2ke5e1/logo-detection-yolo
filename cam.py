import os
from ultralytics import YOLO
import cv2

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'last.pt')
model = YOLO(model_path)

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get the frame width and height
ret, frame = cap.read()
H, W, _ = frame.shape

# Initialize video writer to save the output
video_path_out = 'camera_output.mp4'
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Detection threshold
threshold = 0.5

# Process video frames
while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('Logo Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()