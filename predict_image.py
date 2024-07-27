import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('runs/detect/train2', 'videos')
image_path = os.path.join(IMAGES_DIR, 'Obstacle55.jpg')
image_path_out = '{}_out.jpg'.format(image_path)

model_path = os.path.join('runs/detect/train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Read the image
frame = cv2.imread(image_path)
H, W, _ = frame.shape

# Process the image
results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the output image
cv2.imwrite(image_path_out, frame)

cv2.destroyAllWindows()
