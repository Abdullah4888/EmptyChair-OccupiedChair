import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4-custom_4000.weights", "yolov4-custom.cfg")

# Load classes from predefined file
classes = []
with open("/home/Abdullah/Downloads/Data/ESS/ChairDetection/predefined_classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO network layer names and output layer names
layer_names = net.getLayerNames()
output_layer = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Read and resize the input image
img = cv2.imread("chair.jpeg")
print(img.shape)
img = cv2.resize(img, None, fx=3.55, fy=2.55)
height, width, channels = img.shape
print(img.shape)

# Display the input image
cv2.imshow("ESS", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Preprocess the image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set input for YOLO network and get the output
net.setInput(blob)
outs = net.forward(output_layer)

# Initialize lists to store detected objects
class_ids = []
confidences = []
boxes = []

# Process the YOLO output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)
print(len(indexes))

# Draw bounding boxes and labels on the image
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        accuracy = str(int(round(confidences[i] * 100, 0))) + "%"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(img, label + ":" + accuracy, (x, y + 25), font, 1, (255, 0, 0), 2)

# Display the image with detections
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
