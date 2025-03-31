import os
import cv2
import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Initialize FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=40, keep_all=True, thresholds=[0.4, 0.5, 0.6], min_face_size=15, device='cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face tracking parameters
FACE_TRACKING_THRESHOLD = 50  # Maximum pixel distance for face tracking

# Define FaceClassifier model
class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Define Transformations
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load trained model
def load_model(model_path):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = FaceClassifier(embedding_dim=512, num_classes=len(data['class_names']))
    model.load_state_dict(data['model'])
    model.to(device).eval()
    return model, data['class_names']

# Function to recognize face
def recognize_face(face_img, model, class_names):
    if face_img is None or face_img.shape[0] == 0 or face_img.shape[1] == 0:
        return "Unknown"

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (160, 160))
    face_tensor = test_transform(face_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = facenet(face_tensor)
        output = model(embedding)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.max(output).item()
        return class_names[pred_idx] if confidence > 0.8 else "Unknown"

# Calculate center point of a bounding box
def get_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

# Run webcam for face recognition
def run_webcam(model_path="face_model.pkl"):
    model, class_names = load_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Face tracking state
    tracked_faces = []  # List to store currently tracked faces
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            print(f"Detected {len(boxes)} faces")  # Debugging log
            
            # Process each detected face
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                h, w, _ = frame.shape
                
                # Ensure bounding box is within the frame limits
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:  # Invalid box dimensions
                    continue
                
                face_img = frame[y1:y2, x1:x2]
                label = recognize_face(face_img, model, class_names)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("No faces detected")  # Debugging log
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Webcam Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()

