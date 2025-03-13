import cv2
import pickle
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer

# Initialize Face Detector and FaceNet Model
detector = MTCNN(image_size=160, margin=20, keep_all=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

model_path = "face_model.pkl"

# Function to preprocess and encode a face image
def encode_face(face):
    face = cv2.resize(face, (160, 160))
    face = np.array(face, dtype=np.float32)
    face = (face - 127.5) / 127.5  # Normalize to [-1,1]
    face = np.transpose(face, (2, 0, 1))
    face_tensor = torch.tensor(face).unsqueeze(0)
    with torch.no_grad():
        embedding = facenet(face_tensor).numpy()[0]
    return embedding

# Function to recognize a face using the trained KNN model
def recognize_face(face):
    try:
        # Load the dictionary containing the model, normalizer, and class_names
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data['model']
        normalizer = data['normalizer']
    except FileNotFoundError:
        print("Trained model not found. Please run the training script first.")
        return "Unknown"
    except Exception as e:
        print(f"Error loading model: {e}")
        return "Unknown"
    
    embedding = encode_face(face)
    embedding = normalizer.transform([embedding])
    label = model.predict(embedding)[0]
    confidence = max(model.predict_proba(embedding)[0])
    return label if confidence > 0.8 else "Unknown"

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect face using MTCNN; returns a cropped face image if detected
        face = detector(frame)
        if face is not None:
            # Convert face tensor to a numpy array and adjust color for OpenCV display
            face_img = face.permute(1, 2, 0).int().numpy()
            label = recognize_face(face_img)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Webcam Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
