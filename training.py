import os
import cv2
import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms

# Initialize models with improved parameters
facenet = InceptionResnetV1(pretrained='casia-webface').eval()  # Better for face recognition
detector = MTCNN(
    image_size=160,
    margin=40,  # Increased margin for better context
    keep_all=False,
    thresholds=[0.6, 0.7, 0.7],  # Lower detection thresholds
    min_face_size=40,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Enhanced augmentations using torchvision (for training images)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomResizedCrop(160, scale=(0.9, 1.1)),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Deterministic transform for test images (without randomness)
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_embedding(face_tensor):
    # face_tensor is expected to be of shape (3, 160, 160)
    with torch.no_grad():
        # Add batch dimension inside the function
        return facenet(face_tensor.unsqueeze(0)).squeeze().numpy()

def process_dataset(dataset_path, test_size=0.2):
    X_train, X_test, y_train, y_test = [], [], [], []
    class_names = sorted(os.listdir(dataset_path))
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Stratified split maintaining class balance
        train_files, test_files = train_test_split(images, test_size=test_size, random_state=42)
        
        # Process training images with augmentation
        for path in train_files:
            img = Image.open(path).convert('RGB')
            face = detector(img)
            if face is None:
                continue
            # Convert detected face (tensor shape: [3,160,160] with values in [0,1])
            # Multiply by 255 and convert to uint8 to mimic standard image format
            face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Generate multiple augmented versions
            for _ in range(10):
                augmented = train_transform(face_np)  # Returns tensor of shape (3,160,160)
                embedding = get_embedding(augmented)
                X_train.append(embedding)
                y_train.append(class_name)
        
        # Process test images without augmentation
        for path in test_files:
            img = Image.open(path).convert('RGB')
            face = detector(img)
            if face is None:
                continue
            # Process similar to training: convert tensor to np.uint8 image
            face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # Apply the deterministic test transform
            face_tensor = test_transform(face_np)  # shape: (3,160,160)
            embedding = get_embedding(face_tensor)  # get_embedding adds the needed batch dim internally
            X_test.append(embedding)
            y_test.append(class_name)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

if __name__ == "__main__":
    dataset_path = "dataset"  # Update this to your dataset folder
    X_train, y_train, X_test, y_test = process_dataset(dataset_path)
    
    # Normalize embeddings
    normalizer = Normalizer(norm='l2')
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    
    # KNN Classifier typically works better with few samples
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, knn.predict(X_train))
    test_acc = accuracy_score(y_test, knn.predict(X_test))
    
    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    
    # Save components
    try:
        with open("face_model.pkl", "wb") as f:
            pickle.dump({
                'model': knn,
                'normalizer': normalizer,
                'class_names': sorted(np.unique(y_train))
            }, f)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
