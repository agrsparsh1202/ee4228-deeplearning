import os
import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Initialize FaceNet model (for embeddings)
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=40, keep_all=False, thresholds=[0.6, 0.7, 0.7], min_face_size=40, 
              device='cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomResizedCrop(160, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset Class
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)

        if face is None:
            return None
        
        face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if self.transform:
            face_tensor = self.transform(face_np)
        else:
            face_tensor = transforms.ToTensor()(face_np)
        
        return face_tensor, label

# Load Dataset
def load_dataset(dataset_path, test_size=0.3):
    image_paths, labels = [], []
    class_names = sorted(os.listdir(dataset_path))
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(class_name)
    
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    
    return train_images, train_labels, test_images, test_labels, class_names

# Define FaceClassifier Model
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

if __name__ == "__main__":
    dataset_path = "dataset"  # Ensure dataset folder exists
    train_images, train_labels, test_images, test_labels, class_names = load_dataset(dataset_path)
    
    # Create Dataset Loaders
    train_dataset = FaceDataset(train_images, train_labels, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    num_classes = len(class_names)
    label_map = {label: idx for idx, label in enumerate(class_names)}

    # Initialize Model
    model = FaceClassifier(embedding_dim=512, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            if images is None:
                continue
            
            images = images.to(device)
            labels = torch.tensor([label_map[label] for label in labels]).to(device)
            
            with torch.no_grad():
                embeddings = facenet(images).to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.2f}")

    # Save Model
    with open("face_model.pkl", "wb") as f:
        pickle.dump({'model': model.state_dict(), 'class_names': class_names}, f)
    
    print("Model trained and saved successfully.")
