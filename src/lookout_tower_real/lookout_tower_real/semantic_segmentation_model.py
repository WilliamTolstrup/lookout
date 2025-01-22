import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF

# 1. Dataset Class
class FloorDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_size=(480, 640)):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.startswith('img') and f.endswith('.png')])
        self.labels = sorted([f for f in os.listdir(image_dir) if f.startswith('label') and f.endswith('.png')])
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.image_dir, self.labels[idx])
        
        # Load image and label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        
        # Resize image and label to target size
        image = TF.resize(image, self.target_size)
        label = TF.resize(label, self.target_size, interpolation=Image.NEAREST)

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor (long type for class indices)
        label = torch.as_tensor(np.array(label), dtype=torch.long)

        return image, label


# 2. Data Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
dataset = FloorDataset(image_dir="/home/william/Datasets/floorspace/semantic_segmentation", transform=transform)

# DEBUG
for i in range(len(dataset)):
    image, label = dataset[i]
    print(f"Image shape: {image.shape}, Label shape: {label.shape}")
    print("Unique values in label:", np.unique(label))

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels



train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 4. Model Setup
num_classes = 5

model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)  # Change final layer to match 3 classes
model.aux_classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=1)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 5. Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 6. Training Loop
num_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# 6.5 Save model
# Specify the path to save the model
save_path = "/home/william/repos/lookout/src/lookout_tower_real/segmentation_model.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), save_path)

print(f"Model saved to {save_path}")


# 7. Validation Loop
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)

        # Display results for inspection
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(labels[0].cpu().numpy())
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(preds[0].cpu().numpy())
        plt.show()
