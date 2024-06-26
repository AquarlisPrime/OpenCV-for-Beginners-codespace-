!pip install ultralytics
!pip install -U ipywidgets
!pip install ffmpeg-python

from ultralytics import YOLO
import glob
import os
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import seaborn as sns
import cv2
import random
from IPython.display import HTML, display, Image
from base64 import b64encode

dataset_root = '/kaggle/input/construction-site-safety-image-dataset-roboflow/css-data'

for subset_folder in ['train', 'valid', 'test']:
    subset_images = glob.glob(os.path.join(dataset_root, subset_folder, 'images','*.jpg'))
    print(f"Number of {subset_folder} images:", len(subset_images))

# Define augmentation transformations
aug_transform = A.Compose([
    A.RandomResizedCrop(640, 640, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# Custom dataset class with augmentation
class SafetyDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(self.root_dir, subset, 'images', '*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, img_path

# Paths to dataset
dataset_root = '/kaggle/input/construction-site-safety-image-dataset-roboflow/css-data'
train_data_path = os.path.join(dataset_root, 'train')
valid_data_path = os.path.join(dataset_root, 'valid')
test_data_path = os.path.join(dataset_root, 'test')

# Define augmentation transformations
aug_transform = A.Compose([
    A.RandomResizedCrop(640, 640, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create datasets and data loaders
train_dataset = SafetyDataset(root_dir=dataset_root, subset='train', transform=aug_transform)
valid_dataset = SafetyDataset(root_dir=dataset_root, subset='valid', transform=ToTensorV2())  # No need for augmentation in validation
test_dataset = SafetyDataset(root_dir=dataset_root, subset='test', transform=ToTensorV2())    # No need for augmentation in testing

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# Function to count images in each subset and visualize distribution
def count_images(dataset_root):
    subsets = ['train', 'valid', 'test']
    subset_counts = {}

    for subset in subsets:
        image_paths = glob.glob(os.path.join(dataset_root, subset, 'images', '*.jpg'))
        subset_counts[subset] = len(image_paths)

    return subset_counts

# Count images in each subset
subset_counts = count_images(dataset_root)
print("Number of images in each subset:")
print(subset_counts)

# Visualize distribution of images in each subset
plt.figure(figsize=(10, 6))
sns.barplot(x=list(subset_counts.keys()), y=list(subset_counts.values()))
plt.title('Distribution of Images in Dataset Subsets')
plt.xlabel('Subset')
plt.ylabel('Number of Images')
plt.show()

import random

# Function to display sample images from each subset
def display_sample_images(dataset_root, num_images_per_subset=3):
    subsets = ['train', 'valid', 'test']

    plt.figure(figsize=(15, 10))
    for i, subset in enumerate(subsets):
        image_paths = glob.glob(os.path.join(dataset_root, subset, 'images', '*.jpg'))
        sample_images = random.sample(image_paths, num_images_per_subset)

        for j, img_path in enumerate(sample_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(len(subsets), num_images_per_subset, i * num_images_per_subset + j + 1)
            plt.imshow(img)
            plt.title(f'Subset: {subset}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# Display sample images from each subset
display_sample_images(dataset_root, num_images_per_subset=3)


# Function to analyze class distribution (if annotations are available)
def analyze_class_distribution(dataset_root):
    subsets = ['train', 'valid', 'test']
    class_counts = {'protective_gear': 0, 'no_protective_gear': 0}

    for subset in subsets:
        label_paths = glob.glob(os.path.join(dataset_root, subset, 'labels', '*.txt'))
        for label_path in label_paths:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id == 0:
                        class_counts['protective_gear'] += 1
                    else:
                        class_counts['no_protective_gear'] += 1

    return class_counts

# Analyze class distribution
class_counts = analyze_class_distribution(dataset_root)
print("Class distribution:")
print(class_counts)


# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title('Class Distribution in Dataset')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.show()


# Analyze class distribution
class_counts = analyze_class_distribution(dataset_root)
print("Class distribution:")
print(class_counts)

# Plotting a pie chart for class distribution
labels = list(class_counts.keys())
sizes = list(class_counts.values())

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution: Protective Gear vs No Protective Gear')
plt.axis('equal')
plt.show()

# YOLO model initialization
model = YOLO('yolov8n.pt')

# YAML configuration for YOLO
yolo_yaml = {
    'path': dataset_root,
    'train': train_data_path,
    'val': valid_data_path,
    'test': test_data_path,
    'names': {
        0: 'Hardhat',
        1: 'Mask',
        2: 'NO-Hardhat',
        3: 'NO-Mask',
        4: 'NO-Safety Vest',
        5: 'Person',
        6: 'Safety Cone',
        7: 'Safety Vest',
        8: 'machinery',
        9: 'vehicle'
    }
}

# Save YAML configuration
with open('/kaggle/working/data.yaml', 'w') as file:
    yaml.dump(yolo_yaml, file)

model.train(data='/kaggle/working/data.yaml', epochs=75, batch=32, imgsz=640, name='Construction Safety Gear Detection')

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(12, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class SafetyDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(self.root_dir, subset, 'images', '*.jpg'))
        self.label_paths = [p.replace('images', 'labels').replace('.jpg', '.txt') for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [int(label.split()[0]) for label in labels]  # Extract class labels

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, labels, img_path

# Paths to dataset
valid_data_path = os.path.join(dataset_root, 'valid')

# Create datasets and data loaders
valid_dataset = SafetyDataset(root_dir=valid_data_path, subset='valid', transform=aug_transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define detection and annotation function
def detect_and_annotate(frame, model, class_map, threshold=0.5):
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(rgb_frame)

    # Parse results
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes with scores and class IDs

    persons = []
    protective_gear = []

    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if score > threshold:
            label = class_map[int(class_id)]
            if label == 'Person':
                persons.append((x1, y1, x2, y2, score, class_id))
            else:
                protective_gear.append((x1, y1, x2, y2, score, class_id, label))

    for (x1, y1, x2, y2, score, class_id) in persons:
        has_safety_gear = False
        for (gx1, gy1, gx2, gy2, gscore, gclass_id, glabel) in protective_gear:
            # Check if the protective gear is within the person's bounding box
            if x1 <= gx1 <= x2 and y1 <= gy1 <= y2:
                has_safety_gear = True
                break

        color = (0, 255, 0) if has_safety_gear else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'Person {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for (x1, y1, x2, y2, score, class_id, label) in protective_gear:
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Real-time video capture and display
def main():
    class_map = {
        0: 'Hardhat',
        1: 'Mask',
        2: 'NO-Hardhat',
        3: 'NO-Mask',
        4: 'NO-Safety Vest',
        5: 'Person',
        6: 'Safety Cone',
        7: 'Safety Vest',
        8: 'machinery',
        9: 'vehicle'
    }

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Perform detection and annotation
        annotated_frame = detect_and_annotate(frame, model, class_map)

        # Display the frame
        cv2.imshow('Safety Gear Detection', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




