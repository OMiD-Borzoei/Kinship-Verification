import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Load the pre-trained VGG model (use appropriate model pre-trained on VGGFace2)
model = models.vgg16(pretrained=True)  # Replace with specific VGGFace2 model if needed

# Modify the classifier's last layer to output 256 or 512 features
output_features = 256  # Set to 512 if needed
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, output_features)

# Set the model to evaluation mode
model.eval()

# Define a transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 as expected by VGG
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    
    with torch.no_grad():  # Disable gradient calculation for inference
        features = model(input_tensor)
    return features.squeeze().numpy()  # Convert to numpy array if needed

# Directory containing face images
data_dir = r'D:\Uni\Term 9\Project\KIN_I\aug_kinfacewi_dataset_1024x1024\fd_001_1'

# Iterate through images and extract features
features_dict = {}
for filename in os.listdir(data_dir):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):  # Filter image files
        image_path = os.path.join(data_dir, filename)
        features = extract_features(image_path)
        features_dict[filename] = features
        print(f"Extracted features for {filename}: {features.shape}")

# Example: Save features to a file if needed
import numpy as np
np.save('features.npy', features_dict)
