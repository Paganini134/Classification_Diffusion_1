import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader

# Set device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define base directory and data paths (using "Car_Data")
base_dir = "/home/iotlab/Desktop/CARD-MAIN/classification/Car_Data"
train_data_path = os.path.join(base_dir, "train")
validation_data_path = os.path.join(base_dir, "val")
test_data_path = os.path.join(base_dir, "test")

# Define data transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create datasets using ImageFolder
train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=data_transform)
validation_dataset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=data_transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Dictionary for dataset sizes
dataset_sizes = {
    "train": len(train_dataset),
    "val": len(validation_dataset),
    "test": len(test_dataset)
}
class_names = train_dataset.classes
print("Classes:", class_names)

# Load a pretrained ResNet-18 model and modify for binary classification
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

# For one-phase training, we train all layers together.
for param in model.parameters():
    param.requires_grad = True

# Define loss function and optimizer (use a lower learning rate for fine-tuning)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Single-phase training function
def train_model_single_phase(model, criterion, optimizer, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        
        # We'll do training and validation in each epoch
        for phase, loader in [("train", train_loader), ("val", validation_loader)]:
            if phase == "train":
                model.train()  # training mode
            else:
                model.eval()   # evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Save best model weights based on validation accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    
    print("Best val Acc: {:.4f}".format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

# Train the entire model in one phase
print("Training entire model in one phase...")
model = train_model_single_phase(model, criterion, optimizer, num_epochs=15)

# Evaluate on test set
model.eval()
running_corrects = 0
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device).float().unsqueeze(1)
    outputs = model(inputs)
    preds = (torch.sigmoid(outputs) > 0.5).float()
    running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects.double() / dataset_sizes["test"]
print("Test Accuracy: {:.4f}".format(test_acc))

# Save the final model weights
weights_path = "resnet18_finetuned_weights.pth"
torch.save(model.state_dict(), weights_path)
print(f"\nModel weights saved to {weights_path}")

# Load the saved weights into a new model and perform a forward pass
loaded_model = models.resnet18(pretrained=False)
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_ftrs, 1)
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Load state dict with strict=False to avoid key mismatches (if any)
loaded_state_dict = torch.load(weights_path)
missing_keys, unexpected_keys = loaded_model.load_state_dict(loaded_state_dict, strict=False)
if missing_keys:
    print("Missing keys:", missing_keys)
if unexpected_keys:
    print("Unexpected keys:", unexpected_keys)

# Create a dummy input tensor and run a forward pass
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = loaded_model(dummy_input)
print("\nForward pass output using the loaded weights:")
print(output)
