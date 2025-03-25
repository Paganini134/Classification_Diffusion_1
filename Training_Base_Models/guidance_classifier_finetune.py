import torch
from torch import nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score  # <-- Import metrics here
import subprocess

import os
import torch
import pandas as pd

torch.cuda.empty_cache()
subprocess.run(["python", "/home/iotlab/Desktop/CARD-MAIN/classification/guidance_renet50.py"])
# Directory to save results
save_path = "/home/iotlab/Desktop/CARD-MAIN/diffusion_model_plots"
os.makedirs(save_path, exist_ok=True)
torch.cuda.empty_cache()
# Set up the device for GPU usage if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Augmentation and Normalization
data_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(15),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Datasets
med_data_path = "/home/iotlab/Desktop/CARD-MAIN"
train_data_path = f"{med_data_path}/Data/train"
test_data_path = f"{med_data_path}/Data/test"

train_data = datasets.ImageFolder(root=train_data_path, transform=data_transform)
test_data = datasets.ImageFolder(root=test_data_path, transform=test_transform)

# DataLoader
BATCH_SIZE = 16
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

# ✅ Model Initialization Function
def get_modified_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported model name")

    # Modify First Conv2d Layer for 128x128 Input
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)

    # Modify Last Fully Connected (FC) Layer for 4 Classes
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 4)
    elif hasattr(model, "aux_logits"):  # GoogLeNet Special Case
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, 4)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, 4)

    return model.to(device)

# ✅ Training Function
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)

# ✅ Testing Function (Now includes Precision and Recall)
def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc, total_nll, total_mse = 0, 0, 0, 0
    all_preds, all_targets = [], []
    total_samples = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_softmax = torch.softmax(y_pred, dim=1)
            y_pred_class = torch.argmax(y_pred_softmax, dim=1)

            test_acc += (y_pred_class == y).sum().item()
            total_nll += F.nll_loss(F.log_softmax(y_pred, dim=1), y).item()
            total_mse += F.mse_loss(y_pred_softmax, F.one_hot(y, num_classes=4).float()).item()

            all_preds.extend(y_pred_class.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            total_samples += y.size(0)

    # Calculate F1, Precision, and Recall (macro-averaged)
    total_f1 = f1_score(all_targets, all_preds, average="macro")
    total_precision = precision_score(all_targets, all_preds, average="macro")
    total_recall = recall_score(all_targets, all_preds, average="macro")

    return (
        test_loss / len(dataloader),             # CrossEntropy
        test_acc / total_samples,                # Accuracy
        total_nll / len(dataloader),             # Negative Log Likelihood
        total_mse / len(dataloader),             # MSE
        total_f1,                                # F1 Score
        total_precision,                         # Precision
        total_recall                             # Recall
    )

# ✅ Full Training Pipeline (Tracks Metrics Per Epoch)
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, model_name):
    best_test_acc = 0.0  
    BEST_MODEL_PATH = os.path.join(save_path, f"{model_name}_best_weights.pth")
    epoch_list, acc_list, nll_list, ce_list, mse_list = [], [], [], [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc, test_nll, test_mse, _, _, _ = test_step(model, test_dataloader, loss_fn, device)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        epoch_list.append(epoch + 1)
        acc_list.append(test_acc)
        nll_list.append(test_nll)
        ce_list.append(test_loss)
        mse_list.append(test_mse)

        # Save the best model weights based on highest test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)  
            print(f"New best model saved for {model_name} with Test Accuracy: {best_test_acc:.4f}")

    # Save Metrics per Epoch
    df_metrics = pd.DataFrame({
        "Epoch": epoch_list,
        "Accuracy": acc_list,
        "NLL": nll_list,
        "CrossEntropy": ce_list,
        "MSE": mse_list
    })
    df_metrics.to_csv(os.path.join(save_path, f"{model_name}_epoch_metrics.csv"), index=False)

# ✅ Train and Evaluate Models
EPOCHS = 100
loss_fn = nn.CrossEntropyLoss()


# This runs "other_file.py" as a new process

results = []
model_names = ["resnet18", "googlenet"]

for model_name in model_names:
    print(f"\n===== Training {model_name.upper()} =====")
    model = get_modified_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train(model, train_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, device, model_name)

    # Load Best Model and Evaluate
    best_model = get_modified_model(model_name)
    BEST_MODEL_PATH = os.path.join(save_path, f"{model_name}_best_weights.pth")
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device), strict=False)
    best_model.to(device)
    best_model.eval()

    test_loss, test_acc, test_nll, test_mse, test_f1, test_precision, test_recall = test_step(
        best_model, test_dataloader, loss_fn, device
    )
    
    results.append({
        "Model": model_name,
        "Accuracy": test_acc,
        "F1 Score": test_f1,
        "Precision": test_precision,
        "Recall": test_recall,
        "CrossEntropy": test_loss,
        "MSE": test_mse,
        "NLL": test_nll
    })

# Define the path to the CSV file containing the resnet50 results.
csv_path = "/home/iotlab/Desktop/CARD-MAIN/diffusion_model_plots/model_comparison_results.csv"
# Check if the CSV file exists.
if os.path.exists(csv_path):
    # Read the existing CSV into a DataFrame.
    existing_df = pd.read_csv(csv_path)
    # Convert the new results to a DataFrame.
    new_df = pd.DataFrame(results)
    # Concatenate the existing and new results.
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    # If the CSV does not exist, create a new DataFrame with the results.
    updated_df = pd.DataFrame(results)

# Save the updated DataFrame back to CSV.
updated_df.to_csv(csv_path, index=False)
