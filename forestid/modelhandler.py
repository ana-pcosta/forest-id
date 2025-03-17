from torch.hub import load
from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(
        self,
        num_classes: int,
        backbone_model_family: str = "facebookresearch/dinov2",
        backbone_model_name: str = "dinov2_vitl14_reg",
    ):
        model = load(backbone_model_family, backbone_model_name)
        num_features = model.num_features

        classification_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        model.head = classification_head
        model = model.to(self.device)
        return model

    def train(self, model, train_loader, val_loader, num_epochs: int, lr: float):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            model.train()
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True
            )
            running_loss = 0.0
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            # Validation phase
            model.eval()  # Set the model to evaluation mode
            val_running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():  # No gradients needed for validation
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(self.device), val_labels.to(
                        self.device
                    )
                    val_outputs = model(val_images)
                    val_running_loss += criterion(val_outputs, val_labels).item()

                    _, predicted = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            val_loss = val_running_loss / len(val_loader)  # Average validation loss
            val_accuracy = 100 * correct / total  # Accuracy in percentage

            # Update the progress bar description with validation results
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {running_loss / (progress_bar.n + 1):.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

        print("Training complete.")

    def test(self, model, test_loader):
        # Set the model to evaluation mode
        model.eval()
        predictions = []
        probabilities = []

        # Disable gradient calculations (since we're doing inference)
        with torch.no_grad():
            # Loop over the test set
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Get model predictions
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                probs = torch.softmax(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return predictions, probabilities
