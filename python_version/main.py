# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


# Function to apply the sigmoid function element-wise to normalize dataset values
def squash(value):
    return 1.0 / (1.0 + np.exp(-value))

# Normalizes a dataset using the squash function
def normalize_dataset(dataset):
    return np.vectorize(squash)(dataset)

# Load the MNIST dataset, preprocess it, and normalize pixel values
# def load_mnist():
#     print("Loading MNIST dataset...")
#     transform = transforms.Compose([
#         transforms.ToTensor(),  # Convert images to PyTorch tensors
#         transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5
#     ])
#     # Download and prepare train and test datasets
#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#     # Flatten the images and scale pixel values to [0, 1]
#     x_train = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
#     y_train = train_dataset.targets.numpy()
#     x_test = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
#     y_test = test_dataset.targets.numpy()

#     return normalize_dataset(x_train), y_train, normalize_dataset(x_test), y_test

# fASHION-MNIST dataset UNCOMMENT THIS 
def load_mnist():
    print("Loading Fashion MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    x_train = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test = test_dataset.targets.numpy()

    return normalize_dataset(x_train), y_train, normalize_dataset(x_test), y_test


# Load custom datasets from CSV files
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)

    # Drop non-numeric columns if any
    data = data.select_dtypes(include=['number'])

    # Handle missing values
    data = data.fillna(0)

    # Convert to numpy arrays
    x_data = data.iloc[:, :-1].to_numpy(dtype=float)
    y_data = data.iloc[:, -1].to_numpy(dtype=int)

    return x_data, y_data


# Define a simple feedforward neural network
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(input_dim, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, output_dim)  # Second fully connected layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Plot training loss and accuracy over epochs
def plot_metrics(losses, accuracies):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, label="Loss", color="red")  # Plot loss
    plt.plot(epochs, accuracies, label="Accuracy", color="blue")  # Plot accuracy
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("training_metrics.png")  # Save plot to a file
    plt.show()

# Main function to manage dataset loading, training, and evaluation
def main():
    # Specify input and output folder paths for CSV datasets
    input_folder = "csv_data"
    output_folder = "csv_pro_data"
    label_column = "id"

    # User selects dataset type and number of training epochs
    dataset_choice = int(input("Choose the dataset to use (1 for MNIST, 2 for Custom CSV Data): ").strip())
    epochs = int(input("Enter the number of training epochs: ").strip())

    # Load dataset based on user choice
    if dataset_choice == 1:
        # Use MNIST dataset
        x_train, y_train, x_test, y_test = load_mnist()
        input_dim, output_dim = 784, 10  # Input and output dimensions
    elif dataset_choice == 2:
        # Use custom dataset from a CSV file
        print(os.listdir('data'))
        file_path = input("Enter the path to the CSV file: ").strip()
        x_data, y_data = load_data(file_path)
        train_size = int(0.8 * len(x_data))  # 80-20 train-test split
        x_train, y_train = x_data[:train_size], y_data[:train_size]
        x_test, y_test = x_data[train_size:], y_data[train_size:]
        input_dim, output_dim = x_data.shape[1], len(np.unique(y_data))  # Determine dimensions
    else:
        # Default to MNIST dataset
        print("Invalid choice! Defaulting to MNIST.")
        x_train, y_train, x_test, y_test = load_mnist()
        input_dim, output_dim = 784, 10

    # Set up the device, model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

    # Initialize lists to store losses and accuracies
    losses, accuracies = [], []
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Reset gradients
        outputs = model(x_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor) * 100

        # Log loss and accuracy
        losses.append(loss.item())
        accuracies.append(accuracy)
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    # Plot the training metrics
    plot_metrics(losses, accuracies)

# Entry point for the script
if __name__ == "__main__":
    main()
