import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer


# Function to apply the sigmoid function element-wise to normalize dataset values
def squash(value):
    value = np.clip(value, -500, 500)  # Clip extreme values
    return 1.0 / (1.0 + np.exp(-value))

# Normalizes a dataset using the squash function
def normalize_dataset(dataset):
    return squash(dataset)

# Load MNIST dataset
def load_mnist():
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy() / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy() / 255.0
    y_test = test_dataset.targets.numpy()

    return x_train, y_train, x_test, y_test

# Load Fashion MNIST dataset
def load_fashion_mnist():
    print("Loading Fashion MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy() / 255.0
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy() / 255.0
    y_test = test_dataset.targets.numpy()

    return x_train, y_train, x_test, y_test

# Load CIFAR-10 dataset
def load_cifar10():
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    x_train = train_dataset.data / 255.0
    y_train = np.array(train_dataset.targets)
    x_test = test_dataset.data / 255.0
    y_test = np.array(test_dataset.targets)

    return x_train, y_train, x_test, y_test

# Load custom training and testing datasets from CSV

# Load custom training and testing datasets from CSV
def load_data(train_file_path, test_file_path, task_type='classification'):
    """
    Load and preprocess data from CSV files.
    """
    print(f"Loading training data from {train_file_path}...")
    train_data = pd.read_csv(train_file_path)

    print(f"Loading testing data from {test_file_path}...")
    test_data = pd.read_csv(test_file_path)

    # Target column: 'pathology'
    target_column = 'pathology'

    # Handle missing values and preprocess features
    numeric_columns = train_data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()

    # Remove target column from feature lists
    numeric_columns = [col for col in numeric_columns if col != target_column]
    categorical_columns = [col for col in categorical_columns if col != target_column]

    # Fill missing numeric values with median
    imputer = SimpleImputer(strategy="median")
    train_data[numeric_columns] = imputer.fit_transform(train_data[numeric_columns])
    test_data[numeric_columns] = imputer.transform(test_data[numeric_columns])

    # Fill missing categorical values with mode
    for col in categorical_columns:
        train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

    # One-hot encode categorical columns
    train_data = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)
    test_data = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

    # Align train and test datasets
    train_data, test_data = train_data.align(test_data, join='inner', axis=1, fill_value=0)

    # Split features and target
    x_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Ensure all data is numeric
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    # Normalize all features (since they're all numeric now)
    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)

    # Encode target for classification tasks
    if task_type in ['classification', 'c']:
        label_encoder = LabelEncoder()
        y_train = torch.tensor(label_encoder.fit_transform(y_train), dtype=torch.long)
        y_test = torch.tensor(label_encoder.transform(y_test), dtype=torch.long)
    elif task_type in ['regression', 'r']:
        y_train = torch.tensor(y_train.values.astype(np.float32), dtype=torch.float32)
        y_test = torch.tensor(y_test.values.astype(np.float32), dtype=torch.float32)

    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")
    print(f"Number of features: {x_train.shape[1]}")
    if task_type in ['classification', 'c']:
        print(f"Number of classes: {len(torch.unique(y_train))}")

    return x_train, y_train, x_test, y_test

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plot_metrics(losses, accuracies):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, label="Loss", color="red")
    plt.plot(epochs, accuracies, label="Accuracy", color="blue")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("training_metrics.png")
    plt.show()


# Evaluate model and display confusion matrix and additional metrics
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize

def evaluate_model(model, x_test, y_test, device, task_type='classification'):
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        
        if task_type in ['regression', 'r']:
            test_outputs = test_outputs.squeeze()
            y_true = y_test.cpu().numpy()
            y_pred = test_outputs.cpu().numpy()

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            print(f"\nEvaluation Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        elif task_type in ['classification', 'c']:
            # Existing classification evaluation code remains the same
            probabilities = torch.softmax(test_outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            y_true = y_test.cpu().numpy()
            y_pred = predictions

            # Rest of the classification evaluation code...

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(len(np.unique(y_true))))
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()

            # Calculate and display classification report
            report = classification_report(y_true, y_pred, digits=4)
            print("\nClassification Report:\n")
            print(report)

            # Compute ROC curve and AUC for each class
            n_classes = probabilities.shape[1]
            y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))
            fpr, tpr, roc_auc = {}, {}, {}

            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], probabilities[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.4f})")

            # Plot diagonal
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.show()

            # Calculate macro-average AUC
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)
            print(f"\nMacro-average AUC: {macro_auc:.4f}")


# Main function
def main():
    dataset_choice = int(input(
        "Choose the dataset to use (1 for MNIST, 2 for Fashion MNIST, 3 for CIFAR-10, 4 for Custom CSV Data): ").strip())
    epochs = int(input("Enter the number of training epochs: ").strip())

    # Load dataset
    if dataset_choice == 1:
        x_train, y_train, x_test, y_test = load_mnist()
       #print("X_train shape:", x_train.shape)
        #print("y_train shape:", y_train.shape)
        #print("X_test shape:", x_test.shape)
        #print("y_test shape:", y_test.shape)
        input_dim, output_dim = 28 * 28, 10
        task_type = "classification"
    elif dataset_choice == 2:
        x_train, y_train, x_test, y_test = load_fashion_mnist()
       #print("X_train shape:", x_train.shape)
        #print("y_train shape:", y_train.shape)
        #print("X_test shape:", x_test.shape)
        #print("y_test shape:", y_test.shape)
        input_dim, output_dim = 28 * 28, 10
        task_type = "classification"
    elif dataset_choice == 3:
        x_train, y_train, x_test, y_test = load_cifar10()
       #print("X_train shape:", x_train.shape)
        #print("y_train shape:", y_train.shape)
        #print("X_test shape:", x_test.shape)
        #print("y_test shape:", y_test.shape)
        input_dim, output_dim = 32 * 32 * 3, 10
        task_type = "classification"
    elif dataset_choice == 4:
        train_file_path = input("Enter the path to the training CSV file: ").strip()
        test_file_path = input("Enter the path to the testing CSV file: ").strip()
        task_type = input("Enter the task type (classification or regression): ").strip().lower()
            # Add print statements to debug data shapes
       #print("X_train shape:", x_train.shape)
        #print("y_train shape:", y_train.shape)
        #print("X_test shape:", x_test.shape)
        #print("y_test shape:", y_test.shape)
        if task_type not in ['classification', 'regression', 'c', 'r']:
            raise ValueError("Invalid task type! Choose 'classification' or 'regression'.")
        x_train, y_train, x_test, y_test = load_data(train_file_path, test_file_path, task_type)
        input_dim = x_train.shape[1]
        output_dim = len(np.unique(y_train)) if task_type in ['classification', 'c'] else 1
    else:
        raise ValueError("Invalid dataset choice!")

    # Flatten and convert to PyTorch tensors
    
    x_train = torch.tensor(x_train.reshape(-1, input_dim), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32 if task_type in ['regression', 'r'] else torch.long)
    x_test = torch.tensor(x_test.reshape(-1, input_dim), dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32 if task_type in ['regression', 'r'] else torch.long)

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_dim, output_dim).to(device)
    criterion = nn.MSELoss() if task_type in ['regression', 'r'] else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    losses, accuracies = [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train.to(device))
        loss = criterion(outputs.squeeze(), y_train.to(device))
        loss.backward()
        optimizer.step()

        # Record training loss
        losses.append(loss.item())

        # Compute accuracy for classification tasks
        if task_type in ['classification', 'c']:
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions.cpu() == y_train.cpu()).float().mean().item()
                accuracies.append(accuracy)
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

    # Plot training metrics
    if task_type in ['classification', 'c']:
        plot_metrics(losses, accuracies)

    # Evaluate model
    print("\nEvaluating model on test data...")
    evaluate_model(model, x_test.to(device), y_test.to(device), device, task_type)


if __name__ == "__main__":
    main()
