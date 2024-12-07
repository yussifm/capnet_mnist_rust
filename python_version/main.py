import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



def squash(x):
    """
    Squashes the input values to a range between -1 and 1 using a hyperbolic tangent (tanh) function.
    """
    return np.tanh(x)


def normalize(data):
    """
    Normalizes the input data to a range between 0 and 1.
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)



def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)  # Avoids requiring pandas
    data, labels = mnist.data, mnist.target.astype(int)

    # Normalize and squash data
    data = normalize(data)
    data = squash(data)

    # Updated to use 'sparse_output' instead of 'sparse'
    labels = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print("MNIST dataset loaded successfully.")
    return x_train, y_train, x_test, y_test



def plot_metrics(losses, accuracies):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'r-', label="Loss")
    plt.plot(epochs, accuracies, 'b-', label="Accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig("training_metrics.png")
    print("Training metrics plotted and saved as 'training_metrics.png'.")


def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    print("Choose the dataset to use:")
    print("1 for MNIST")
    choice = input("Enter your choice: ").strip()
    if choice != "1":
        print("Invalid choice! Defaulting to MNIST.")
    x_train, y_train, x_test, y_test = load_mnist()

    print("Enter the number of training epochs: ")
    epochs = input().strip()
    epochs = int(epochs) if epochs.isdigit() else 10

    model = build_model(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2)

    losses = history.history['loss']
    accuracies = history.history['accuracy']
    plot_metrics(losses, accuracies)

    print("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
