
# Dataset Training and Metrics Visualization

This Python program allows you to load datasets such as MNIST or custom CSV datasets, train a simple neural network, and visualize training metrics (loss and accuracy) using Matplotlib. It leverages PyTorch for training and provides an easy way to experiment with different datasets.

## Features

- Load and normalize the MNIST dataset or a custom CSV dataset.
- Train a fully connected neural network.
- Plot and save training loss and accuracy metrics as an image.
- Use CPU or GPU for training (if available).

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- torchvision

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yussifm/capnet_mnist_rust/tree/load_and_format_csv
   cd capnet_mnist_rust/python_version
   ```

2. **Create and activate a virtual environment**:
   - **MacOS/Linux**:
     ```bash
     python3 -m venv env
     source env/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv env
     env\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```

2. **Select dataset**:
   - Choose `1` to use the MNIST dataset.
   - Choose `2` to use a custom CSV dataset.
     - Provide the path to the CSV file when prompted.

3. **Specify training parameters**:
   - Enter the number of epochs for training when prompted.

4. **View results**:
   - The script will display the training progress (loss and accuracy per epoch).
   - Training metrics will be saved as a plot (`training_metrics.png`) in the current directory.

## Dataset Format

- **MNIST**:
  The MNIST dataset will be automatically downloaded and processed.

- **Custom CSV**:
  - The dataset should be a CSV file where:
    - All feature columns are numeric.
    - The last column represents the target labels.
  - Example:
    ```
    feature1,feature2,feature3,label
    0.5,0.8,0.3,1
    0.2,0.4,0.6,0
    ```

## Project Structure

```plaintext
capnet_mnist/
├── python_version/
│   ├── main.py             # Main script to train and evaluate the model
│   ├── requirements.txt    # List of dependencies
│   ├── data/               # Directory for datasets
│   └── training_metrics.png # Output plot of training metrics
```

## Contributing

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Let me know if you'd like to modify anything further!