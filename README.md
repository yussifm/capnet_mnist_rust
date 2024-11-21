
# CapNet MNIST Rust Project

This repository provides a Rust-based implementation of neural network models for classifying images from the MNIST dataset and exploring CIFAR-10 classification. The project demonstrates the use of Rust in deep learning, leveraging libraries like `tch-rs` (for Tensor operations using PyTorch bindings in Rust), `mnist` (for handling the MNIST dataset), and `plotters` (for visualizing training metrics such as loss and accuracy).

## Features

- **Automatic Dataset Handling**  
  - Automatically downloads and extracts the MNIST dataset if not already available.
  - Supports additional datasets like CIFAR-10 for extended experimentation.
  
- **Dataset Processing**  
  - Normalizes image pixel values to range between 0 and 1 for better model convergence.

- **Neural Network Implementation**  
  - Implements simple fully connected layers for classifying handwritten digits (MNIST).
  - Designed to be extendable to other datasets like CIFAR-10.

- **Training and Visualization**  
  - Trains the model using the Adam optimizer over multiple epochs.
  - Generates and saves visual plots of training metrics (loss and accuracy) using the `plotters` library.

- **CUDA Support**  
  - Leverages CUDA for accelerated computation if available.

## Installation

Ensure you have Rust installed. You can install Rust using [rustup](https://rustup.rs/).

Clone the repository:

```bash
git clone https://github.com/yussifm/capnet_mnist_rust.git
cd capnet_mnist_rust
```

Build and run the project:

```bash
cargo run --release
```

## Usage

1. **Data Handling**  
   The project processes and normalizes datasets into a format suitable for neural network inputs.

2. **Model Training**  
   Train the neural network by running the main script. Adjust training parameters such as epochs and batch size in the code.

3. **Metrics Visualization**  
   During training, loss and accuracy metrics are saved as image files in the project directory for analysis.

## Dependencies

The project relies on the following libraries:

- [tch-rs](https://github.com/LaurentMazare/tch): PyTorch bindings for Rust.
- [tch-rs](https://crates.io/crates/tch): Crate PyTorch bindings for Rust.
- [ndarray](https://docs.rs/ndarray/latest/ndarray/): For numerical array operations.
- [ndarray](https://crates.io/crates/ndarray): Crate For numerical array operations.
- [mnist](https://docs.rs/mnist/latest/mnist/): For MNIST dataset handling.
- [mnist](https://crates.io/crates/mnist): Crate For MNIST dataset handling.
- [plotters](https://docs.rs/plotters/latest/plotters/): For generating training metric plots.
- [plotters](https://crates.io/crates/plotters):Crate For generating training metric plots.

## Data Sources

The project utilizes datasets from the following sources:

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): A dataset for image classification tasks.
- [MNIST](https://github.com/zalandoresearch/fashion-mnist): A dataset for handwritten digit recognition.
  
## Download LibTorch for Local setup
- [LibTorch](https://pytorch.org/get-started/locally/): required to run and compile.
if you are using windows add **LibTorch** folder to  **"C:\LibTorch"** and to path 



### Where to Get Data for `load_other_data`

You can use data from various sources, depending on your use case. Here are some suggestions:

1. **Open Data Repositories**
   - [Kaggle](https://www.kaggle.com/): Hosts datasets across numerous fields such as machine learning, finance, and healthcare.
   - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php): Offers classic datasets for experimentation.
   - [Data.gov](https://www.data.gov/): U.S. government open data portal.
   - [Google Dataset Search](https://datasetsearch.research.google.com/): Search datasets across the web.

2. **Synthetic Data**
   - Generate data using Python libraries like `numpy` or `pandas`.
   - Useful for testing and debugging when no real-world data is available.

3. **Custom Data**
   - Collect your own data relevant to your project using surveys, sensors, or APIs.

---

### Formatting and Preparing Data for `load_other_data`

The function expects two CSV files:
1. `features.csv`: Contains input data (independent variables).
2. `labels.csv`: Contains corresponding labels or output data (dependent variables).

#### Steps to Prepare the Data:
1. **Ensure Proper File Names and Structure**
   - Save the input features in a file named `features.csv`.
   - Save the labels in a file named `labels.csv`.
   - Place these files in the same directory.

2. **Data Format for `features.csv`**
   - Each row represents one data point.
   - Each column represents a feature.
   - Ensure numeric values for all features.
   - Example:
     ```
     1.5, 2.3, 3.1
     4.2, 5.1, 6.3
     7.4, 8.2, 9.0
     ```

3. **Data Format for `labels.csv`**
   - Each row contains a single numeric label corresponding to a row in `features.csv`.
   - Ensure the labels align with the rows in `features.csv`.
   - Example:
     ```
     0
     1
     0
     ```



## Contributing

Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

