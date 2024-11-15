
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

- [tch-rs](https://github.com/LaurentMazare/tch-rs): PyTorch bindings for Rust.
- [ndarray](https://docs.rs/ndarray/latest/ndarray/): For numerical array operations.
- [mnist](https://docs.rs/mnist/latest/mnist/): For MNIST dataset handling.
- [plotters](https://docs.rs/plotters/latest/plotters/): For generating training metric plots.

## Data Sources

The project utilizes datasets from the following sources:

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): A dataset for image classification tasks.
- [MNIST](https://github.com/zalandoresearch/fashion-mnist): A dataset for handwritten digit recognition.

## Contributing

Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

