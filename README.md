Here's a suggested README description for your GitHub repository:

---

# CapNet MNIST Rust Project

This repository provides a Rust-based implementation of a neural network model for classifying images from the MNIST dataset. The project leverages libraries like `tch-rs` (for Tensor operations using PyTorch bindings in Rust), `mnist` (for handling the MNIST dataset), and `plotters` (for visualizing training metrics such as loss and accuracy). The repository demonstrates key components of data processing, model training, and visualization, and it aims to offer a practical example of Rust's capabilities in deep learning and data handling.

## Features

- **Automatic Download and Extraction**: The code automatically downloads and extracts the MNIST dataset if not present.
- **Dataset Normalization**: Images are normalized to ensure pixel values range between 0 and 1.
- **Neural Network Implementation**: The network consists of simple fully connected layers for classifying handwritten digits from 0 to 9.
- **Training Metrics Visualization**: Uses the `plotters` library to generate plots for loss and accuracy during training.
- **CUDA Support**: If available, the model utilizes CUDA for accelerated computation.

## Installation

Ensure you have Rust installed. You can install Rust using [rustup](https://rustup.rs/).

Clone the repository:

```sh
git clone https://github.com/yussifm/capnet_mnist_rust.git
cd capnet_mnist_rust
```

Build and run the project:

```sh
cargo run --release
```

## Usage

- **Data Handling**: The code processes and normalizes the MNIST data into a format suitable for input to the neural network.
- **Model Training**: The model is trained over a number of epochs, optimizing using the Adam optimizer.
- **Visualization**: Training metrics (loss and accuracy) are plotted and saved as an image using `plotters`.

## Dependencies

- [tch-rs](https://github.com/LaurentMazare/tch-rs)
- [ndarray](https://docs.rs/ndarray/latest/ndarray/)
- [mnist](https://docs.rs/mnist/latest/mnist/)
- [plotters](https://docs.rs/plotters/latest/plotters/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust or expand on any section to match your specific needs and goals!"# capnet_mnist_rust" 
