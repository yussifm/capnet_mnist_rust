use std::fs::File; // Import for file handling.
use ndarray::{s, Array}; // Import ndarray for working with multi-dimensional arrays.
use plotters::prelude::*; // Import Plotters for creating charts.
use std::error::Error; // Import for error handling.
use tch::{ // Import Torch bindings for Rust (Tch) for tensor operations and deep learning.
    nn::{self, ModuleT, OptimizerConfig}, // Neural network module and optimizer configurations.
    Device, Kind, Tensor, // Device selection, data type handling, and tensor operations.
};

// Import custom data loader modules.
mod data_loaders;
use crate::data_loaders::{load_cifar100, load_mnist, load_other_data}; 

// Function to plot training metrics (loss and accuracy).
fn plot_metrics(losses: &[f32], accuracies: &[f32]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("training_metrics.png", (800, 600)).into_drawing_area(); 
    // Create a drawing area for the plot with specified dimensions.
    root.fill(&WHITE)?; // Fill the background with white color.

    let max_loss = *losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(); 
    // Determine the maximum value of the loss for scaling the plot.
    let max_acc = 100.0; // Assume accuracy is scaled to 100%.

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss and Accuracy", ("Arial", 20)) // Title of the chart.
        .margin(10) // Set margin around the chart.
        .set_all_label_area_size(50) // Allocate space for labels.
        .build_cartesian_2d(0..losses.len(), 0f32..max_loss.max(1.0))?; 
        // Create a 2D cartesian coordinate system with x and y ranges.

    chart.configure_mesh().draw()?; // Draw the grid and axis.

    // Plot the loss values as a red line.
    chart
        .draw_series(LineSeries::new(
            (0..).zip(losses.iter()).map(|(x, y)| (x, *y)), // Loss points (x, y).
            &RED, // Use red color.
        ))?
        .label("Loss") // Add legend label.
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED)); // Style the legend.

    // Plot the accuracy values as a blue line.
    chart
        .draw_series(LineSeries::new(
            (0..).zip(accuracies.iter().map(|a| a * max_loss / max_acc)), // Accuracy points.
            &BLUE, // Use blue color.
        ))?
        .label("Accuracy") // Add legend label.
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE)); // Style the legend.

    chart.configure_series_labels().border_style(&BLACK).draw()?; // Draw the legend box.

    Ok(()) // Return success.
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = String::new(); // Variable for user input.
    println!("Choose the dataset to use:");
    println!("1 for MNIST, 2 for CIFAR-100, 3 for Custom CSV Data"); 
    // Prompt user to select a dataset.
    std::io::stdin().read_line(&mut input)?; // Read user's dataset choice.
    let dataset_choice = input.trim().parse::<u32>().unwrap_or(1); 
    // Parse the input or default to 1.

    println!("Enter the number of training epochs: "); 
    input.clear(); // Clear the input variable for re-use.
    std::io::stdin().read_line(&mut input)?; // Read the number of epochs.
    let epochs = input.trim().parse::<usize>().unwrap_or(10); 
    // Parse the input or default to 10 epochs.

    let vs = nn::VarStore::new(Device::cuda_if_available()); 
    // Create a variable store, selecting GPU if available.

    // Load dataset and determine dimensions based on the user's choice.
    let (x_train, y_train, x_test, y_test, input_dim, output_dim) = match dataset_choice {
        1 => { // Load MNIST dataset.
            let (x_train, y_train, x_test, y_test) = load_mnist();
            (x_train, y_train, x_test, y_test, 784, 10) // Set dimensions for MNIST.
        }
        2 => { // Load CIFAR-100 dataset.
            let data_dir = "cifar-100-binary"; // Path to CIFAR-100 data.
            let (train_images, train_labels, test_images, test_labels) = load_cifar100(data_dir);
            (
                train_images.mapv(|x| x as f32 / 255.0).into_shape((50_000, 32 * 32 * 3)).unwrap(),
                Array::from_shape_vec((50_000, 1), train_labels).unwrap().mapv(|x| x as f32),
                test_images.mapv(|x| x as f32 / 255.0).into_shape((10_000, 32 * 32 * 3)).unwrap(),
                Array::from_shape_vec((10_000, 1), test_labels).unwrap().mapv(|x| x as f32),
                32 * 32 * 3, // CIFAR-100 input dimension.
                100, // CIFAR-100 output classes.
            )
        }
        3 => { // Load custom dataset.
            println!("Enter the directory path for the custom dataset:");
            input.clear();
            std::io::stdin().read_line(&mut input)?; // Read the custom dataset path.
            let data_dir = input.trim();

            let (x_data, y_data) = load_other_data(data_dir); // Load custom dataset.

            // Split into training (80%) and testing (20%) sets.
            let train_size = (x_data.shape()[0] as f32 * 0.8).round() as usize;
            let x_train = x_data.slice(s![..train_size, ..]).to_owned();
            let y_train = y_data.slice(s![..train_size, ..]).to_owned();
            let x_test = x_data.slice(s![train_size.., ..]).to_owned();
            let y_test = y_data.slice(s![train_size.., ..]).to_owned();

            (
                x_train,
                y_train,
                x_test,
                y_test,
                x_data.shape()[1], // Dynamic input dimension.
                y_data.shape()[1], // Dynamic output classes.
            )
        }
        _ => { // Handle invalid input with MNIST as default.
            eprintln!("Invalid choice! Defaulting to MNIST.");
            let (x_train, y_train, x_test, y_test) = load_mnist();
            (x_train, y_train, x_test, y_test, 784, 10)
        }
    };

    // Define the neural network structure.
    let net = nn::seq()
        .add(nn::linear(&vs.root(), input_dim as i64, 256, Default::default())) // Input to hidden.
        .add_fn(|xs| xs.relu()) // ReLU activation.
        .add(nn::linear(&vs.root(), 256, output_dim as i64, Default::default())); // Hidden to output.

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?; // Initialize Adam optimizer.

    let mut losses = Vec::new(); // Vector to store loss values.
    let mut accuracies = Vec::new(); // Vector to store accuracy values.

    for epoch in 1..=epochs {
        // Prepare training data as Tensors.
        let x_train_tensor = Tensor::f_from_slice(x_train.as_slice().unwrap())
            .unwrap()
            .to_device(vs.device())
            .reshape(&[x_train.shape()[0] as i64, input_dim as i64]);

        let y_train_tensor = Tensor::f_from_slice(y_train.as_slice().unwrap())
            .unwrap()
            .to_device(vs.device())
            .to_kind(Kind::Int64)
            .reshape(&[x_train.shape()[0] as i64]);

        let preds = net.forward_t(&x_train_tensor, true); // Forward pass.
        let loss = preds.cross_entropy_for_logits(&y_train_tensor); // Compute loss.
        opt.backward_step(&loss); // Backward pass and optimizer step.

        let predicted_labels = preds.argmax(1, false); // Get predicted labels.
        let accuracy = predicted_labels
            .eq_tensor(&y_train_tensor) // Compare predictions to labels.
            .to_kind(Kind::Float)
            .mean(Kind::Float) // Compute mean accuracy.
            .double_value(&[]);

        losses.push(loss.double_value(&[]) as f32); // Record loss.
        accuracies.push(accuracy as f32 * 100.0); // Record accuracy.

        println!(
            "Epoch: {:2} | Loss: {:.4} | Accuracy: {:.2}%",
            epoch,
            loss.double_value(&[]) as f32,
            accuracy * 100.0
        );
    }

    plot_metrics(&losses, &accuracies)?; // Plot training metrics.

    Ok(()) // Exit program successfully.
}
