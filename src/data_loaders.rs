use std::fs::File; // File handling for reading files.
use std::io::{BufReader, Read}; // Input-output utilities for buffered reading and file operations.
use mnist::{Mnist, MnistBuilder}; // MNIST crate for downloading and processing the MNIST dataset.
use ndarray::{s, Array, Array2, Array3, Axis, stack}; // ndarray crate for multi-dimensional arrays and operations.
use csv::Reader; // CSV crate for reading data from CSV files.
use image::imageops::{resize, FilterType}; // Image crate for resizing images and specifying filters.

// Function to load and normalize the MNIST dataset
pub fn load_mnist() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    println!("Attempting to download and extract MNIST dataset...");
    
    // Use the MnistBuilder to download and extract MNIST data.
    let Mnist {
        trn_img, // Training images.
        trn_lbl, // Training labels.
        tst_img, // Testing images.
        tst_lbl, // Testing labels.
        ..
    } = MnistBuilder::new()
        .label_format_digit() // Configure to return labels as digits.
        .download_and_extract() // Automatically download and extract MNIST data.
        .training_set_length(50_000) // Set training dataset size.
        .validation_set_length(10_000) // Set validation dataset size.
        .test_set_length(10_000) // Set testing dataset size.
        .finalize(); // Finalize the builder and retrieve the dataset.

    println!("Dataset successfully loaded!");

    // Normalize training images to the range [0, 1].
    let x_train = Array::from_shape_vec((50_000, 28 * 28), trn_img)
        .unwrap()
        .mapv(|x| x as f32 / 255.0);

    // Normalize testing images to the range [0, 1].
    let x_test = Array::from_shape_vec((10_000, 28 * 28), tst_img)
        .unwrap()
        .mapv(|x| x as f32 / 255.0);

    // Convert training labels to a 2D array.
    let y_train = Array::from_shape_vec((50_000, 1), trn_lbl)
        .unwrap()
        .mapv(|x| x as f32);

    // Convert testing labels to a 2D array.
    let y_test = Array::from_shape_vec((10_000, 1), tst_lbl)
        .unwrap()
        .mapv(|x| x as f32);

    // Return normalized images and labels.
    (x_train, y_train, x_test, y_test)
}

// Function to load CIFAR-100 dataset
pub fn load_cifar100(data_dir: &str) -> (Array3<u8>, Vec<u8>, Array3<u8>, Vec<u8>) {
    // Helper function to read binary CIFAR-100 files.
    fn read_bin(file: &str) -> (Array3<u8>, Vec<u8>) {
        let mut file = File::open(file).expect("Failed to open CIFAR binary file"); // Open the binary file.
        let mut buf = vec![]; // Buffer to hold file data.
        file.read_to_end(&mut buf).expect("Failed to read CIFAR file"); // Read the file content into buffer.

        let num_images = buf.len() / 3074; // 1 label + 3072 pixels (32x32 RGB).
        let mut labels = Vec::new(); // Vector to store labels.
        let mut images = Array3::<u8>::zeros((num_images, 3, 32 * 32)); // Array to store images.

        for (i, chunk) in buf.chunks_exact(3074).enumerate() {
            labels.push(chunk[1]); // Extract fine label (CIFAR-100 uses fine and coarse labels).
            images.slice_mut(s![i, .., ..]).assign(
                &Array::from_shape_vec((3, 32 * 32), chunk[2..].to_vec())
                    .unwrap()
                    .mapv(|x| x), // Convert pixel data into the array.
            );
        }
        (images, labels) // Return images and labels.
    }

    // Paths to CIFAR-100 training and testing files.
    let train_file = format!("{}/train.bin", data_dir);
    let test_file = format!("{}/test.bin", data_dir);

    // Load training and testing data.
    let (train_images, train_labels) = read_bin(&train_file);
    let (test_images, test_labels) = read_bin(&test_file);

    // Return training and testing images and labels.
    (train_images, train_labels, test_images, test_labels)
}

// Function to load a generic dataset from CSV files.
pub fn load_other_data(data_dir: &str) -> (Array2<f32>, Array2<f32>) {
    let features_path = format!("{}/features.csv", data_dir); // Path to features CSV.
    let labels_path = format!("{}/labels.csv", data_dir); // Path to labels CSV.

    // Load features from the CSV file.
    let features = read_csv_to_array2(&features_path);
    // Load labels from the CSV file.
    let labels = read_csv_to_array2(&labels_path);

    println!("Successfully loaded other data from: {}", data_dir);

    // Return features and labels arrays.
    (features, labels)
}

// Helper function to read a CSV file and convert it to Array2<f32>.
fn read_csv_to_array2(file_path: &str) -> Array2<f32> {
    let mut rdr = Reader::from_path(file_path).expect("Failed to open CSV file"); // Open the CSV file.
    let mut data = vec![]; // Vector to hold data rows.

    // Read each record in the CSV file.
    for result in rdr.records() {
        let record = result.expect("Failed to read record"); // Read a row.
        let row: Vec<f32> = record
            .iter()
            .map(|x| x.parse::<f32>().expect("Failed to parse float")) // Convert each value to f32.
            .collect(); // Collect row values into a vector.
        data.extend(row); // Append the row to data.
    }

    let num_rows = rdr.records().count(); // Count the number of rows.
    let num_cols = data.len() / num_rows; // Calculate the number of columns.
    Array2::from_shape_vec((num_rows, num_cols), data).expect("Failed to convert CSV data to Array2")
    // Convert the data vector into a 2D array.
}
