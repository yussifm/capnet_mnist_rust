use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array, Array2, Array3};
use plotters::prelude::*;
use std::error::Error;
use tch::{
    nn::{self, ModuleT, OptimizerConfig},
    Device, Kind, Tensor,
};

// Function to load and normalize the MNIST dataset
fn load_mnist() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    println!("Attempting to download and extract MNIST dataset...");
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .download_and_extract()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    println!("Dataset successfully loaded!");

    let x_train = Array::from_shape_vec((50_000, 28 * 28), trn_img)
        .unwrap()
        .mapv(|x| x as f32 / 255.0);
    let x_test = Array::from_shape_vec((10_000, 28 * 28), tst_img)
        .unwrap()
        .mapv(|x| x as f32 / 255.0);
    let y_train = Array::from_shape_vec((50_000, 1), trn_lbl)
        .unwrap()
        .mapv(|x| x as f32);
    let y_test = Array::from_shape_vec((10_000, 1), tst_lbl)
        .unwrap()
        .mapv(|x| x as f32);

    (x_train, y_train, x_test, y_test)
}

// Function to load CIFAR-100 dataset
fn load_cifar100(data_dir: &str) -> (Array3<u8>, Vec<u8>, Array3<u8>, Vec<u8>) {
    fn read_bin(file: &str) -> (Array3<u8>, Vec<u8>) {
        let mut file = File::open(file).expect("Failed to open CIFAR binary file");
        let mut buf = vec![];
        file.read_to_end(&mut buf).expect("Failed to read CIFAR file");

        let num_images = buf.len() / 3074; // 1 label + 3072 pixels (32x32 RGB)
        let mut labels = Vec::new();
        let mut images = Array3::<u8>::zeros((num_images, 3, 32 * 32));

        for (i, chunk) in buf.chunks_exact(3074).enumerate() {
            labels.push(chunk[1]); // Fine label (CIFAR-100)
            images.slice_mut(s![i, .., ..]).assign(
                &Array::from_shape_vec((3, 32 * 32), chunk[2..].to_vec())
                    .unwrap()
                    .mapv(|x| x),
            );
        }
        (images, labels)
    }

    let train_file = format!("{}/train.bin", data_dir);
    let test_file = format!("{}/test.bin", data_dir);

    // Load training and testing data
    let (train_images, train_labels) = read_bin(&train_file);
    let (test_images, test_labels) = read_bin(&test_file);

    (train_images, train_labels, test_images, test_labels)
}

// Function to plot training metrics (loss and accuracy)
fn plot_metrics(losses: &[f32], accuracies: &[f32]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("training_metrics.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = *losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_acc = 100.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss and Accuracy", ("Arial", 20))
        .margin(10)
        .set_all_label_area_size(50)
        .build_cartesian_2d(0..losses.len(), 0f32..max_loss.max(1.0))?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..).zip(losses.iter()).map(|(x, y)| (x, *y)),
            &RED,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            (0..).zip(accuracies.iter().map(|a| a * max_loss / max_acc)),
            &BLUE,
        ))?
        .label("Accuracy")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    println!("Choose the dataset to use (1 for MNIST, 2 for CIFAR-100): ");
    std::io::stdin().read_line(&mut input)?;
    let dataset_choice = input.trim().parse::<u32>().unwrap_or(1);

    println!("Enter the number of training epochs: ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let epochs = input.trim().parse::<usize>().unwrap_or(10);

    let vs = nn::VarStore::new(Device::cuda_if_available());

    let input_dim = if dataset_choice == 1 { 784 } else { 32 * 32 * 3 };
    let output_dim = if dataset_choice == 1 { 10 } else { 100 };

    let net = nn::seq()
        .add(nn::linear(&vs.root(), input_dim, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root(), 256, output_dim, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    let (x_train, y_train, x_test, y_test) = if dataset_choice == 1 {
        load_mnist()
    } else {
        let data_dir = "cifar-100-binary"; // Update with your CIFAR-100 data directory
        let (train_images, train_labels, test_images, test_labels) = load_cifar100(data_dir);
        (
            train_images.mapv(|x| x as f32 / 255.0).into_shape((50_000, 32 * 32 * 3)).unwrap(),
            Array::from_shape_vec((50_000, 1), train_labels).unwrap().mapv(|x| x as f32),
            test_images.mapv(|x| x as f32 / 255.0).into_shape((10_000, 32 * 32 * 3)).unwrap(),
            Array::from_shape_vec((10_000, 1), test_labels).unwrap().mapv(|x| x as f32),
        )
    };

    let mut losses = Vec::new();
    let mut accuracies = Vec::new();

    for epoch in 1..=epochs {
        let x_train_tensor = Tensor::of_slice(x_train.as_slice().unwrap())
            .to_device(vs.device())
            .reshape(&[50_000, input_dim as i64]);
        let y_train_tensor = Tensor::of_slice(y_train.as_slice().unwrap())
            .to_device(vs.device())
            .to_kind(Kind::Int64)
            .reshape(&[50_000]);

        let preds = net.forward_t(&x_train_tensor, true);
        let loss = preds.cross_entropy_for_logits(&y_train_tensor);
        opt.backward_step(&loss);

        let predicted_labels = preds.argmax(1, false);
        let accuracy = predicted_labels
            .eq_tensor(&y_train_tensor)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);

        losses.push(f32::from(loss.shallow_clone()));
        accuracies.push(accuracy as f32 * 100.0);

        println!(
            "Epoch: {:2} | Loss: {:.4} | Accuracy: {:.2}%",
            epoch,
            f32::from(loss),
            accuracy * 100.0
        );
    }

    plot_metrics(&losses, &accuracies)?;

    Ok(())
}
