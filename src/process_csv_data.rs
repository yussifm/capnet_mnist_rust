use csv::{ReaderBuilder, WriterBuilder};
use std::fs::{self, create_dir_all};
use std::io;
use std::path::Path;

/// Processes a CSV file to separate features and labels
/// and saves them in a specified directory.
pub fn process_csv(input_path: &str, output_dir: &str, label_column: &str) -> io::Result<()> {
    // Ensure the output directory exists
    create_dir_all(output_dir)?;

    // Open the input CSV file
    let mut rdr = ReaderBuilder::new()
        .has_headers(true) // Read headers if present
        .from_path(input_path)?;

    // Get headers to locate the label column
    let headers = rdr.headers()?.clone();
    let label_index = headers
        .iter()
        .position(|h| h == label_column)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Label column not found"))?;

    // Prepare paths for feature and label files
    let features_file_path = format!("{}/features.csv", output_dir);
    let labels_file_path = format!("{}/labels.csv", output_dir);

    // Create writers for features and labels
    let mut features_writer = WriterBuilder::new().from_path(&features_file_path)?;
    let mut labels_writer = WriterBuilder::new().from_path(&labels_file_path)?;

    // Write headers for features and labels
    let feature_headers: Vec<String> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != label_index) // Exclude the label column
        .map(|(_, h)| h.to_string())
        .collect();

    features_writer.write_record(&feature_headers)?;
    labels_writer.write_record(&[label_column])?;

    // Process each row
    for result in rdr.records() {
        let record = result?;

        // Collect feature values (excluding the label column)
        let features: Vec<&str> = record
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != label_index)
            .map(|(_, value)| value)
            .collect();

        // Extract the label value
        let label = record
            .get(label_index)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing label value"))?;

        // Write features and labels to their respective files
        features_writer.write_record(&features)?;
        labels_writer.write_record(&[label])?;
    }

    // Ensure all data is written to the files
    features_writer.flush()?;
    labels_writer.flush()?;

    println!(
        "Processed file '{}' and saved to:\n- {}\n- {}",
        input_path, features_file_path, labels_file_path
    );

    Ok(())
}

/// Processes all CSV files in a folder.
pub fn process_all_csvs_in_folder(input_folder: &str, output_folder: &str, label_column: &str) {
    // Ensure the output directory exists
    create_dir_all(output_folder).expect("Failed to create output directory");

    // Read all files in the input folder
    let input_path = Path::new(input_folder);
    if input_path.is_dir() {
        for entry in fs::read_dir(input_path).expect("Failed to read input folder") {
            let entry = entry.expect("Failed to read folder entry");
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("csv") {
                let file_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let output_subfolder = format!("{}/{}", output_folder, file_name);
                if let Err(e) = process_csv(
                    path.to_str().expect("Invalid file path"),
                    &output_subfolder,
                    label_column,
                ) {
                    eprintln!("Error processing file '{}': {:?}", path.display(), e);
                }
            }
        }
    } else {
        eprintln!("Input folder '{}' is not a valid directory", input_folder);
    }
}
