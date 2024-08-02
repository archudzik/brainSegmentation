# Brain Structural Segmentation Workflow

This project processes MRI data using FreeSurfer and extracts specific regions of interest (ROIs) from the MRI data. The extracted data includes volumes and centroids of various brain structures, which are then aggregated and normalized.

## Requirements

- FreeSurfer
- Python 3.x
- Required Python packages (see requirements.txt)

## Installation

### FreeSurfer

Follow the installation instructions on the FreeSurfer website: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall.

### Python and Required Packages

Install the required Python packages:

```python
pip install -r requirements.txt
```

## Usage

To run the script, you need to provide the path to the MRI file, study ID, and age of the patient. The output will be saved in the studies_output directory and a summary CSV file will be generated.

### Example Command

```python
python workflow.py --mri "/path/to/mri/file.nii" --id "study123" --age 60
```

### Command-Line Arguments

- --mri: Path to the MRI file (.nii format).
- --id: Study ID (any string).
- --age: Age of the patient (integer).

## Output

The script generates two output files:

1. Raw results CSV: Contains volumes and centroids of various ROIs.
2. Summary CSV: Aggregated and normalized results.

## Notes

- Ensure FreeSurfer is correctly installed and the FREESURFER_HOME environment variable is set.
- The script will run FreeSurfer's recon-all command, which might take several hours to complete depending on the MRI file size and system performance.

## Tips

#### a. Lack of .nii file from the scan

If you don't have a .nii file, install FreeSurfer and then run the command:

```
dcm2niix -o "/path/to/mri/" "/path/to/mri/"
```

#### b. Operating system compatibility

You can install FreeSurfer on Linux, macOS, and Windows (using WSL: https://learn.microsoft.com/en-us/windows/wsl/about).
