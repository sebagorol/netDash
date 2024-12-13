# DNS Anomaly Detection and Visualization

This project consists of two main components:

1. **Training Script**: A script to train machine learning models for DNS anomaly detection.
2. **netDash Dashboard**: A real-time web-based dashboard for monitoring DNS traffic and detecting anomalies.

---

## Prerequisites

### **System Requirements**

- Python 3.9 or later
- Recommended OS: Windows 10/11, macOS, or a Linux distribution
- Network interface access (for packet sniffing using Scapy in the `netDash` dashboard)

### **Required Tools**

- Python Package Manager (`pip`)
- Virtual Environment Manager (`venv` or `conda`)

---

## Setup for Both Components

### **Unified Virtual Environment**

To streamline the setup process, use a single virtual environment for both scripts:

1. **Create the Virtual Environment**:

   ```bash
   python -m venv netdash_train_env
   ```

2. **Activate the Virtual Environment**:

   - **Windows**:
     ```bash
     netdash_train_env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source netdash_train_env/bin/activate
     ```

3. **Install All Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 1. Training Script

### **Overview**

The training script prepares machine learning models for classifying DNS traffic as either benign or malicious. It supports multiple models, including neural networks and scikit-learn models.

### **Usage**

1. **Edit File Paths**:
   Update the file paths for datasets and model/scaler output in `trainModel.py`. Look for the following variables:
   - `benign_file_path`
   - `viriback_file_path`
   - `phishtank_file_path`
   - Output paths for models and scalers (`MODEL_OUTPUT_PATH`, `SCALER_OUTPUT_PATH`).

2. **Run the Script**:

   ```bash
   python trainModel.py
   ```

3. **Output**:
   - Models are saved as `.pkl` or `.h5` files.
   - Logs are written to `model_training.log`.
   - Performance metrics are saved to `model_performance_metrics.csv`.

---

## 2. netDash Dashboard

### **Overview**

`netDash` is a real-time dashboard for monitoring DNS traffic, classifying domains as benign or malicious, and displaying live metrics.

### **Usage**

1. **Edit Configuration**:

   - Update the `NETWORK_INTERFACE` variable in `netDash.py` with your active network interface (e.g., `"Wi-Fi"`).
   - Ensure the following paths are set correctly:
     - `MODEL_PATH`
     - `SCALER_PATH`

2. **Run the Dashboard**:

   ```bash
   python netDash.py
   ```

3. **Access the Dashboard**:

   Open your browser and navigate to:

   ```
   http://127.0.0.1:8050
   ```

4. **Features**:
   - View total DNS traffic, benign domains, and malicious domains.
   - Filter traffic by classification (all, benign, or malicious).
   - See recent classified domains with detailed logs.

---

## Logging

Both scripts produce detailed logs:

- **Training Script Logs**: `model_training.log`
- **netDash Logs**: `application.log`

Logs include:

- Initialization and setup details
- Errors and warnings
- Model performance metrics

---

## Additional Notes

1. **Virtual Environments**:
   Ensure the correct virtual environment is activated when running each script.

2. **Dependencies**:
   If a package is missing, re-install dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Network Interface**:
   Update the `NETWORK_INTERFACE` variable in `netDash.py` with your active network adapter name.

4. **Whitelisted Domains**:
   The `WHITELIST` variable in `netDash.py` contains the predefined list of domains considered benign. Update this list as needed.

5. **File Paths**:
   - For the training script, paths for input datasets and output models need to be updated.
   - For the `netDash` script, ensure that `MODEL_PATH` and `SCALER_PATH` point to the correct files.

---

## Troubleshooting

- **Missing Packages**:
  Ensure all dependencies are installed. Use `pip install -r requirements.txt` to resolve missing packages.

- **Permission Issues**:
  Run `netDash.py` with elevated privileges if Scapy fails to capture packets.

- **Model/Scaler Not Found**:
  Verify that the paths for `MODEL_PATH` and `SCALER_PATH` are correct in the scripts.

---

## Contact

For issues or questions, please contact the project maintainer or open an issue in the repository.

