# DNS Anomaly Detection

This project is designed to detect DNS anomalies using machine learning models. It comprises two main components:

1. **Training Script**: To train machine learning models for classifying DNS traffic.
2. **Real-Time Dashboard (`netDash`)**: To monitor and classify DNS traffic in real time.

---

## Prerequisites

### System Requirements

- **Python Version**: Python 3.9 or later.
- **Operating System**: Windows, macOS, or Linux.
- **Network Interface**: Ensure access to an active network adapter for live packet sniffing.

### Required Tools

- Python Package Manager (`pip`).
- Git (to clone the repository).
- Virtual Environment Manager (`venv` or `conda`).

---

## Step-by-Step Instructions

### Clone the Repository

Start by cloning the project to your local machine:

```bash
git clone <repository_url>
cd <repository_folder>
```

---

## 1. **Setting up the Virtual Environment**

Create and activate a virtual environment to isolate dependencies for the project.

### Create the Virtual Environment

```bash
python -m venv dns_env
```

### Activate the Virtual Environment

- **Windows**:
  ```bash
  dns_env\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source dns_env/bin/activate
  ```

---

## 2. **Install Dependencies**

Install all required packages using the combined `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 3. **Training Script Setup**

### Overview

The training script prepares models for DNS anomaly detection.

### Configure the Script

Edit the `trainModel.py` file to specify the correct file paths. Locate and update the following variables:

- **Lines 407, 409, and 411**:
  - `benign_file_path`: Path to the benign dataset (e.g., `top-1m.csv`).
  - `viriback_file_path`: Path to the viriback malicious dataset (e.g., `blackbook.csv`).
  - `phishtank_file_path`: Path to the phishtank malicious dataset (e.g., `verified_online.csv`).

Example:
```python
benign_file_path = r"YOUR_PATH\\netDash\\top-1m.csv\\top-1m.csv"
viriback_file_path = r"YOUR_PATH\\netDash\\download\\blackbook.csv"
phishtank_file_path = r"YOUR_PATH\\netDash\\download\\verified_online.csv"
```

### Run the Training Script

Run the script to train the models:

```bash
python trainModel.py
```

### Output

- Trained models saved to specified paths.
- Performance metrics saved to `model_performance_metrics.csv`.
- Logs saved to `model_training.log`.

---

## 4. **Real-Time Dashboard (`netDash`) Setup**

### Overview

`netDash` provides a real-time view of DNS traffic, classifies domains, and logs anomalies.

### Configure the Script

Edit the `netDash.py` file to specify the correct file paths. Locate and update the following variables:

- **Line 22**:
  - `MODEL_PATH`: Path to the saved Keras model (e.g., `Neural_Network_model.h5`).
- **Line 23**:
  - `SCALER_PATH`: Path to the saved scaler (e.g., `scaler.joblib`).

Example:
```python
MODEL_PATH = r"YOUR_PATH\\netDash\\Neural_Network_model.h5"
SCALER_PATH = r"YOUR_PATH\\netDash\\scaler.joblib"
```

Ensure these paths match the output files generated by `trainModel.py`.

### Run the Dashboard

Start the dashboard by running:

```bash
python netDash.py
```

### Access the Dashboard

Open your browser and visit:

```
http://127.0.0.1:8050
```

### Features

1. **Real-Time Traffic Metrics**:
   - Total DNS traffic.
   - Number of benign and malicious domains.
2. **Filter Traffic**:
   - View all traffic, benign domains, or malicious domains.
3. **Recent Logs**:
   - Detailed classification logs with timestamps, domain names, and features.

---

## Notes and Best Practices

### Common Path Variables to Update

- **In `trainModel.py`**:
  - Lines 407, 408, 409 for dataset file paths.

- **In `netDash.py`**:
  - Line 22 for `MODEL_PATH`.
  - Line 23 for `SCALER_PATH`.

### Permissions

For `netDash`, Scapy may require administrator/root privileges for live packet sniffing.

---

## Troubleshooting

### Missing Dependencies

Reinstall dependencies using:

```bash
pip install -r requirements.txt
```

### Permission Denied

Run `netDash.py` with elevated privileges:

- **Windows**: Run your terminal as Administrator.
- **Linux/macOS**: Use `sudo` before the command.

### Incorrect Network Interface

Update the `NETWORK_INTERFACE` variable in `netDash.py` to match your system's active network adapter.

---

## Contact

For further assistance, open an issue in the repository or contact the project maintainer.
