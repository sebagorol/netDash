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

Edit the training script to specify the correct file paths:

- **Datasets**:
  - `benign_file_path`
  - `viriback_file_path`
  - `phishtank_file_path`
- **Output Files**:
  - Model files (e.g., `dns_model.pkl`, `Neural_Network_model.h5`).
  - Scaler (e.g., `scaler.joblib`).

### Run the Training Script

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

Edit the following variables in `netDash.py` to match your system setup:

- **Network Interface**:
  - Update `NETWORK_INTERFACE` with your active adapter name (e.g., `"Wi-Fi"`).
- **Paths**:
  - `MODEL_PATH` (Path to the trained `.h5` model).
  - `SCALER_PATH` (Path to the saved scaler).

Ensure the database (`dashboard_data.db`) exists in the working directory. If not, the script will initialize it automatically.

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

### Logging

- **Training Logs**: Located in `model_training.log`.
- **Dashboard Logs**: Located in `application.log`.

### Common Path Variables to Update

- **In `trainModel.py`**:
  - Dataset file paths (e.g., `benign_file_path`).
  - Output paths for models and scaler.

- **In `netDash.py`**:
  - `MODEL_PATH` and `SCALER_PATH`.

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

