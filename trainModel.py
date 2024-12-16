#Sebastian Skubisz
import numpy as np 
import pandas as pd
import logging
import time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from math import pi
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

def extract_features(domain):
    """
    Extracts various features from a domain name.
    Features:
    1. Length of the domain
    2. Digit Ratio
    3. Number of subdomains
    4. Hyphen Ratio
    5. Special Char Ratio
    6. Homoglyph Count
    7. Max Subdomain Length
    8. Avg Subdomain Length
    9. TLD length
    """
    if not domain:
        return [0] * 9  # Updated feature count after removing entropy

    try:
        length = len(domain)
        digit_count = sum(1 for char in domain if char.isdigit())
        subdomains = domain.count('.')
        hyphen_count = domain.count('-')
        special_char_count = hyphen_count + domain.count('_')  # Including underscore
        homoglyph_count = sum(1 for char in domain if char in "10OIl")

        # Subdomain lengths
        subdomain_parts = domain.split('.')[:-1]  # Exclude TLD
        subdomain_lengths = [len(sub) for sub in subdomain_parts]
        max_subdomain_length = max(subdomain_lengths) if subdomain_lengths else 0
        avg_subdomain_length = np.mean(subdomain_lengths) if subdomain_lengths else 0

        # TLD Extraction
        tld = domain.split('.')[-1] if '.' in domain else ''

        return [
            length,
            digit_count / length if length > 0 else 0,  # Digit Ratio
            subdomains,
            hyphen_count / length if length > 0 else 0,  # Hyphen Ratio
            special_char_count / length if length > 0 else 0,  # Special Char Ratio
            homoglyph_count,
            max_subdomain_length,
            avg_subdomain_length,
            len(tld)
        ]
    except Exception as e:
        logging.error(f"Error extracting features for domain '{domain}': {e}")
        return [0] * 9  # Return default values on error

def extract_features_multithreaded(domains, batch_size=1000):
    logging.info("Extracting features with multithreading...")
    features = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(0, len(domains), batch_size):
            batch = domains[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1} ({len(batch)} domains)")
            batch_features = list(executor.map(extract_features, batch))
            features.extend(batch_features)

    return np.array(features)

def load_malicious_dataset_viriback(path):
    """
    Loads the ViriBack malicious domains dataset.
    Assumes the dataset has columns: Domain, Malware, Date added, Source
    """
    try:
        viriback_data = pd.read_csv(path)
        # Ensure necessary columns are present
        expected_columns = {"Domain", "Malware", "Date added", "Source"}
        if not expected_columns.issubset(viriback_data.columns):
            logging.error(f"ViriBack dataset missing required columns: {expected_columns - set(viriback_data.columns)}")
            return pd.DataFrame()
        logging.info(f"Loaded {len(viriback_data)} malicious domains from ViriBack dataset.")
        return viriback_data
    except Exception as e:
        logging.error(f"Error loading ViriBack dataset from {path}: {e}")
        return pd.DataFrame()

def load_malicious_dataset_phishtank(path):
    """
    Loads the PhishTank malicious domains dataset.
    Assumes the dataset has columns: phish_id, url, phish_detail_url, submission_time, verified, verification_time, online, target
    """
    try:
        phishtank_data = pd.read_csv(path)
        # Ensure necessary columns are present
        expected_columns = {"phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"}
        if not expected_columns.issubset(phishtank_data.columns):
            logging.error(f"PhishTank dataset missing required columns: {expected_columns - set(phishtank_data.columns)}")
            return pd.DataFrame()
        # Extract domain from 'url'
        phishtank_data["Domain"] = phishtank_data["url"].apply(
            lambda url: urlparse(url).netloc.lower().replace("www.", "") if pd.notna(url) else None
        )
        # Optionally, extract additional features like 'target'
        phishtank_data["Target"] = phishtank_data["target"].fillna("Unknown")
        logging.info(f"Loaded {phishtank_data['Domain'].nunique()} unique malicious domains from PhishTank dataset.")
        return phishtank_data
    except Exception as e:
        logging.error(f"Error loading PhishTank dataset from {path}: {e}")
        return pd.DataFrame()

def load_benign_dataset(path):
    """
    Loads the benign domains from a CSV file.
    Assumes the benign dataset has a single column: Domain
    """
    try:
        benign_data = pd.read_csv(path, header=None, names=["Domain"])
        logging.info(f"Loaded {len(benign_data)} benign domains from {path}.")
        return benign_data["Domain"].tolist()[:1000000]  # Adjust as needed
    except Exception as e:
        logging.error(f"Error loading benign domains from {path}: {e}")
        return []

def prepare_dataset(benign_domains, viriback_data, phishtank_data):
    if not benign_domains or viriback_data.empty or phishtank_data.empty:
        logging.error("One or more datasets are empty. Exiting.")
        exit()

    benign_labels = [0] * len(benign_domains)
    
    # ViriBack Malicious Labels
    viriback_labels = [1] * len(viriback_data)
    
    # PhishTank Malicious Labels
    phishtank_labels = [1] * len(phishtank_data)
    
    # Combine all malicious data
    malicious_domains = pd.concat([viriback_data, phishtank_data], ignore_index=True)
    
    # Combine domains and labels
    domains = benign_domains + malicious_domains["Domain"].tolist()
    labels = benign_labels + viriback_labels + phishtank_labels
    
    logging.info(f"Prepared dataset with {len(domains)} domains: {len(benign_domains)} benign and {len(malicious_domains)} malicious.")
    
    return domains, labels, malicious_domains

def evaluate_model(y_test, y_pred, model_name="Model"):
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_pred),
        "Average Precision": average_precision_score(y_test, y_pred)
    }
    logging.info(f"{model_name} Performance: {metrics}")
    return metrics

def build_autoencoder(input_dim):
    """
    Builds a simple autoencoder model for anomaly detection.
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def compute_reconstruction_errors(model, data):
    """
    Computes reconstruction errors for the autoencoder.
    """
    reconstructed = model.predict(data)
    errors = np.mean((data - reconstructed) ** 2, axis=1)
    return errors

def plot_feature_importances(importances, feature_names, model_name="Random Forest"):
    """
    Plots feature importances.
    """
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index)
    plt.title(f"Feature Importances from {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.show()

def plot_confusion_matrix_custom(y_true, y_pred, model_name="Model"):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()

def plot_correlation_heatmap(features_df):
    """
    Plots a correlation heatmap of the features.
    """
    corr = features_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png")
    plt.show()

def plot_class_distribution(before_counts, after_counts):
    """
    Plots the class distribution before and after SMOTE.
    """
    labels = ['Benign', 'Malicious']
    x = np.arange(len(labels))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(x - width/2, before_counts, width, label='Before SMOTE')
    ax.bar(x + width/2, after_counts, width, label='After SMOTE')

    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution Before and After SMOTE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig("class_distribution_before_after_smote.png")
    plt.show()

def plot_smote_scatter(X_resampled, y_resampled, X_train, y_train):
    """
    Plots a scatter plot of the data after SMOTE using PCA for dimensionality reduction.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_resampled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_resampled, palette={0: 'blue', 1: 'red'}, alpha=0.5)
    plt.title('SMOTE Resampled Data (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Class', labels=['Benign', 'Malicious'])
    plt.tight_layout()
    plt.savefig("smote_scatter_plot.png")
    plt.show()

def plot_radar_chart(metrics_df, models, metrics):
    """
    Plots a radar chart comparing different models across multiple metrics.
    """
    categories = list(metrics)
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    for model in models:
        values = metrics_df[metrics_df['Model'] == model].iloc[0][metrics].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=7)
    plt.ylim(0, 1)  # Assuming all metrics are between 0 and 1

    plt.title('Model Performance Comparison', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("model_performance_radar_chart.png")
    plt.show()

def save_all_models(model_objects):
    """
    Saves all models (scikit-learn and Keras) to disk with distinct filenames.
    """
    for model_name, model in model_objects.items():
        if isinstance(model, (RandomForestClassifier, KNeighborsClassifier, XGBClassifier, IsolationForest, LocalOutlierFactor)):
            filename = f"{model_name.replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            logging.info(f"Saved scikit-learn model '{model_name}' as '{filename}'")
        elif isinstance(model, Model) or isinstance(model, Sequential):
            filename = f"{model_name.replace(' ', '_')}_model.h5"
            model.save(filename)
            logging.info(f"Saved Keras model '{model_name}' as '{filename}'")
        else:
            logging.warning(f"Model '{model_name}' is of unsupported type and was not saved.")

def save_best_model(metrics_df, models, model_objects):
    """
    Identifies the best scikit-learn model based on a chosen metric and saves it as 'dns_model.pkl'.
    Also saves the best Keras model as 'dns_model_nn.h5'.
    """
    # Define the list of scikit-learn models
    sklearn_models = ["Random Forest", "KNN", "XGBoost"]
    
    # Filter the metrics dataframe to include only scikit-learn models
    sklearn_metrics_df = metrics_df[metrics_df['Model'].isin(sklearn_models)]
    
    if sklearn_metrics_df.empty:
        logging.warning("No scikit-learn models found in metrics. No scikit-learn model was saved.")
    else:
        # Choose the metric to determine the best model. For example, F1-Score.
        best_metric = 'F1-Score'
        best_model_row = sklearn_metrics_df.loc[sklearn_metrics_df[best_metric].idxmax()]
        best_model_name = best_model_row['Model']
        logging.info(f"Best scikit-learn model based on {best_metric}: {best_model_name}")
        
        # Map model names to their objects
        model_map = {
            "Random Forest": model_objects['Random Forest'],
            "KNN": model_objects['KNN'],
            "XGBoost": model_objects['XGBoost']
        }
        
        dns_model = model_map.get(best_model_name, None)
        
        if dns_model is not None:
            # Save the best scikit-learn model to disk
            joblib.dump(dns_model, "dns_model.pkl")
            logging.info(f"Saved best scikit-learn model '{best_model_name}' as 'dns_model.pkl'")
        else:
            logging.warning(f"The best scikit-learn model '{best_model_name}' could not be found in model_objects.")
    
    # Identify and save the best Keras model based on F1-Score
    keras_models = ["Neural Network", "Autoencoder"]
    keras_metrics_df = metrics_df[metrics_df['Model'].isin(keras_models)]
    
    if keras_metrics_df.empty:
        logging.warning("No Keras models found in metrics. No Keras model was saved.")
    else:
        best_keras_model_row = keras_metrics_df.loc[keras_metrics_df['F1-Score'].idxmax()]
        best_keras_model_name = best_keras_model_row['Model']
        logging.info(f"Best Keras model based on F1-Score: {best_keras_model_name}")
        
        best_keras_model = model_objects.get(best_keras_model_name, None)
        
        if best_keras_model is not None:
            filename = f"{best_keras_model_name.replace(' ', '_')}_model.h5"
            best_keras_model.save(filename)
            logging.info(f"Saved best Keras model '{best_keras_model_name}' as '{filename}'")
        else:
            logging.warning(f"The best Keras model '{best_keras_model_name}' could not be found in model_objects.")

def main():
    # File paths
    benign_file_path = r"C:\\Users\\sskub\\OneDrive\\Desktop\\netDash\\top-1m.csv\\top-1m.csv"
    viriback_file_path = r"C:\\Users\\sskub\\OneDrive\\Desktop\\netDash\\download\\blackbook.csv"
    phishtank_file_path = r"C:\\Users\\sskub\\OneDrive\\Desktop\\netDash\\download\\verified_online.csv"  # Adjust as needed

    # Load datasets
    benign_domains = load_benign_dataset(benign_file_path)
    viriback_data = load_malicious_dataset_viriback(viriback_file_path)
    phishtank_data = load_malicious_dataset_phishtank(phishtank_file_path)

    # Prepare dataset
    domains, labels, malicious_domains = prepare_dataset(benign_domains, viriback_data, phishtank_data)

    # Feature Extraction
    start_feature_time = time.time()
    features = extract_features_multithreaded(domains)
    end_feature_time = time.time()
    logging.info(f"Feature extraction completed in {end_feature_time - start_feature_time:.2f} seconds.")

    # Convert labels to NumPy array
    labels = np.array(labels)

    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    logging.info("Feature scaling completed.")

    # Save the scaler as 'scaler.joblib' for consistency with the inference script
    joblib.dump(scaler, "scaler.joblib")
    logging.info("Saved scaler to 'scaler.joblib'")

    # Create a DataFrame for correlation heatmap
    feature_names = [
        "Length", "Digit_Ratio", "Subdomains", "Hyphen_Ratio",
        "Special_Char_Ratio", "Homoglyph_Count", "Max_Subdomain_Length",
        "Avg_Subdomain_Length", "TLD_Length"
    ]
    features_df = pd.DataFrame(scaled_features, columns=feature_names)

    # Plot Feature Correlation Heatmap
    plot_correlation_heatmap(features_df)

    # Split dataset (stratify to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logging.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Handle class imbalance using SMOTE
    logging.info("Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"After SMOTE, training set size: {X_resampled.shape[0]}, Class distribution: {np.bincount(y_resampled)}")

    # Plot Class Distribution Before and After SMOTE
    before_counts = np.bincount(y_train)
    after_counts = np.bincount(y_resampled)
    plot_class_distribution(before_counts, after_counts)

    # Plot SMOTE Scatter Plot
    plot_smote_scatter(X_resampled, y_resampled, X_train, y_train)

    # -----------------------
    # Supervised Models

    model_objects = {}

    # 1. Random Forest with Default Hyperparameters
    logging.info("Training Random Forest with default hyperparameters...")
    rf_model = RandomForestClassifier(
        n_estimators=200,           # Predefined number of trees
        max_depth=None,             # Let trees expand until all leaves are pure
        min_samples_split=2,        # Minimum number of samples required to split an internal node
        max_features='sqrt',        # Changed from 'auto' to 'sqrt' to fix the error
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    start_rf_time = time.time()
    rf_model.fit(X_resampled, y_resampled)
    end_rf_time = time.time()
    logging.info(f"Random Forest training completed in {end_rf_time - start_rf_time:.2f} seconds.")

    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, model_name="Random Forest")
    plot_confusion_matrix_custom(y_test, rf_pred, model_name="Random Forest")
    model_objects["Random Forest"] = rf_model

    # 2. Neural Network with Default Architecture and Parameters
    logging.info("Training Neural Network with default architecture and parameters...")
    input_dim = X_resampled.shape[1]

    nn_model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    start_nn_time = time.time()
    history = nn_model.fit(
        X_resampled, y_resampled,
        epochs=100,
        batch_size=128,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    end_nn_time = time.time()
    logging.info(f"Neural Network training completed in {end_nn_time - start_nn_time:.2f} seconds.")

    nn_pred_prob = nn_model.predict(X_test).flatten()

    # Threshold Tuning for Neural Network
    precision, recall, thresholds_pr = precision_recall_curve(y_test, nn_pred_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    logging.info(f"Optimal threshold for Neural Network: {optimal_threshold}")

    nn_pred = (nn_pred_prob > optimal_threshold).astype(int)
    nn_metrics = evaluate_model(y_test, nn_pred, model_name="Neural Network")
    plot_confusion_matrix_custom(y_test, nn_pred, model_name="Neural Network")
    model_objects["Neural Network"] = nn_model

    # 3. K-Nearest Neighbors (KNN) with Default Hyperparameters
    logging.info("Training K-Nearest Neighbors with default hyperparameters...")
    knn_model = KNeighborsClassifier(
        n_neighbors=5,             # Predefined number of neighbors
        weights='uniform',
        metric='euclidean',
        n_jobs=-1
    )
    start_knn_time = time.time()
    knn_model.fit(X_resampled, y_resampled)
    end_knn_time = time.time()
    logging.info(f"KNN training completed in {end_knn_time - start_knn_time:.2f} seconds.")

    knn_pred = knn_model.predict(X_test)
    knn_metrics = evaluate_model(y_test, knn_pred, model_name="KNN")
    plot_confusion_matrix_custom(y_test, knn_pred, model_name="KNN")
    model_objects["KNN"] = knn_model

    # 4. XGBoost with Default Hyperparameters
    logging.info("Training XGBoost with default hyperparameters...")
    xgb_model = XGBClassifier(
        n_estimators=200,          # Predefined number of trees
        max_depth=6,               # Typical default value
        learning_rate=0.1,         # Typical default value
        subsample=1.0,             # Use all samples
        colsample_bytree=1.0,      # Use all features
        random_state=42,
        scale_pos_weight=(y_resampled == 0).sum() / (y_resampled == 1).sum(),
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    start_xgb_time = time.time()
    xgb_model.fit(X_resampled, y_resampled)
    end_xgb_time = time.time()
    logging.info(f"XGBoost training completed in {end_xgb_time - start_xgb_time:.2f} seconds.")

    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred, model_name="XGBoost")
    plot_confusion_matrix_custom(y_test, xgb_pred, model_name="XGBoost")
    model_objects["XGBoost"] = xgb_model

    # -----------------------
    # Unsupervised Anomaly Detection Models

    # 5. Isolation Forest (Trained on Benign Domains)
    logging.info("Training Isolation Forest (Benign) with default hyperparameters...")
    # Train Isolation Forest only on benign data
    benign_train = X_resampled[y_resampled == 0]
    isolation_forest_benign = IsolationForest(
        contamination=0.5,        # Adjust based on test set distribution
        random_state=42,
        n_jobs=-1
    )
    start_iso_benign_time = time.time()
    isolation_forest_benign.fit(benign_train)
    end_iso_benign_time = time.time()
    logging.info(f"Isolation Forest (Benign) training completed in {end_iso_benign_time - start_iso_benign_time:.2f} seconds.")

    iso_benign_pred = isolation_forest_benign.predict(X_test)
    # Convert Isolation Forest outputs (-1 for anomalies, 1 for normal) to binary labels (1 for malicious, 0 for benign)
    iso_benign_pred_binary = np.where(iso_benign_pred == -1, 1, 0)
    iso_benign_metrics = evaluate_model(y_test, iso_benign_pred_binary, model_name="Isolation Forest (Benign)")
    plot_confusion_matrix_custom(y_test, iso_benign_pred_binary, model_name="Isolation Forest (Benign)")
    model_objects["Isolation Forest (Benign)"] = isolation_forest_benign

    # 6. Isolation Forest (Trained on Malicious Domains)
    logging.info("Training Isolation Forest (Malicious) with default hyperparameters...")
    # Train Isolation Forest only on malicious data
    malicious_train = X_resampled[y_resampled == 1]
    isolation_forest_malicious = IsolationForest(
        contamination=0.5,        # Adjust based on test set distribution
        random_state=42,
        n_jobs=-1
    )
    start_iso_malicious_time = time.time()
    isolation_forest_malicious.fit(malicious_train)
    end_iso_malicious_time = time.time()
    logging.info(f"Isolation Forest (Malicious) training completed in {end_iso_malicious_time - start_iso_malicious_time:.2f} seconds.")

    iso_malicious_pred = isolation_forest_malicious.predict(X_test)
    # Convert Isolation Forest outputs (-1 for anomalies, 1 for normal) to binary labels (1 for malicious, 0 for benign)
    iso_malicious_pred_binary = np.where(iso_malicious_pred == -1, 0, 1)
    iso_malicious_metrics = evaluate_model(y_test, iso_malicious_pred_binary, model_name="Isolation Forest (Malicious)")
    plot_confusion_matrix_custom(y_test, iso_malicious_pred_binary, model_name="Isolation Forest (Malicious)")
    model_objects["Isolation Forest (Malicious)"] = isolation_forest_malicious

    # 7. Local Outlier Factor (LOF) with Corrected Implementation
    logging.info("Training Local Outlier Factor (LOF) with default hyperparameters...")
    lof = LocalOutlierFactor(
        n_neighbors=20, 
        contamination=0.5,         # Adjust based on test set distribution
        novelty=True,              # Enable novelty detection
        n_jobs=-1
    )
    start_lof_time = time.time()
    lof.fit(benign_train)         # Fit on benign data
    end_lof_time = time.time()
    logging.info(f"LOF training completed in {end_lof_time - start_lof_time:.2f} seconds.")

    lof_pred = lof.predict(X_test)    # Predict on test data
    # Convert LOF outputs (-1 for anomalies, 1 for normal) to binary labels (1 for malicious, 0 for benign)
    lof_pred_binary = np.where(lof_pred == -1, 1, 0)
    lof_metrics = evaluate_model(y_test, lof_pred_binary, model_name="Local Outlier Factor")
    plot_confusion_matrix_custom(y_test, lof_pred_binary, model_name="Local Outlier Factor")
    model_objects["Local Outlier Factor"] = lof

    # 8. Autoencoder for Anomaly Detection with Default Hyperparameters
    logging.info("Training Autoencoder for anomaly detection with default hyperparameters...")
    autoencoder = build_autoencoder(input_dim=benign_train.shape[1])
    start_ae_time = time.time()
    autoencoder.fit(
        benign_train.astype(np.float32),
        benign_train.astype(np.float32),
        epochs=100,                # Adjust epochs as needed
        batch_size=256,            # Adjust batch size based on memory
        verbose=1,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    end_ae_time = time.time()
    logging.info(f"Autoencoder training completed in {end_ae_time - start_ae_time:.2f} seconds.")

    # Compute reconstruction errors on training benign data to set threshold
    train_recon_errors = compute_reconstruction_errors(autoencoder, benign_train.astype(np.float32))
    threshold = np.percentile(train_recon_errors, 95)  # 95th percentile
    logging.info(f"Autoencoder Reconstruction Error Threshold: {threshold}")

    # Compute reconstruction errors on test set
    test_recon_errors = compute_reconstruction_errors(autoencoder, X_test.astype(np.float32))
    # Label as anomaly if reconstruction error > threshold
    ae_pred = (test_recon_errors > threshold).astype(int)
    ae_metrics = evaluate_model(y_test, ae_pred, model_name="Autoencoder")
    plot_confusion_matrix_custom(y_test, ae_pred, model_name="Autoencoder")
    model_objects["Autoencoder"] = autoencoder

    # -----------------------
    # Compile Metrics
    metrics_df = pd.DataFrame([
        {"Model": "Random Forest", **rf_metrics},
        {"Model": "Neural Network", **nn_metrics},
        {"Model": "KNN", **knn_metrics},
        {"Model": "XGBoost", **xgb_metrics},
        {"Model": "Isolation Forest (Benign)", **iso_benign_metrics},
        {"Model": "Isolation Forest (Malicious)", **iso_malicious_metrics},
        {"Model": "Local Outlier Factor", **lof_metrics},
        {"Model": "Autoencoder", **ae_metrics}
    ])
    logging.info(f"Metrics Summary:\n{metrics_df}")

    # Save Metrics to CSV
    metrics_df.to_csv("model_performance_metrics.csv", index=False)
    logging.info("Saved metrics to model_performance_metrics.csv")

    # -----------------------
    # Plot Feature Importances from Random Forest
    importances = rf_model.feature_importances_
    plot_feature_importances(importances, feature_names, model_name="Random Forest")

    # Optional: Plot ROC Curve for Best Model (e.g., XGBoost)
    y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob_xgb)
    roc_auc = roc_auc_score(y_test, y_pred_prob_xgb)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'XGBoost ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("xgboost_roc_curve.png")
    plt.show()

    # Optional: Plot Precision-Recall Curve for Best Model (e.g., XGBoost)
    precision_xgb, recall_xgb, thresholds_pr_xgb = precision_recall_curve(y_test, y_pred_prob_xgb)
    average_precision_xgb = average_precision_score(y_test, y_pred_prob_xgb)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_xgb, precision_xgb, label=f'XGBoost PR curve (AP = {average_precision_xgb:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for XGBoost')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("xgboost_pr_curve.png")
    plt.show()

    # -----------------------
    # Plot Radar Chart for Model Performance
    selected_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC", "Average Precision"]
    plot_radar_chart(metrics_df, metrics_df['Model'].tolist(), selected_metrics)

    # -----------------------
    # Save All Models
    save_all_models(model_objects)

    # -----------------------
    # Determine and Save the Best Model
    save_best_model(metrics_df, metrics_df['Model'].tolist(), model_objects)

    # -----------------------
    # Display first 5 rows of resampled data (for verification)
    print("First 5 rows of X_resampled:")
    print(X_resampled[:5])
    print("First 5 labels in y_resampled:")
    print(y_resampled[:5])

if __name__ == "__main__":
    main()
