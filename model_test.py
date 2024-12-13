import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

# Paths
MODEL_PATH = r"C:\Users\sskub\OneDrive\Desktop\netDash\Neural_Network_model.h5"
SCALER_PATH = r"C:\Users\sskub\OneDrive\Desktop\netDash\scaler.joblib"

# Load model and scaler
scaler = joblib.load(SCALER_PATH)
domain_model = load_model(MODEL_PATH)

def extract_features(domain):
    length = len(domain)
    digit_count = sum(1 for char in domain if char.isdigit())
    subdomains = domain.count('.')
    hyphen_count = domain.count('-')
    special_char_count = hyphen_count + domain.count('_')
    homoglyph_count = sum(1 for char in domain if char in "01OIl")
    subdomain_parts = domain.split('.')[:-1]
    avg_subdomain_length = np.mean([len(sub) for sub in subdomain_parts]) if subdomain_parts else 0.0
    max_subdomain_length = max([len(sub) for sub in subdomain_parts]) if subdomain_parts else 0
    tld_length = len(domain.split('.')[-1]) if '.' in domain else 0
    return [
        length,
        digit_count / length if length > 0 else 0,
        subdomains,
        hyphen_count / length if length > 0 else 0,
        special_char_count / length if length > 0 else 0,
        homoglyph_count,
        avg_subdomain_length,
        max_subdomain_length,
        tld_length
    ]

def classify_domain(domain):
    features = np.array([extract_features(domain)])
    scaled_features = scaler.transform(features)
    probability = domain_model.predict(scaled_features)[0][0]
    classification = "Malicious" if probability > 0.5 else "Benign"
    return classification, probability

# Test a domain
if __name__ == "__main__":
    test_domain = "bingting.xyz.ru.iamphish-phishg123"
    classification, probability = classify_domain(test_domain)
    print(f"Domain: {test_domain}")
    print(f"Classification: {classification}")
    print(f"Probability: {probability}")
