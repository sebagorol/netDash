import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import sqlite3
import numpy as np
from scapy.all import sniff, DNS, IP
import threading
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from waitress import serve
import logging
import os
import sys
import json
import joblib  # For loading the scaler

# ----------------------- Configuration ----------------
NETWORK_INTERFACE = "Wi-Fi"  # Replace with your actual network interface name
APPLICATION_LOG_FILE = "application.log"
MODEL_PATH = r"C:\Users\sskub\OneDrive\Desktop\netDash\Neural_Network_model.h5"  # Path to your saved Keras model
SCALER_PATH = r"C:\Users\sskub\OneDrive\Desktop\netDash\scaler.joblib"  # Path to your saved scaler
DB_PATH = "dashboard_data.db"

# WHITELIST for top 50 domains
WHITELIST = {
    "microsoft.com", "google.com", "data.microsoft.com", "events.data.microsoft.com", "windowsupdate.com",
    "ctldl.windowsupdate.com", "www.google.com", "office.com", "live.com", "apple.com",
    "microsoftonline.com", "login.microsoftonline.com", "e2ro.com", "node.e2ro.com", 
    "settings-win.data.microsoft.com", "mp.microsoft.com", "update.googleapis.com", "ecs.office.com", 
    "bing.com", "digicert.com", "amazon.com", "edge.microsoft.com", "skype.com", "amazonaws.com", 
    "clientservices.googleapis.com", "dsp.mp.microsoft.com", "office.net", "officeapps.live.com", 
    "do.dsp.mp.microsoft.com", "prod.do.dsp.mp.microsoft.com", "safebrowsing.googleapis.com", 
    "netflix.com", "mobile.events.data.microsoft.com", "ocsp.digicert.com", "self.events.data.microsoft.com", 
    "logs.netflix.com", "login.live.com", "accounts.google.com", "v10.events.data.microsoft.com", 
    "teams.microsoft.com", "edge.skype.com", "config.edge.skype.com", "facebook.com", "office365.com", 
    "doubleclick.net", "msn.com", "outlook.office365.com", "googleusercontent.com", "lencr.org", "icloud.com"
}

# ----------------------- Logging Setup ----------------
logging.basicConfig(
    filename=APPLICATION_LOG_FILE,
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("Starting the network monitoring dashboard...")

# ----------------------- Load Scaler and Model ----------------
def load_resources():
    """
    Loads the scaler and the trained Keras model.
    Returns:
        scaler: The loaded scaler object.
        model: The loaded Keras model object.
        model_type: 'keras'
    """
    # Load Scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        logging.info("Scaler loaded successfully.")
    except FileNotFoundError:
        scaler = None
        logging.error(f"Scaler file not found at {SCALER_PATH}. Ensure the scaler is saved and the path is correct.")
        sys.exit("Exiting due to missing scaler.")
    except Exception as e:
        scaler = None
        logging.error(f"Failed to load scaler: {e}")
        sys.exit("Exiting due to scaler loading failure.")

    # Load Keras Model
    try:
        if MODEL_PATH.endswith('.h5'):
            model = load_model(MODEL_PATH)
            model_type = 'keras'
            logging.info("Keras model loaded successfully.")
        else:
            model = None
            model_type = None
            logging.error("Unsupported model file format. Use .h5 for Keras models.")
            sys.exit("Exiting due to unsupported model file format.")
    except FileNotFoundError:
        model = None
        model_type = None
        logging.error(f"Model file not found at {MODEL_PATH}. Ensure the model is saved and the path is correct.")
        sys.exit("Exiting due to missing model.")
    except Exception as e:
        model = None
        model_type = None
        logging.error(f"Failed to load model: {e}")
        sys.exit("Exiting due to model loading failure.")

    return scaler, model, model_type

scaler, domain_model, model_type = load_resources()

# ----------------------- Initialize Database ----------------
def init_db():
    """
    Initializes the SQLite database with the required schema.
    """
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS network_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_ip TEXT,
            dest_ip TEXT,
            domain TEXT,
            classification TEXT,
            features TEXT,
            probability REAL
        )
        """)
        conn.commit()
        conn.close()
        logging.info("Database initialized with correct schema.")
    else:
        # Verify the schema
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(network_data);")
        columns = cursor.fetchall()
        expected_columns = [
            (0, 'id', 'INTEGER', 0, None, 1),
            (1, 'timestamp', 'TEXT', 0, None, 0),
            (2, 'source_ip', 'TEXT', 0, None, 0),
            (3, 'dest_ip', 'TEXT', 0, None, 0),
            (4, 'domain', 'TEXT', 0, None, 0),
            (5, 'classification', 'TEXT', 0, None, 0),
            (6, 'features', 'TEXT', 0, None, 0),
            (7, 'probability', 'REAL', 0, None, 0)
        ]
        if len(columns) != len(expected_columns):
            logging.error("Database schema mismatch. Expected columns not found.")
            conn.close()
            sys.exit("Exiting due to database schema mismatch.")
        for i, column in enumerate(columns):
            if column[1].lower() != expected_columns[i][1] or column[2].upper() != expected_columns[i][2]:
                logging.error(f"Column '{column[1]}' does not match expected name/type ('{expected_columns[i][1]}', '{expected_columns[i][2]}').")
                conn.close()
                sys.exit("Exiting due to database schema mismatch.")
        conn.close()
        logging.info("Database already exists with correct schema.")

init_db()

# ----------------------- Helper Functions ----------------
def extract_features(domain):
    """
    Extracts 9 features from a domain name.
    Features:
    1. Length of the domain
    2. Digit Ratio
    3. Number of subdomains
    4. Hyphen Ratio
    5. Special Char Ratio
    6. Homoglyph Count
    7. Average Subdomain Length
    8. Maximum Subdomain Length
    9. TLD Length
    """
    if not domain:
        return [0] * 9  # Return default values for empty domains

    try:
        length = len(domain)
        digit_count = sum(1 for char in domain if char.isdigit())
        subdomains = domain.count('.')
        hyphen_count = domain.count('-')
        special_char_count = hyphen_count + domain.count('_')  # Including underscore
        homoglyph_count = sum(1 for char in domain if char in "01OIl")

        # Subdomain lengths
        subdomain_parts = domain.split('.')[:-1]  # Exclude TLD
        if subdomain_parts:
            avg_subdomain_length = np.mean([len(sub) for sub in subdomain_parts])
            max_subdomain_length = max([len(sub) for sub in subdomain_parts])
        else:
            avg_subdomain_length = 0.0
            max_subdomain_length = 0

        # TLD Length
        tld_length = len(domain.split('.')[-1]) if '.' in domain else 0

        return [
            length,
            digit_count / length if length > 0 else 0,         # Digit Ratio
            subdomains,
            hyphen_count / length if length > 0 else 0,       # Hyphen Ratio
            special_char_count / length if length > 0 else 0, # Special Char Ratio
            homoglyph_count,
            round(avg_subdomain_length, 2),
            max_subdomain_length,
            tld_length
        ]
    except Exception as e:
        logging.error(f"Error extracting features for domain '{domain}': {e}")
        return [0] * 9  # Return default values on error

def classify_domain(domain):
    """
    Classifies the domain using the pre-trained Keras model.
    Returns 'Malicious', 'Benign', or 'Unknown' along with feature values and probability.
    """
    # Check if the domain is in the whitelist
    if domain in WHITELIST:
        logging.info(f"Domain '{domain}' is in the whitelist. Automatically classified as 'Benign'.")
        return "Benign", "{}", 1.0  # Return default features and a high probability for benign

    if not domain_model:
        return "Unknown", "{}", 0.0  # Return empty JSON string

    try:
        features = extract_features(domain)
        features_array = np.array([features], dtype=float)
        if scaler:
            features_array = scaler.transform(features_array)
            logging.debug(f"Scaled features: {features_array}")
        else:
            logging.warning("No scaler loaded. Features are not scaled.")

        if model_type == 'keras':
            probability = domain_model.predict(features_array)[0][0]
            probability = float(probability)
            probability = round(probability, 4)
            classification = "Malicious" if probability > 0.5 else "Benign"
        else:
            classification = "Unknown"
            probability = 0.0

        feature_dict = {
            "length": features[0],
            "digit_ratio": features[1],
            "subdomains": features[2],
            "hyphen_ratio": features[3],
            "special_char_ratio": features[4],
            "homoglyph_count": features[5],
            "avg_subdomain_length": features[6],
            "max_subdomain_length": features[7],
            "tld_length": features[8]
        }

        # Serialize feature_dict to JSON string
        features_str = json.dumps(feature_dict)
        return classification, features_str, probability
    except Exception as e:
        logging.error(f"Classification error for domain '{domain}': {e}")
        return "Unknown", "{}", 0.0  # Return empty JSON string on error

def capture_network_data(packet):
    """
    Captures network packets and processes DNS queries.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        if packet.haslayer(IP):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            source_ip = packet[IP].src
            dest_ip = packet[IP].dst
            domain = None
            probability = 0.0
            classification = "Unknown"

            if packet.haslayer(DNS) and packet[DNS].qd:
                try:
                    # Decode domain name with error handling
                    domain_bytes = packet[DNS].qd.qname
                    if isinstance(domain_bytes, bytes):
                        domain = domain_bytes.decode('utf-8', errors='ignore').strip('.')
                    else:
                        domain = str(domain_bytes).strip('.')
                    
                    classification, features_str, probability = classify_domain(domain)

                    # Ensure features_str is a string
                    if not isinstance(features_str, str):
                        features_str = str(features_str)

                    # Insert into database
                    cursor.execute("""
                    INSERT INTO network_data (timestamp, source_ip, dest_ip, domain, classification, features, probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (timestamp, source_ip, dest_ip, domain, classification, features_str, probability))
                    conn.commit()
                    logging.info(f"Inserted domain: {domain} | Classification: {classification} | Features: {features_str} | Probability: {probability}")
                except Exception as e:
                    logging.error(f"Error processing DNS query: {e}")
    except Exception as e:
        logging.error(f"Error capturing packet: {e}")
    finally:
        conn.close()

def start_network_capture():
    """
    Starts the network packet capture in a separate thread.
    """
    try:
        logging.info(f"Starting packet capture on interface: {NETWORK_INTERFACE}")
        sniff(prn=capture_network_data, iface=NETWORK_INTERFACE, filter="udp port 53", store=0)
    except Exception as e:
        logging.error(f"Failed to start network capture: {e}")

# ----------------------- Dashboard Setup ----------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Live DNS Anomaly Detection Dashboard"),
    
    # Counters Section
    html.Div([
        html.Div([
            html.H3("Total Traffic"),
            html.H4(id="total-traffic")
        ], style={"width": "30%", "textAlign": "center", "padding": "10px", "border": "1px solid #ccc", "border-radius": "5px"}),
        
        html.Div([
            html.H3("Benign Domains"),
            html.H4(id="total-benign")
        ], style={"width": "30%", "textAlign": "center", "padding": "10px", "border": "1px solid #4CAF50", "border-radius": "5px"}),
        
        html.Div([
            html.H3("Malicious Domains"),
            html.H4(id="total-malicious")
        ], style={"width": "30%", "textAlign": "center", "padding": "10px", "border": "1px solid #f44336", "border-radius": "5px"}),
    ], style={"display": "flex", "justify-content": "space-around", "margin-top": "20px"}),
    
    # Filter Dropdown
    html.Div([
        html.Label("Filter Domains:"),
        dcc.Dropdown(
            id='filter-dropdown',
            options=[
                {'label': 'All Traffic', 'value': 'All'},
                {'label': 'Benign Domains', 'value': 'Benign'},
                {'label': 'Malicious Domains', 'value': 'Malicious'}
            ],
            value='All',
            clearable=False,
            style={"width": "200px"}
        )
    ], style={"textAlign": "center", "margin-top": "30px"}),
    
    # Recent Classified Domains Section
    html.Div([
        html.H3("Recent Classified Domains"),
        dcc.Textarea(
            id="flagged-traffic",
            style={"width": "100%", "height": "300px", "padding": "10px", "border": "1px solid #ccc", "border-radius": "5px"},
            value="",
            readOnly=True
        )
    ], style={"margin-top": "30px"}),
    
    # Update Interval
    dcc.Interval(
        id="update-interval",
        interval=5*1000,  # 5 seconds
        n_intervals=0
    ),
], style={"width": "90%", "margin": "0 auto"})

@app.callback(
    [
        Output("total-traffic", "children"),
        Output("total-benign", "children"),
        Output("total-malicious", "children"),
        Output("flagged-traffic", "value")
    ],
    [
        Input("update-interval", "n_intervals"),
        Input("filter-dropdown", "value")
    ]
)
def update_dashboard(n, filter_value):
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Fetch Total Traffic
        total_traffic = pd.read_sql_query("SELECT COUNT(*) AS count FROM network_data", conn)["count"][0]
        
        # Fetch Total Benign Domains
        total_benign = pd.read_sql_query("SELECT COUNT(*) AS count FROM network_data WHERE classification='Benign'", conn)["count"][0]
        
        # Fetch Total Malicious Domains
        total_malicious = pd.read_sql_query("SELECT COUNT(*) AS count FROM network_data WHERE classification='Malicious'", conn)["count"][0]
        
        # Fetch Recent Classified Domains based on filter
        if filter_value == 'All':
            query = """
                SELECT timestamp, domain, classification, features, probability 
                FROM network_data 
                ORDER BY id DESC 
                LIMIT 10
            """
        else:
            query = f"""
                SELECT timestamp, domain, classification, features, probability 
                FROM network_data 
                WHERE classification = '{filter_value}'
                ORDER BY id DESC 
                LIMIT 10
            """
        
        flagged_df = pd.read_sql_query(query, conn)
        conn.close()

        logging.debug(f"Total Traffic: {total_traffic}")
        logging.debug(f"Total Benign: {total_benign}")
        logging.debug(f"Total Malicious: {total_malicious}")
        logging.debug(f"Flagged Data:\n{flagged_df}")

        if flagged_df.empty:
            flagged_domains = "No data detected."
        else:
            flagged_domains = ""
            for _, row in flagged_df.iterrows():
                # Ensure all fields are properly formatted
                timestamp = str(row['timestamp']) if row['timestamp'] else "N/A"
                domain = str(row['domain']) if row['domain'] else "N/A"
                classification = str(row['classification']) if row['classification'] else "N/A"
                features_json = row['features']
                if isinstance(features_json, bytes):
                    features_json = features_json.decode('utf-8', errors='ignore')

                try:
                    features_dict = json.loads(features_json)
                except json.JSONDecodeError:
                    features_dict = "Malformed features data"

                probability_value = row['probability']
                if isinstance(probability_value, bytes):
                    probability_value = probability_value.decode('utf-8', errors='ignore')
                try:
                    confidence = float(probability_value)
                except (ValueError, TypeError):
                    confidence = 0.0  # Assign a default value or handle as needed

                flagged_domains += f"{timestamp} - {domain} | Classification: {classification} | Features: {features_dict} | Confidence: {confidence:.2f}\n"

        return total_traffic, total_benign, total_malicious, flagged_domains.strip() or "No data detected."
    except Exception as e:
        logging.error(f"Error updating dashboard: {e}")
        return "Error", "Error", "Error", "Error fetching data."

# ----------------------- Start Packet Capture ----------------
network_thread = threading.Thread(target=start_network_capture, daemon=True)
network_thread.start()

# ----------------------- Run the Dashboard ----------------
if __name__ == "__main__":
    try:
        logging.info("Starting Dash server...")
        serve(app.server, host="127.0.0.1", port=8050)
    except Exception as e:
        logging.error(f"Failed to start the server: {e}")
        sys.exit("Exiting due to server startup failure.")
