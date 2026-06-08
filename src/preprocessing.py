import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE, RandomOverSampler

def load_and_clean_data(file1, file2):
    """
    Loads data1 and data2, concatenates them, and handles missing values.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Handle missing values
    df["cmd_arguments"] = df["cmd_arguments"].fillna("none")
    df["username"] = df["username"].fillna("unknown")
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp_str"], errors="coerce")
    
    return df

def compute_entropy(x):
    """
    Computes Shannon entropy of command name frequencies.
    """
    counts = pd.Series(x).value_counts()
    probs = counts / len(x)
    return -np.sum(probs * np.log2(probs + 1e-12))

def extract_session_features(df):
    """
    Performs session-level aggregation and extracts statistical features.
    """
    # Sort for time calculations
    df = df.sort_values(by=["sandbox_id", "timestamp"])
    
    # Session duration (in seconds)
    duration_df = df.groupby("sandbox_id")["timestamp"].agg(lambda x: (x.max() - x.min()).total_seconds()).reset_index()
    duration_df.columns = ["sandbox_id", "session_duration"]
    
    # Aggregated features
    agg_df = df.groupby("sandbox_id").agg({
        "cmd_name": ["count", "nunique", compute_entropy],
        "cmd_type": "nunique",
        "ip": "nunique",
        "hostname": "nunique",
        "username": "nunique",
        "level": ["mean", lambda x: np.sum(x >= 45)]
    }).reset_index()
    
    # Flatten column multi-index
    agg_df.columns = [
        "sandbox_id",
        "total_commands",
        "unique_command_names",
        "command_entropy",
        "unique_cmd_types",
        "unique_ips",
        "unique_hosts",
        "unique_users",
        "avg_level",
        "privileged_commands_count"
    ]
    
    # Merge duration
    session_df = pd.merge(agg_df, duration_df, on="sandbox_id")
    
    # Derivate features
    session_df["commands_per_minute"] = session_df["total_commands"] / (session_df["session_duration"] / 60.0 + 1e-5)
    
    session_df["average_time_between_commands"] = np.where(
        session_df["total_commands"] > 1,
        session_df["session_duration"] / (session_df["total_commands"] - 1),
        0.0
    )
    
    session_df["ratio_of_privileged_commands"] = session_df["privileged_commands_count"] / session_df["total_commands"]
    session_df["distinct_user_ratio"] = session_df["unique_users"] / session_df["total_commands"]
    
    # Drop intermediate columns
    session_df = session_df.drop(columns=["privileged_commands_count"])
    
    return session_df

def extract_nlp_features(df, max_features=20):
    """
    Extracts NLP features using TF-IDF on combined command name and arguments.
    """
    df["cmd_text"] = df["cmd_name"].astype(str) + " " + df["cmd_arguments"].astype(str)
    
    # Group by session and join command strings
    session_text = df.groupby("sandbox_id")["cmd_text"].apply(lambda x: " ".join(x)).reset_index()
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(session_text["cmd_text"]).toarray()
    
    tfidf_cols = [f"tfidf_{i}" for i in range(max_features)]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)
    tfidf_df["sandbox_id"] = session_text["sandbox_id"]
    
    return tfidf_df

def generate_labels(features_df, method="isolation_forest", contamination=0.10):
    """
    Generates labels for sessions using different strategies:
    - 'percentile_95': Original method (top 5% by total_commands)
    - 'isolation_forest': Isolation Forest on session features
    - 'local_outlier_factor': LOF on session features
    """
    # Drop non-feature column
    X = features_df.drop(columns=["sandbox_id"])
    
    # Scale features for distance/density based models (LOF/IForest)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == "percentile_95":
        threshold = features_df["total_commands"].quantile(0.95)
        labels = (features_df["total_commands"] > threshold).astype(int).values
    elif method == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(X_scaled)
        # map -1 (anomaly) to 1, and 1 (normal) to 0
        labels = np.where(preds == -1, 1, 0)
    elif method == "local_outlier_factor":
        # Using fit_predict on the whole dataset
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        preds = model.fit_predict(X_scaled)
        labels = np.where(preds == -1, 1, 0)
    else:
        raise ValueError(f"Unknown labeling method: {method}")
        
    return labels

def prepare_mlp_data(session_features, tfidf_features, labels, balance_method=None):
    """
    Merges session and NLP features, scales them, and applies class balancing (SMOTE/ROS).
    """
    # Merge tabular and TF-IDF features
    full_df = pd.merge(session_features, tfidf_features, on="sandbox_id")
    
    X = full_df.drop(columns=["sandbox_id"])
    y = labels
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified split first
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply balancing only on training set
    if balance_method == "smote":
        # Use smaller k_neighbors if minority class has very few samples
        min_class_size = min(pd.Series(y_train).value_counts())
        k_neigh = min(3, max(1, min_class_size - 1))
        if min_class_size > 1:
            sm = SMOTE(k_neighbors=k_neigh, random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
    elif balance_method == "ros":
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
    return X_train, X_test, y_train, y_test

def prepare_sequence_data(df, labels, max_len=50):
    """
    Prepares command sequences for LSTM and GRU models.
    """
    df = df.sort_values(by=["sandbox_id", "timestamp"])
    
    le = LabelEncoder()
    df["cmd_encoded"] = le.fit_transform(df["cmd_name"])
    
    # Get vocabulary size
    vocab_size = df["cmd_encoded"].nunique() + 1 # +1 for padding index 0
    
    # Increment encoding by 1 to reserve 0 for padding
    df["cmd_encoded"] = df["cmd_encoded"] + 1
    
    sequences_series = df.groupby("sandbox_id")["cmd_encoded"].apply(list)
    
    # Padding
    from keras.utils import pad_sequences
    X_seq = pad_sequences(sequences_series, maxlen=max_len, padding="post", value=0)
    
    # Ensure labels match the sequences index order
    # sequences_series.index matches sandbox_ids
    labels_series = pd.DataFrame({"sandbox_id": sequences_series.index})
    
    return X_seq, vocab_size
