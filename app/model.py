import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "./data/Telco-Customer-Churn.csv"
MODEL_PATH = "./models/customer_churn_model.pkl"
ENCODER_PATH = "./models/encoders.pkl"

def train_model():
    """Train the model and save it"""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])  # Remove unnecessary column

    # Handle missing values
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

    # Label encoding for target variable
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    # Encode categorical features
    object_columns = df.select_dtypes(include="object").columns
    encoders = {}

    for column in object_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder

    # Save encoders
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)

    # Save model
    model_data = {"model": model, "feature_names": X.columns.to_list()}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print("Model trained and saved successfully!")

def load_model():
    """Load the trained model"""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
