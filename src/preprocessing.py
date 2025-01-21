import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, test_size=0.2, random_state=42):
    """Preprocess the data for training"""
    # Remove duplicates and handle missing values
    data = data.drop_duplicates()
    data = data.dropna()
    
    # Scale features
    scaler = StandardScaler()
    features = data.drop('Class', axis=1)
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, 
        data['Class'],
        test_size=test_size,
        random_state=random_state,
        stratify=data['Class']
    )
    
    return X_train, X_test, y_train, y_test