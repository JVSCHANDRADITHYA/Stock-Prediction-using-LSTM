import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads data from a CSV file."""
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

def preprocess_data(data):
    """Preprocesses data: scales and splits into train/test sets."""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    X = []
    y = []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])  # last 60 days as features
        y.append(data_scaled[i, 3])    # Close price as target
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
