from src.utils.data_preprocessing import load_data, preprocess_data
from src.networks.lstm_model import create_lstm_model
from src.networks.gru_model import create_gru_model
from src.networks.bidirectional_lstm_model import create_bidirectional_lstm_model
from src.utils.model_utils import train_model, evaluate_model

# Load and preprocess data
data_path = 'data/stock_prices.csv'
data = load_data(data_path)
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Initialize and train models
lstm_model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
gru_model = create_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
bidirectional_lstm_model = create_bidirectional_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Train and evaluate LSTM model
train_model(lstm_model, X_train, y_train, 'lstm_model')
lstm_mse = evaluate_model(lstm_model, X_test, y_test, scaler, 'lstm')

# Train and evaluate GRU model
train_model(gru_model, X_train, y_train, 'gru_model')
gru_mse = evaluate_model(gru_model, X_test, y_test, scaler, 'gru')

# Train and evaluate Bidirectional LSTM model
train_model(bidirectional_lstm_model, X_train, y_train, 'bidirectional_lstm_model')
bidirectional_lstm_mse = evaluate_model(bidirectional_lstm_model, X_test, y_test, scaler, 'bidirectional_lstm')

# Save evaluations
with open('results/evaluation.txt', 'w') as f:
    f.write(f"LSTM Mean Squared Error: {lstm_mse}\n")
    f.write(f"GRU Mean Squared Error: {gru_mse}\n")
    f.write(f"Bidirectional LSTM Mean Squared Error: {bidirectional_lstm_mse}\n")
