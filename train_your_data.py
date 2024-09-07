import os
from utils.data_preprocessing import load_data, preprocess_data
from networks.lstm_model import create_lstm_model
from utils.visualization import plot_predictions
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and preprocess data
data_path = 'data/stock_prices.csv'
data = load_data(data_path)
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Initialize model
model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the model
model.save('models/lstm_model.h5')

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 4)), predictions), axis=1))[:, -1]

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Model Mean Squared Error: {mse}")

# Plot the predictions
plot_predictions(y_test, predictions)

# Save results
np.savetxt('results/predictions.csv', predictions, delimiter=',')
with open('results/evaluation.txt', 'w') as f:
    f.write(f"Model Mean Squared Error: {mse}\n")
