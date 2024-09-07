import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from src.utils.data_preprocessing import load_data, preprocess_data
from src.utils.visualization import plot_predictions

# Set up Streamlit app title
st.title('Stock Price Prediction App')
st.write('This app predicts future stock prices using different models.')

# Sidebar for user input
st.sidebar.header('User Input')
data_path = st.sidebar.text_input('Enter the path to the stock prices dataset', 'data/stock_prices.csv')

# Load and preprocess data
if data_path:
    data = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Load the trained models
    models = {
        'LSTM': 'models/lstm_model.h5',
        'GRU': 'models/gru_model.h5',
        'Bidirectional LSTM': 'models/bidirectional_lstm_model.h5'
    }

    # Allow user to select the model
    model_name = st.sidebar.selectbox('Select a model for prediction', list(models.keys()))
    model = load_model(models[model_name])

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 4)), predictions), axis=1))[:, -1]

    # Display the dataset
    st.subheader('Data Preview')
    st.write(data.tail())

    # Plot actual vs. predicted stock prices
    st.subheader(f'Stock Price Predictions ({model_name})')
    fig, ax = plt.subplots()
    ax.plot(y_test, color='red', label='Real Stock Price')
    ax.plot(predictions, color='blue', label='Predicted Stock Price')
    ax.set_title(f'Stock Price Prediction using {model_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

    # Show evaluation metrics
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, predictions)
    st.write(f'Mean Squared Error of the {model_name} model: {mse}')

    # Allow user to download the prediction results
    st.subheader('Download Predictions')
    st.write('You can download the predictions in CSV format.')
    predictions_df = pd.DataFrame(predictions, columns=['Predicted Price'])
    st.download_button(label='Download Predictions', data=predictions_df.to_csv(index=False), mime='text/csv')

else:
    st.write('Please provide a valid path to the dataset.')
