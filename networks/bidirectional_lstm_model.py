from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional

def create_bidirectional_lstm_model(input_shape):
    """Creates and returns a Bidirectional LSTM model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
