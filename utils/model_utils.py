from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(model, X_train, y_train, model_name):
    """Trains a model and saves it."""
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    model.save(f'models/{model_name}.h5')
    return model

def evaluate_model(model, X_test, y_test, scaler, model_name):
    """Evaluates a model and saves predictions."""
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 4)), predictions), axis=1))[:, -1]
    mse = mean_squared_error(y_test, predictions)
    print(f"{model_name} Mean Squared Error: {mse}")
    np.savetxt(f'results/predictions_{model_name}.csv', predictions, delimiter=',')
    return mse
