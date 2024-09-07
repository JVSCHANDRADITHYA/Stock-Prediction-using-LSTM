# Stock Prediction using LSTM
 The goal of this project is to build a machine learning model that can predict future stock prices based on historical data. It utilizes a Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN) that is particularly well-suited for time series forecasting.


## Directory Structure

- `data/`: Contains raw and processed data.
- `models/`: Contains saved trained models.
- `notebooks/`: Jupyter notebooks for data exploration.
- `src/`: Source code for data preprocessing, model training, and visualization.
- `test/`: Unit tests for model validation.
- `results/`: Model output, predictions, and evaluation metrics.

## Installation

1. Clone the repository.
2. Install dependencies using conda : ```conda env create -f env.yml```

## Usage

1. Place your stock price data in ```data/stock_prices.csv```
2. Run the training script: ```python train.py```
3. To use the interactive GUI created using Streamlit : ```streamlit run app.py```

## Results

Results are saved in the `results/` directory.
