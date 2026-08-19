# stock_price_prediction
Google Stock Price Prediction using LSTM

📌 Overview

This project uses a Long Short-Term Memory (LSTM) neural network to analyze historical Google (GOOG) stock-price data and predict future closing prices.

The workflow covers historical market-data collection, exploratory visualization, moving-average analysis, data normalization, time-series sequence creation, LSTM model training, prediction, and comparison of predicted values with actual stock prices.

Note: This project is for educational and experimentation purposes only. Stock-price predictions are inherently uncertain and should not be treated as financial advice.

🎯 Objectives

Collect historical Google stock data using yfinance.

Explore historical closing-price trends.

Calculate a 100-day moving average.

Normalize stock-price data using MinMaxScaler.

Build sequential time-series training samples using the previous 100 days.

Train a multi-layer LSTM model for stock-price prediction.

Generate predictions on the test period.

Transform predictions back toward the original price scale.

Visualize actual vs. predicted stock prices.

📊 Dataset

The project downloads GOOG historical stock data using Yahoo Finance through the yfinance Python library.

Data period

Start: 2012-01-01

End: 2022-12-21

Ticker: GOOG

Rows downloaded: 2,761

Columns used from the downloaded market data: Date, Close, High, Low, Open, Volume

The notebook separates the data into:

Training observations: 2,208

Testing observations: 553

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib

yfinance

Scikit-learn

TensorFlow / Keras

LSTM Neural Networks

🧠 Machine Learning Approach

1. Data Collection

Historical GOOG stock data is downloaded with yfinance.

import yfinance as yf

start = "2012-01-01"
end = "2022-12-21"
stock = "GOOG"

data = yf.download(stock, start, end)

2. Data Preparation

The downloaded data is reset to make the date index a regular column.

data.reset_index(inplace=True)

A 100-day rolling average is also calculated to examine the longer-term trend.

ma_100_days = data.Close.rolling(100).mean()

3. Scaling

The training data is normalized to the range 0–1 using MinMaxScaler.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

4. Sequence Creation

The model uses the previous 100 time steps to predict the next target value.

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i, 0])

This creates sequential input data suitable for an LSTM model.

🤖 LSTM Architecture

The project uses a stacked LSTM architecture with dropout regularization:

Layer

Configuration

LSTM

50 units, ReLU, return sequences

Dropout

20%

LSTM

60 units, ReLU, return sequences

Dropout

30%

LSTM

80 units, ReLU, return sequences

Dropout

40%

LSTM

120 units, ReLU

Dropout

50%

Dense

1 output unit

The final model contains 178,761 trainable parameters.

🏋️ Model Training

The model is trained for:

Epochs: 50

Batch size: 32

Training loss reached approximately 0.0017 by the final epoch in the provided notebook run.

model.fit(
    x,
    y,
    epochs=50,
    batch_size=32,
    verbose=1
)

🔮 Prediction Process

For testing, the last 100 training observations are combined with the test data so that the model has the required historical context for the first test prediction.

The trained model then generates predictions:

y_predict = model.predict(x)

The predictions and target values are subsequently transformed back toward the original price scale using the fitted scaler.

📈 Results

The project generates predictions for the test period and compares them with the corresponding actual values.

The notebook includes visualizations for:

Historical GOOG closing prices

100-day moving average

Actual vs. predicted stock prices

The model's decreasing training loss demonstrates that it learned patterns from the training sequences. However, training loss alone does not establish real-world predictive accuracy.

For a stronger evaluation, future versions should report metrics such as:

MAE — Mean Absolute Error

RMSE — Root Mean Squared Error

MAPE — Mean Absolute Percentage Error

Directional accuracy

📁 Suggested Project Structure

Google-Stock-Price-Prediction/
│
├── google_stock_prediction.ipynb
├── README.md
├── requirements.txt
└── images/
    ├── stock_trend.png
    └── actual_vs_predicted.png

⚙️ Installation

Clone the repository and install the required dependencies:

git clone <your-repository-url>
cd Google-Stock-Price-Prediction
pip install -r requirements.txt

Install dependencies manually

pip install numpy pandas matplotlib yfinance scikit-learn tensorflow

▶️ How to Run

Install Python and the required libraries.

Open the Jupyter Notebook.

Run the cells sequentially.

The notebook downloads the GOOG historical data through yfinance.

Review the exploratory analysis and moving-average plots.

Train the LSTM model.

Generate predictions.

Compare the actual and predicted values through the generated visualizations.

🔑 Key Learning Outcomes

Through this project, you can demonstrate practical understanding of:

Time-series data analysis

Financial data collection with APIs

Data preprocessing and normalization

Rolling-window calculations

Sequence generation for deep learning

LSTM architecture

Dropout regularization

Model training and prediction

Visualization with Matplotlib

End-to-end machine-learning workflow

🚀 Future Improvements

Possible improvements include:

Add MAE, RMSE and MAPE evaluation.

Use separate scalers for features and target values.

Add early stopping and model checkpointing.

Tune LSTM units, dropout rates and sequence length.

Compare LSTM with GRU and traditional forecasting models.

Add technical indicators such as RSI and MACD.

Build an interactive Streamlit dashboard.

Allow users to select different stock tickers.

Add walk-forward validation for more realistic time-series evaluation.

Deploy the prediction dashboard to Streamlit Cloud.

⚠️ Disclaimer

This project is created for learning, data-analysis, and machine-learning experimentation. Historical stock patterns do not guarantee future performance. Predictions produced by this model should not be used as a substitute for professional financial advice or as a basis for investment decisions.

👩‍💻 Author

Isha Padhal

B.Tech Computer Science Student | Python | Data Analytics | Machine Learning | Generative AI
