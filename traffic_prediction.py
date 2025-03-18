import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server rendering
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def generate_traffic_forecast(csv_file='data/traffic.csv', periods=7):
    """
    Function to read traffic data, train Prophet model, and generate a forecast plot.
    Returns the path to the saved plot image.
    """
    # Load and preprocess the traffic data
    df = pd.read_csv(csv_file)
    
    # Ensure the column names match the CSV file
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.rename(columns={'Date': 'ds', 'Vehicle_Count': 'y'})

    # Train the Prophet model
    model = Prophet()
    model.fit(df)

    # Make future predictions (30 days)
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Actual Traffic', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Traffic', color='orange')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightgray', label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Vehicle Count')
    plt.title('Traffic Prediction for the Next 30 Days')
    plt.legend()

    # Save the plot
    plot_path = "static/traffic_forecast.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path
