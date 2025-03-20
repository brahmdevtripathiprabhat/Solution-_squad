import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <-- Use non-GUI backend for server rendering
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# ✅ Ensure the static output directory exists
if not os.path.exists('static'):
    os.makedirs('static')


# 📊 **1. Load and Preprocess Traffic Data**
def load_traffic_data(csv_file='data/traffic.csv'):
    """Load and preprocess the traffic data from a CSV file."""
    df = pd.read_csv(csv_file)
    
    # Convert dates to datetime format
    df['Date & Time'] = pd.to_datetime(df['Date & Time'], errors='coerce')

    # ✅ Print columns for debugging
    print("Columns in CSV:", df.columns)

    # Sort by date
    df.sort_values('Date & Time', inplace=True)
    
    return df


# 🚦 **2. Traffic Density Analysis**
def generate_traffic_density_chart(df):
    """Generate a chart showing traffic density by route."""
    
    density_col = 'Congestion Level' if 'Congestion Level' in df.columns else df.select_dtypes(include='number').columns[0]

    # ✅ Clean non-numeric values
    df[density_col] = pd.to_numeric(df[density_col], errors='coerce')

    avg_density = df.groupby('Route Name')[density_col].mean().reset_index()

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Route Name', y=density_col, data=avg_density, palette='coolwarm')
    plt.title('Average Traffic Density by Route')
    plt.xticks(rotation=45)
    plt.xlabel('Route')
    plt.ylabel('Traffic Density')

    density_chart_path = 'static/traffic_density.png'
    plt.savefig(density_chart_path)
    plt.close()

    return density_chart_path


# 🚌 **3. Bus Demand vs. Supply Comparison**
def generate_bus_demand_vs_supply_chart(df):
    """Generate a chart comparing bus demand vs. supply on each route."""
    
    demand_col = 'Passenger Count' if 'Passenger Count' in df.columns else df.select_dtypes(include='number').columns[0]
    supply_col = 'Number of Stops' if 'Number of Stops' in df.columns else df.select_dtypes(include='number').columns[1]

    # ✅ Clean non-numeric values
    df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
    df[supply_col] = pd.to_numeric(df[supply_col], errors='coerce')

    route_demand = df.groupby('Route Name')[[demand_col, supply_col]].sum().reset_index()

    plt.figure(figsize=(14, 8))
    plt.bar(route_demand['Route Name'], route_demand[demand_col], label='Demand (Passengers)', color='skyblue')
    plt.bar(route_demand['Route Name'], route_demand[supply_col], label='Supply (Stops)', color='orange', alpha=0.7)
    plt.title('Bus Demand vs. Supply by Route')
    plt.xlabel('Route')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()

    demand_supply_chart_path = 'static/bus_demand_vs_supply.png'
    plt.savefig(demand_supply_chart_path)
    plt.close()

    return demand_supply_chart_path

# ⏳ **4. Delay Prediction Using Linear Regression**
def generate_delay_prediction_chart(df):
    """Predict delay based on congestion and traffic volume."""

    # ✅ Validate column names
    congestion_col = 'Congestion Level' if 'Congestion Level' in df.columns else df.select_dtypes(include='number').columns[0]
    delay_col = 'Delay Time (minutes)' if 'Delay Time (minutes)' in df.columns else df.select_dtypes(include='number').columns[1]

    # ✅ Clean non-numeric values and handle NaN
    df[congestion_col] = pd.to_numeric(df[congestion_col], errors='coerce')
    df[delay_col] = pd.to_numeric(df[delay_col], errors='coerce')

    # ✅ Remove NaN rows
    df = df.dropna(subset=[congestion_col, delay_col])

    # 🚫 Handle empty DataFrame case
    if df.empty:
        print("No valid data available for delay prediction.")
        return None

    # Prepare features and target
    X = df[[congestion_col]]
    y = df[delay_col]

    # ✅ Ensure enough data for splitting
    if len(X) < 2:
        print("Not enough data for model training.")
        return None

    # Train-test split with safety check
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ensure training set is not empty
        if len(X_train) == 0 or len(y_train) == 0:
            print("Insufficient data for training.")
            return None

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.scatter(X_test, y_test, label='Actual Delays', color='blue')
        plt.plot(X_test, predictions, label='Predicted Delays', color='red')
        plt.title('Delay Prediction Based on Traffic Congestion')
        plt.xlabel('Traffic Congestion Level')
        plt.ylabel('Delay Time (minutes)')
        plt.legend()

        delay_chart_path = 'static/delay_prediction.png'
        plt.savefig(delay_chart_path)
        plt.close()

        return delay_chart_path

    except ValueError as e:
        print(f"Error during model training: {e}")
        return None

# 🚍 **5. Bus Type Demand Analysis**
def generate_bus_type_demand_chart(df):
    """Generate a chart showing demand by bus type (AC, Express, Local, etc.)."""
    
    bus_type_col = 'Bus Type' if 'Bus Type' in df.columns else 'Vehicle Number'  # Fallback column
    passenger_col = 'Passenger Count' if 'Passenger Count' in df.columns else df.select_dtypes(include='number').columns[0]

    # ✅ Clean non-numeric values
    df[passenger_col] = pd.to_numeric(df[passenger_col], errors='coerce')

    bus_type_demand = df.groupby(bus_type_col)[passenger_col].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.barplot(x=bus_type_col, y=passenger_col, data=bus_type_demand, palette='viridis')
    plt.title('Bus Type Demand (Passenger Count)')
    plt.xlabel('Bus Type')
    plt.ylabel('Total Passengers')
    plt.xticks(rotation=45)

    bus_type_chart_path = 'static/bus_type_demand.png'
    plt.savefig(bus_type_chart_path)
    plt.close()

    return bus_type_chart_path


# ✅ **6. Forecasting Traffic Trends Using Prophet**
def generate_traffic_forecast(df):
    """Forecast future traffic volume trends using Prophet."""
    
    date_col = 'Date & Time' if 'Date & Time' in df.columns else 'Date'
    volume_col = 'Traffic Volume' if 'Traffic Volume' in df.columns else df.select_dtypes(include='number').columns[0]

    # Prepare the data
    forecast_df = df[[date_col, volume_col]].rename(columns={date_col: 'ds', volume_col: 'y'})

    # ✅ Clean non-numeric values
    forecast_df['y'] = pd.to_numeric(forecast_df['y'], errors='coerce')

    # Handle NaN values
    forecast_df.dropna(subset=['y'], inplace=True)

    # Fit the model
    model = Prophet()
    model.fit(forecast_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save the forecast chart
    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Traffic Volume', fontsize=12)

    forecast_chart_path = 'static/traffic_forecast.png'
    fig.savefig(forecast_chart_path)
    
    return forecast_chart_path
