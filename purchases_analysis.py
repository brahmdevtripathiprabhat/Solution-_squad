import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import os

# ✅ Ensure output directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# 📊 **1. Load and Preprocess Data**
def load_purchases_data(csv_file='data/purchases.csv'):
    df = pd.read_csv(csv_file)
    
    # Convert dates to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df.sort_values('date', inplace=True)
    
    return df

# 📈 **2. Bar Chart: Top-performing Suppliers**
def generate_top_suppliers_chart(df):
    top_suppliers = df.groupby('supplier')['quantity'].sum().reset_index()
    top_suppliers = top_suppliers.sort_values('quantity', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='quantity', y='supplier', data=top_suppliers, palette='viridis')
    plt.title('Top-Performing Suppliers by Purchase Quantity')
    plt.xlabel('Total Quantity Purchased')
    plt.ylabel('Supplier')

    chart_path = 'static/top_suppliers.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# 🔮 **3. Time Series Forecasting for Future Purchase Trends**
def generate_purchase_forecast(df, periods=30):
    forecast_df = df[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})

    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3, label='Confidence Interval')
    plt.scatter(forecast_df['ds'], forecast_df['y'], label='Actual Data', color='blue')
    plt.title('Purchase Forecast for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Purchase Quantity')
    plt.legend()

    forecast_path = 'static/purchase_forecast.png'
    plt.savefig(forecast_path)
    plt.close()

    return forecast_path

# 💰 **4. Price Optimization Using Linear Regression**
def generate_price_optimization_chart(df):
    X = df[['quantity']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    plt.scatter(X_test, y_test, label='Actual Prices', color='blue')
    plt.plot(X_test, predictions, label='Predicted Prices', color='red')
    plt.title('Price Optimization (Regression Model)')
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.legend()

    price_chart_path = 'static/price_optimization.png'
    plt.savefig(price_chart_path)
    plt.close()

    return price_chart_path

# ✅ **5. Supplier Reliability Classification**
def generate_reliability_chart(df):
    X = df[['delivery_time']]
    y = df['reliability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_test['delivery_time'], y=predictions, label='Predicted Reliability', color='orange')
    sns.scatterplot(x=X_test['delivery_time'], y=y_test, label='Actual Reliability', color='blue')
    plt.title('Supplier Reliability Prediction')
    plt.xlabel('Delivery Time (Days)')
    plt.ylabel('Reliability (1 = Reliable, 0 = Unreliable)')
    plt.legend()

    reliability_chart_path = 'static/reliability_chart.png'
    plt.savefig(reliability_chart_path)
    plt.close()

    return reliability_chart_path
