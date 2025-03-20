import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

import os

# ✅ Ensure the static output directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# 📊 **1. Load and Preprocess Data**
def load_purchases_data(csv_file='data/purchases.csv'):
    """Load and preprocess the purchases data from a CSV file."""
    df = pd.read_csv(csv_file)
    
    # Convert dates to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Print columns for debugging
    print("Columns in CSV:", df.columns)

    # Sort by date
    df.sort_values('date', inplace=True)
    
    return df

# 📈 **2. Bar Chart: Top-performing Suppliers**
def generate_top_suppliers_chart(df):
    """Generate a bar chart showing top-performing suppliers by purchase quantity."""
    
    # ✅ Handle missing or differently named columns
    quantity_col = 'quantity' if 'quantity' in df.columns else df.select_dtypes(include='number').columns[0]

    # ✅ Clean non-numeric values
    df[quantity_col] = df[quantity_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    top_suppliers = df.groupby('supplier')[quantity_col].sum().reset_index()
    top_suppliers = top_suppliers.sort_values(quantity_col, ascending=False)

    plt.figure(figsize=(14, 10))
    sns.barplot(x=quantity_col, y='supplier', data=top_suppliers, palette='viridis')
    plt.title('Top-Performing Suppliers by Purchase Quantity')
    plt.xlabel('Total Quantity Purchased',fontsize=14)
    plt.ylabel('Supplier',fontsize=14)
    
 # ✅ Adjust label size and add padding
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
        # ✅ Add more space on the left if labels are too long
    plt.subplots_adjust(left=0.3)  # Increase left margin
    chart_path = 'static/top_suppliers.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# 🔮 **3. Time Series Forecasting for Future Purchase Trends**
def generate_purchase_forecast(purchases_df):
    """Generate a future forecast chart for purchases using Prophet."""

    # ✅ Handle missing or differently named amount/price columns
    if 'amount' in purchases_df.columns:
        amount_col = 'amount'
    elif 'price' in purchases_df.columns:
        amount_col = 'price'
    elif 'value' in purchases_df.columns:
        amount_col = 'value'
    else:
        # Fallback to the first numeric column
        amount_col = purchases_df.select_dtypes(include='number').columns[0]

    print(f"Using column '{amount_col}' for forecasting.")

    # Prepare the dataframe for Prophet
    forecast_df = purchases_df[['date', amount_col]].rename(columns={'date': 'ds', amount_col: 'y'})

    # ✅ Clean non-numeric values
    forecast_df['y'] = forecast_df['y'].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    # Fit the model
    model = Prophet()
    model.fit(forecast_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Forecast for 30 days
    forecast = model.predict(future)


    # Save the forecast chart
    fig = model.plot(forecast)
    # ✅ Change the X and Y-axis labels
    ax = fig.gca()  # Get current axes
    ax.set_xlabel('Date', fontsize=12)   # Replace 'ds' with 'Date'
    ax.set_ylabel('Purchase Amount', fontsize=12)  # Replace 'y' with 'Purchase Amount'

    fig.savefig('static/purchase_forecast.png')

    return 'static/purchase_forecast.png'

# 💰 **4. Price Optimization Using Linear Regression**
def generate_price_optimization_chart(df):
    """Generate a chart for price optimization using linear regression."""
    
    # ✅ Handle flexible column names
    quantity_col = 'quantity' if 'quantity' in df.columns else df.select_dtypes(include='number').columns[0]
    price_col = 'price' if 'price' in df.columns else df.select_dtypes(include='number').columns[1]

    # ✅ Clean non-numeric values
    df[quantity_col] = df[quantity_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)
    df[price_col] = df[price_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    # ✅ Ensure no NaN or infinite values
    df = df.dropna(subset=[quantity_col, price_col])

    X = df[[quantity_col]]
    y = df[price_col]

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
    """Generate a chart for supplier reliability classification using Random Forest."""

    # ✅ Handle dynamic column names
    delivery_col = 'delivery_time' if 'delivery_time' in df.columns else df.select_dtypes(include='number').columns[0]
    reliability_col = 'reliability' if 'reliability' in df.columns else df.select_dtypes(include='number').columns[1]

    # ✅ Clean non-numeric values
    df[delivery_col] = df[delivery_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)
    df[reliability_col] = df[reliability_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    # ✅ Ensure no NaN values
    df = df.dropna(subset=[delivery_col, reliability_col])

    X = df[[delivery_col]]
    y = df[reliability_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_test[delivery_col], y=predictions, label='Predicted Reliability', color='orange')
    sns.scatterplot(x=X_test[delivery_col], y=y_test, label='Actual Reliability', color='blue')
    plt.title('Supplier Reliability Prediction')
    plt.xlabel('Delivery Time (Days)')
    plt.ylabel('Reliability (1 = Reliable, 0 = Unreliable)')
    plt.legend()

    reliability_chart_path = 'static/reliability_chart.png'
    plt.savefig(reliability_chart_path)
    plt.close()

    return reliability_chart_path
