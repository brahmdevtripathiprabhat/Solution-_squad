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
def load_accounts_data(csv_file='data/accounts.csv'):
    """Load and preprocess the accounts data from a CSV file."""
    df = pd.read_csv(csv_file)
    
    # Convert dates to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Print columns for debugging
    print("Columns in CSV:", df.columns)

    # Sort by date
    df.sort_values('date', inplace=True)
    
    return df

# 📈 **2. Revenue Trend Analysis**
def generate_revenue_trend_chart(df):
    """Generate a line chart showing monthly revenue trends."""
    
    # ✅ Handle flexible column names
    revenue_col = 'revenue' if 'revenue' in df.columns else df.select_dtypes(include='number').columns[0]

    # ✅ Clean non-numeric values
    df[revenue_col] = df[revenue_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    monthly_revenue = df.resample('M', on='date')[revenue_col].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.lineplot(x='date', y=revenue_col, data=monthly_revenue, marker='o', color='green')
    plt.title('Monthly Revenue Trends', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Revenue', fontsize=12)
    
    plt.xticks(rotation=45)
    plt.grid(True)
    
    chart_path = 'static/revenue_trend.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# 🔮 **3. Revenue Forecasting Using Prophet**
def generate_revenue_forecast(df):
    """Generate a future forecast chart for revenue using Prophet."""

    # ✅ Handle flexible column names
    revenue_col = 'revenue' if 'revenue' in df.columns else df.select_dtypes(include='number').columns[0]

    # ✅ Clean non-numeric values
    df[revenue_col] = df[revenue_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    forecast_df = df[['date', revenue_col]].rename(columns={'date': 'ds', revenue_col: 'y'})

    # Fit the model
    model = Prophet()
    model.fit(forecast_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Forecast for 30 days
    forecast = model.predict(future)

    # Save the forecast chart
    fig = model.plot(forecast)
    ax = fig.gca()  
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Revenue', fontsize=12)

    forecast_chart_path = 'static/revenue_forecast.png'
    fig.savefig(forecast_chart_path)

    return forecast_chart_path

# 💵 **4. Expense Trend Analysis**
def generate_expense_trend_chart(df):
    """Generate a bar chart showing monthly expense trends."""

    # ✅ Handle flexible column names
    expense_col = 'expense' if 'expense' in df.columns else df.select_dtypes(include='number').columns[1]

    # ✅ Clean non-numeric values
    df[expense_col] = df[expense_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    monthly_expense = df.resample('M', on='date')[expense_col].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.barplot(x='date', y=expense_col, data=monthly_expense, palette='Reds')
    plt.title('Monthly Expense Trends', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Expense', fontsize=12)
    
    plt.xticks(rotation=45)
    plt.grid(True)

    chart_path = 'static/expense_trend.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# 📊 **5. Profitability Assessment**
def generate_profitability_chart(df):
    """Generate a chart showing monthly profitability trends."""
    
    # ✅ Handle flexible column names
    revenue_col = 'revenue' if 'revenue' in df.columns else df.select_dtypes(include='number').columns[0]
    expense_col = 'expense' if 'expense' in df.columns else df.select_dtypes(include='number').columns[1]

    # ✅ Clean non-numeric values
    df[revenue_col] = df[revenue_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)
    df[expense_col] = df[expense_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    # Calculate profitability
    df['profit'] = df[revenue_col] - df[expense_col]

    monthly_profit = df.resample('M', on='date')['profit'].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.lineplot(x='date', y='profit', data=monthly_profit, marker='o', color='blue')
    plt.title('Monthly Profitability Trends', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Profit', fontsize=12)

    plt.xticks(rotation=45)
    plt.grid(True)

    chart_path = 'static/profitability_chart.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# ✅ **6. Expense Classification Using Random Forest**
def generate_expense_classification_chart(df):
    """Generate a classification chart for expenses using Random Forest."""

    # ✅ Handle dynamic column names
    expense_col = 'expense' if 'expense' in df.columns else df.select_dtypes(include='number').columns[1]
    category_col = 'category' if 'category' in df.columns else df.select_dtypes(include='object').columns[0]

    # ✅ Clean non-numeric values
    df[expense_col] = df[expense_col].astype(str).str.replace(r'[^0-9.]', '', regex=True).astype(float)

    # Prepare data for classification
    X = df[[expense_col]]
    y = df[category_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_test[expense_col], y=predictions, label='Predicted Category', color='orange')
    plt.title('Expense Classification')
    plt.xlabel('Expense Amount')
    plt.ylabel('Category')
    plt.legend()

    chart_path = 'static/expense_classification.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path
