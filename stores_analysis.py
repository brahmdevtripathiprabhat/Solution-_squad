import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# ✅ Ensure output directory exists
output_dir = "static"
os.makedirs(output_dir, exist_ok=True)

# 📊 1. Inventory Turnover Analysis
def generate_inventory_turnover(csv_file):
    df = pd.read_csv(csv_file)

    # Group by Category and sum the stock quantity
    turnover_df = df.groupby('Category')['Stock_Quantity'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y='Stock_Quantity', data=turnover_df, palette='viridis')
    plt.title('Inventory Turnover by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Stock Quantity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inventory_turnover.png"))
    plt.close()

# 📈 2. Stock Level Forecasting
def generate_stock_forecast(csv_file):
    df = pd.read_csv(csv_file)

    # Convert date to datetime format
    df['Last_Stock_Update'] = pd.to_datetime(df['Last_Stock_Update'])

    # Sort by date
    df = df.sort_values('Last_Stock_Update')

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Last_Stock_Update', y='Stock_Quantity', hue='Category', data=df)
    plt.title('Stock Forecast by Category')
    plt.xlabel('Date')
    plt.ylabel('Stock Quantity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stock_forecast.png"))
    plt.close()

# 🔥 3. Stockout Risk Detection
def generate_stockout_risk(csv_file):
    df = pd.read_csv(csv_file)

    # Create Risk Level based on stock threshold
    df['Risk'] = np.where(df['Stock_Quantity'] < 100, 'High Risk', 'Low Risk')

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Risk', data=df, palette='coolwarm')
    plt.title('Stockout Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Count of Items')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stockout_risk.png"))
    plt.close()

# 💡 4. Optimal Reorder Point Calculation
def generate_reorder_point(csv_file):
    df = pd.read_csv(csv_file)

    # Simulating Average Demand per day (random values for now)
    np.random.seed(42)
    df['Daily_Demand'] = np.random.randint(5, 15, len(df))  # Random daily demand
    lead_time = 7  # Assume lead time of 7 days

    # Reorder point calculation
    df['Reorder_Point'] = df['Daily_Demand'] * lead_time

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Daily_Demand', y='Reorder_Point', hue='Category', data=df, palette='Set2')
    plt.title('Optimal Reorder Point by Demand')
    plt.xlabel('Daily Demand')
    plt.ylabel('Reorder Point')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reorder_point.png"))
    plt.close()
