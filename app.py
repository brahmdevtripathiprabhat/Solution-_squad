from flask import Flask, render_template, request
import pandas as pd
import os

# Import traffic forecast
from traffic_analysis import generate_traffic_forecast  

# Import purchases analysis functions
from purchases_analysis import (  
    load_purchases_data,
    generate_top_suppliers_chart,
    generate_purchase_forecast,
    generate_price_optimization_chart,
    generate_reliability_chart
)

# Import accounts analysis functions
from  account_chart import (
    load_accounts_data,
    generate_revenue_trend_chart,
    generate_expense_trend_chart,
    generate_profitability_chart,
    generate_revenue_forecast
)
# Import traffic analysis functions
from traffic_analysis import (
    load_traffic_data,
    generate_traffic_density_chart,
    generate_bus_demand_vs_supply_chart,
    generate_delay_prediction_chart,
    generate_bus_type_demand_chart,
    generate_traffic_forecast
)
app = Flask(__name__)

# 📊 Path to CSV files
DATA_FOLDER = 'data'
if not os.path.exists('static'):
    os.makedirs('static')

# 🏠 Dashboard Route
@app.route('/')
def home():
    """Homepage displaying available CSV departments."""
    files = os.listdir(DATA_FOLDER)
    csv_files = [f.replace('.csv', '') for f in files if f.endswith('.csv')]
    return render_template('index.html', departments=csv_files)

# 📊 Unified Department Route
@app.route('/department/<name>')
def department(name):
    """Render department-specific data and charts."""
    
    file_path = os.path.join(DATA_FOLDER, f"{name}.csv")

    if not os.path.exists(file_path):
        return f"No data available for {name} department."

    # Read CSV data
    df = pd.read_csv(file_path)


    # 🛑 Purchases Section
    top_suppliers_chart = None
    forecast_chart = None
    price_chart = None
    reliability_chart = None

    if name.lower() == "purchases":
        purchases_df = load_purchases_data(file_path)
        
        # Generate charts
        top_suppliers_chart = generate_top_suppliers_chart(purchases_df)
        forecast_chart = generate_purchase_forecast(purchases_df)
        price_chart = generate_price_optimization_chart(purchases_df)
        reliability_chart = generate_reliability_chart(purchases_df)

    # 🛑 Accounts Section
    revenue_chart = None
    expense_chart = None
    profitability_chart = None
    account_forecast_chart = None

    if name.lower() == "accounts":
        accounts_df = load_accounts_data(file_path)

        # Generate accounts-specific charts
        revenue_chart = generate_revenue_trend_chart(accounts_df)
        expense_chart = generate_expense_trend_chart(accounts_df)
        profitability_chart = generate_profitability_chart(accounts_df)


        account_forecast_chart = generate_revenue_forecast(accounts_df)

# 🚍 Traffic Section
    traffic_density_chart = None
    bus_demand_chart = None
    delay_chart = None
    bus_type_demand_chart = None
    traffic_forecast_chart = None

    if name.lower() == "traffic":
        traffic_df = load_traffic_data(file_path)

        # Generate traffic charts
        traffic_density_chart = generate_traffic_density_chart(traffic_df)
        bus_demand_chart = generate_bus_demand_vs_supply_chart(traffic_df)
        delay_chart = generate_delay_prediction_chart(traffic_df)
        bus_type_demand_chart = generate_bus_type_demand_chart(traffic_df)
        traffic_forecast_chart = generate_traffic_forecast(traffic_df)

    return render_template( 
        'department.html',
        name=name,
        columns=df.columns.tolist(),
        data=df.to_dict(orient='records'),
        
        
        # Purchases charts
        top_suppliers_chart=top_suppliers_chart,
        forecast_chart=forecast_chart,
        price_chart=price_chart,
        reliability_chart=reliability_chart,

        # Accounts charts
        revenue_chart=revenue_chart,
        expense_chart=expense_chart,
        profitability_chart=profitability_chart,
        account_forecast_chart=account_forecast_chart,

         # Traffic charts
        traffic_density_chart=traffic_density_chart,
        bus_demand_chart=bus_demand_chart,
        delay_chart=delay_chart,
        bus_type_demand_chart=bus_type_demand_chart,
        traffic_forecast_chart=traffic_forecast_chart
    )

# 🌐 Run the app
if __name__ == '__main__':
    app.run(debug=True)
