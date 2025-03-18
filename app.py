from flask import Flask, render_template, request
import pandas as pd
import os
from traffic_prediction import generate_traffic_forecast  # Import traffic forecast
from purchases_analysis import (  # Import purchases analysis functions
    load_purchases_data,
    generate_top_suppliers_chart,
    generate_purchase_forecast,
    generate_price_optimization_chart,
    generate_reliability_chart
)

app = Flask(__name__)

# 📊 Path to CSV files
DATA_FOLDER = 'data'
if not os.path.exists('static'):
    os.makedirs('static')

# 🏠 Dashboard Route
@app.route('/')
def home():
    files = os.listdir(DATA_FOLDER)
    csv_files = [f.replace('.csv', '') for f in files if f.endswith('.csv')]
    return render_template('index.html', departments=csv_files)

# 📊 Unified Department Route
@app.route('/department/<name>')
def department(name):
    file_path = os.path.join(DATA_FOLDER, f"{name}.csv")

    if not os.path.exists(file_path):
        return f"No data available for {name} department."

    # Read CSV data
    df = pd.read_csv(file_path)

    # 🛑 Traffic Prediction Section
    plot_path = None
    if name.lower() == "traffic":
        plot_path = generate_traffic_forecast(file_path)

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

    return render_template(
        'department.html',
        name=name,
        columns=df.columns.tolist(),
        data=df.to_dict(orient='records'),
        image=plot_path if name.lower() == "traffic" else None,
        top_suppliers_chart=top_suppliers_chart,
        forecast_chart=forecast_chart,
        price_chart=price_chart,
        reliability_chart=reliability_chart
    )

# 🌐 Run the app
if __name__ == '__main__':
    app.run(debug=True)
