from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

app = Flask(__name__)

# Load Trained XGBoost Model
xgb_model = joblib.load('xgb_model.pkl')

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Year-Month'] = pd.to_datetime(df['Year-Month'])
    df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce')
    df['Stock Available'] = df['Stock Available'].fillna(0).astype(int)

    df = df.groupby(['Medicine', 'Category', 'Year-Month', 'Season'], as_index=False).agg({
        'Units Sold': 'sum',
        'Stock Available': 'last',
        'Expiry Date': 'first'
    })
    df = df.sort_values(by='Year-Month')
    return df

df = load_data("medicine_sales_monthly_12rows.csv")

# List of months
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

@app.route('/')
@app.route('/')
def home():
    medicine_list = df['Medicine'].unique()  # Get unique medicine names from the dataset
    return render_template('index.html', medicine_list=medicine_list)


@app.route('/predict', methods=['POST'])
def predict():
    medicine_name = request.form['medicine_name']
    med_data = df[df["Medicine"].str.lower() == medicine_name.lower()].copy()

    if med_data.empty:
        return render_template('result.html', error="Medicine not found!")

    try:
        med_data['Month'] = med_data['Year-Month'].dt.month
        med_data['Year'] = med_data['Year-Month'].dt.year

        future_dates = pd.date_range(start=med_data['Year-Month'].max(), periods=12, freq='MS')
        future_features = pd.DataFrame({
            'Month': future_dates.month,
            'Year': future_dates.year,
            'Stock Available': med_data['Stock Available'].iloc[-1]
        })

        # Predict using Loaded XGBoost Model
        future_sales_ml = xgb_model.predict(future_features)

        sarima_model = SARIMAX(med_data['Units Sold'], order=(1,1,1), seasonal_order=(1,1,1,12), trend='c')
        sarima_fit = sarima_model.fit(disp=False)
        future_sales_sarima = sarima_fit.forecast(steps=12)

        future_sales = (0.7 * future_sales_ml + 0.3 * future_sales_sarima.values)

        latest_entry = med_data.iloc[-1]
        current_stock = latest_entry["Stock Available"]
        expiry_date = latest_entry["Expiry Date"]
        required_production = max(0, int(sum(future_sales) - current_stock))
        predicted_expiry = max(latest_entry['Expiry Date'] + pd.DateOffset(years=1), latest_entry['Year-Month'] + pd.DateOffset(years=1))

        peak_season = med_data.groupby("Season")["Units Sold"].sum().idxmax()

        # Generate Pie Chart
        season_sales = med_data.groupby("Season")["Units Sold"].sum()
        fig, ax = plt.subplots(figsize=(7,7))
        ax.pie(season_sales, labels=season_sales.index, autopct='%1.1f%%',
               colors=['skyblue', 'lightgreen', 'coral', 'gold'],
               startangle=140, wedgeprops={'edgecolor': 'black'})
        ax.set_title(f"Sales Distribution Across Seasons for {medicine_name}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return render_template('result.html',
                               medicine_name=medicine_name,
                               future_sales=future_sales,
                               current_stock=current_stock,
                               required_production=required_production,
                               expiry_date=expiry_date.date(),
                               predicted_expiry=predicted_expiry.date(),
                               peak_season=peak_season,
                               chart_base64=chart_base64,
                               months=months)

    except Exception as e:
        return render_template('result.html', error=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
