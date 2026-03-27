import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel("C:\\Users\\HP\\Downloads\\Kevin Cookie Company Financials.xlsx", sheet_name="Cookie Sales")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# =========================
# 🔹 FEATURE ENGINEERING
# =========================

# Monthly sales aggregation (IMPORTANT for ML)
monthly_sales = df.groupby(['Year', 'Month Number'])['Revenue'].sum().reset_index()

# Create proper date column for monthly data
monthly_sales['Date'] = pd.to_datetime(monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month Number'].astype(str) + '-01')

# Country-wise sales
country_sales = df.groupby('Country').agg({
    'Revenue': 'sum',
    'Profit': 'sum'
}).reset_index()

# Product performance
product_sales = df.groupby('Product').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'Units Sold': 'sum'
}).reset_index()

# Profit Margin
df['Profit Margin'] = df['Profit'] / df['Revenue']

# =========================
# 🔹 VISUALIZATION
# =========================

plt.figure()
plt.plot(monthly_sales['Date'], monthly_sales['Revenue'])
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# =========================
# 🔹 PREDICTIVE ANALYTICS
# =========================

# Convert date to numeric
monthly_sales['Date_ordinal'] = monthly_sales['Date'].map(pd.Timestamp.toordinal)

X = monthly_sales[['Date_ordinal']]
y = monthly_sales['Revenue']

model = LinearRegression()
model.fit(X, y)

# Predict next 6 months
future_dates = pd.date_range(
    start=monthly_sales['Date'].max() + pd.offsets.MonthEnd(1),
    periods=6,
    freq='ME'
)
future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1,1)

predictions = model.predict(future_ordinal)

# Create prediction DataFrame
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Revenue': predictions
})

# =========================
# 🔹 EXPORT DATA FOR DASHBOARD
# =========================

monthly_sales.to_csv("monthly_sales.csv", index=False)
country_sales.to_csv("country_sales.csv", index=False)
product_sales.to_csv("product_sales.csv", index=False)
forecast_df.to_csv("forecast_sales.csv", index=False)

print("✅ Data exported successfully!")