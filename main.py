from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
base_dir = Path(__file__).resolve().parent
sales_df = pd.read_csv(base_dir / 'sales_data (2).csv')
churn_df = pd.read_csv(base_dir / 'customer_churn.csv')

# 2. Data Cleaning & Preparation 
# Extract date parts
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
sales_df['Month'] = sales_df['Date'].dt.month
sales_df['Year'] = sales_df['Date'].dt.year

# Standardize IDs to merge (Match 'CUST001' to 'C00001')
sales_df['CustomerID_clean'] = sales_df['Customer_ID'].str.replace('CUST', 'C00')

# 3. Merging/Joining
merged_df = pd.merge(sales_df, churn_df, left_on='CustomerID_clean', right_on='CustomerID', how='inner')

# 4. Filter data using multiple conditions (AND operation)
retained_high_value = merged_df[(merged_df['Churn'] == 0) & (merged_df['Total_Sales'] > 100000)]

# 5. Aggregations (3 Types)
# Type 1: Group by Month (Sum)
monthly_sales = merged_df.groupby('Month')['Total_Sales'].sum().reset_index()

# Type 2: Group by Customer (Sum) for LTV
customer_ltv = merged_df.groupby('Customer_ID')['Total_Sales'].sum().sort_values(ascending=False).reset_index()

# Type 3: Group by Region (Mean & Count)
regional_performance = merged_df.groupby('Region').agg(
    Avg_Order_Value=('Total_Sales', 'mean'),
    Order_Count=('Total_Sales', 'count')
).reset_index()

# 6. Pivot Table (Summarization by Region and Product)
pivot_table = pd.pivot_table(
    merged_df, 
    values='Total_Sales', 
    index='Region', 
    columns='Product', 
    aggfunc='sum', 
    fill_value=0
)

# 7. Print Terminal Output for Report
print(f"Total Revenue: ${merged_df['Total_Sales'].sum():,.2f}")
print(f"Total Customers: {merged_df['Customer_ID'].nunique()}")
print(f"Average Order Value: ${merged_df['Total_Sales'].mean():,.2f}")
print(f"Top Customer: {customer_ltv.iloc[0]['Customer_ID']} - ${customer_ltv.iloc[0]['Total_Sales']:,.2f}")

print("\n--- Region vs Product Pivot Table ---")
print(pivot_table)

# 8. Visualizations (Dashboard Creation)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Customer Sales Analysis Dashboard', fontsize=16)

# Bar chart: Monthly Trends
axes[0, 0].bar(monthly_sales['Month'].astype(str), monthly_sales['Total_Sales'], color='skyblue')
axes[0, 0].set_title('Monthly Revenue Trend')

# Bar chart: Product Sales
product_sales = merged_df.groupby('Product')['Total_Sales'].sum().sort_values()
axes[0, 1].barh(product_sales.index, product_sales.values, color='lightgreen')
axes[0, 1].set_title('Sales by Product')

# Pie chart: Churn vs Retained Revenue
churn_revenue = merged_df.groupby('Churn')['Total_Sales'].sum()
axes[1, 0].pie(churn_revenue, labels=['Retained', 'Churned'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
axes[1, 0].set_title('Revenue: Retained vs Churned')

# Line Chart: Regional Avg Order Value
axes[1, 1].plot(regional_performance['Region'], regional_performance['Avg_Order_Value'], marker='o', color='purple')
axes[1, 1].set_title('Average Order Value by Region')

plt.tight_layout()
plt.show()