import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 200
regions    = ['North','South','East','West','Central']
categories = ['Electronics','Clothing','Furniture','Food','Sports']
months = pd.date_range('2022-01-01', periods=24, freq='ME')

rows = []
for i in range(n):
    month  = months[np.random.randint(0, len(months))]
    region = np.random.choice(regions)
    cat    = np.random.choice(categories)
    base   = {'Electronics':5000,'Clothing':2000,'Furniture':3500,'Food':800,'Sports':1500}[cat]
    sales  = max(0, base + np.random.normal(0, base*0.3))
    profit = sales * np.random.uniform(0.1, 0.35)
    rows.append({'Date': month.strftime('%Y-%m-%d'), 'Region': region, 'Category': cat,
                 'Sales': round(sales,2), 'Profit': round(profit,2),
                 'Units_Sold': int(sales / np.random.uniform(20, 200)),
                 'Discount': round(np.random.uniform(0, 0.3), 2)})

pd.DataFrame(rows).to_csv('/home/claude/ai_viz_assistant/uploads/sample_sales.csv', index=False)
print("Created sample_sales.csv")

students = pd.DataFrame({
    'Student_ID': range(1,151),
    'Gender': np.random.choice(['Male','Female'], 150),
    'Age': np.random.randint(18,26,150),
    'Department': np.random.choice(['CS','Math','Physics','Chemistry','Biology'],150),
    'Year': np.random.choice([1,2,3,4],150),
    'GPA': np.round(np.random.uniform(2.0,4.0,150),2),
    'Attendance': np.random.randint(60,100,150),
    'Score_Math': np.random.randint(40,100,150),
    'Score_Science': np.random.randint(40,100,150),
    'Score_English': np.random.randint(40,100,150),
})
students.to_csv('/home/claude/ai_viz_assistant/uploads/sample_students.csv', index=False)
print("Created sample_students.csv")
