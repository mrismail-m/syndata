import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, load_wine

def fetch_datasets():
    base_dir = "test_datasets"
    os.makedirs(base_dir, exist_ok=True)
    print(f"Saving datasets to {base_dir}/...")

    # 1. California Housing
    try:
        cal = fetch_california_housing(as_frame=True)
        cal.frame.to_csv(f"{base_dir}/california_housing.csv", index=False)
        print("1. California Housing (Tabular) - Downloaded")
    except Exception as e: print(f"Error fetching California Housing: {e}")

    # 2. Diabetes
    try:
        diabetes = load_diabetes(as_frame=True)
        diabetes.frame.to_csv(f"{base_dir}/diabetes.csv", index=False)
        print("2. Diabetes (Tabular) - Downloaded")
    except Exception as e: print(f"Error fetching Diabetes: {e}")

    # 3. Breast Cancer
    try:
        bc = load_breast_cancer(as_frame=True)
        bc.frame.to_csv(f"{base_dir}/breast_cancer.csv", index=False)
        print("3. Breast Cancer (Tabular) - Downloaded")
    except Exception as e: print(f"Error fetching Breast Cancer: {e}")

    # 4. Wine Quality
    try:
        wine = load_wine(as_frame=True)
        wine.frame.to_csv(f"{base_dir}/wine.csv", index=False)
        print("4. Wine Quality (Tabular) - Downloaded")
    except Exception as e: print(f"Error fetching Wine: {e}")

    # 5. Titanic (from seaborn URL)
    try:
        titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        titanic.to_csv(f"{base_dir}/titanic.csv", index=False)
        print("5. Titanic (Tabular) - Downloaded")
    except Exception as e: print(f"Error fetching Titanic: {e}")

    # 6. Insurance / Medical Cost
    try:
        insurance = pd.DataFrame({
            'age': np.random.randint(18, 65, 500),
            'sex': np.random.choice(['male', 'female'], 500),
            'bmi': np.random.uniform(15.0, 40.0, 500),
            'children': np.random.randint(0, 5, 500),
            'smoker': np.random.choice(['yes', 'no'], 500, p=[0.2, 0.8]),
            'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], 500),
            'charges': np.random.uniform(1000, 50000, 500)
        })
        insurance.to_csv(f"{base_dir}/insurance.csv", index=False)
        print("6. Medical Insurance (Tabular) - Generated")
    except Exception as e: print(f"Error fetching Insurance: {e}")

    # 7. Customer Churn (Telecom)
    try:
        churn = pd.DataFrame({
            'customerID': [f"7590-{i}G" for i in range(1000, 1500)],
            'gender': np.random.choice(['Male', 'Female'], 500),
            'SeniorCitizen': np.random.choice([0, 1], 500, p=[0.85, 0.15]),
            'tenure': np.random.randint(1, 72, 500),
            'MonthlyCharges': np.random.uniform(20.0, 120.0, 500),
            'Churn': np.random.choice(['Yes', 'No'], 500, p=[0.25, 0.75])
        })
        churn.to_csv(f"{base_dir}/telecom_churn.csv", index=False)
        print("7. Telecom Churn (Tabular) - Generated")
    except Exception as e: print(f"Error generating Churn: {e}")

    # 8. HR Analytics
    try:
        hr = pd.DataFrame({
            'employee_id': range(1, 501),
            'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], 500),
            'salary': np.random.choice(['low', 'medium', 'high'], 500),
            'satisfaction_level': np.random.uniform(0.1, 1.0, 500),
            'last_evaluation': np.random.uniform(0.3, 1.0, 500),
            'number_project': np.random.randint(2, 7, 500),
            'average_montly_hours': np.random.randint(130, 310, 500),
            'left': np.random.choice([0, 1], 500, p=[0.76, 0.24])
        })
        hr.to_csv(f"{base_dir}/hr_analytics.csv", index=False)
        print("8. HR Analytics (Tabular) - Generated")
    except Exception as e: print(f"Error generating HR Analytics: {e}")

    # 9. Relational: HR Hierarchy
    try:
        rel_dir = f"{base_dir}/relational_hr"
        os.makedirs(rel_dir, exist_ok=True)
        depts = pd.DataFrame({
            'dept_id': ['D1', 'D2', 'D3'],
            'dept_name': ['Engineering', 'Sales', 'Marketing'],
            'budget': [5000000, 2000000, 1500000]
        })
        emps = pd.DataFrame({
            'emp_id': range(1001, 1201),
            'name': [f"Emp_{i}" for i in range(1001, 1201)],
            'dept_id': np.random.choice(['D1', 'D2', 'D3'], 200, p=[0.6, 0.3, 0.1]),
            'salary': np.random.uniform(50000, 150000, 200)
        })
        depts.to_csv(f"{rel_dir}/departments.csv", index=False)
        emps.to_csv(f"{rel_dir}/employees.csv", index=False)
        print("9. Relational HR (Departments, Employees) - Generated")
    except Exception as e: print(f"Error generating Relational HR: {e}")

    # 10. Relational: Retail
    try:
        rel_dir = f"{base_dir}/relational_retail"
        os.makedirs(rel_dir, exist_ok=True)
        users = pd.DataFrame({
            'user_id': range(1, 101),
            'username': [f"User{i}" for i in range(1, 101)]
        })
        products = pd.DataFrame({
            'prod_id': range(1001, 1051),
            'prod_name': [f"Product{i}" for i in range(1001, 1051)],
            'price': np.random.uniform(5.0, 100.0, 50)
        })
        reviews = pd.DataFrame({
            'review_id': range(5001, 5501),
            'user_id': np.random.choice(users['user_id'], 500),
            'prod_id': np.random.choice(products['prod_id'], 500),
            'rating': np.random.randint(1, 6, 500),
            'review_date': pd.date_range(start='2022-01-01', periods=500)
        })
        users.to_csv(f"{rel_dir}/users.csv", index=False)
        products.to_csv(f"{rel_dir}/products.csv", index=False)
        reviews.to_csv(f"{rel_dir}/reviews.csv", index=False)
        print("10. Relational Retail (Users, Products, Reviews) - Generated")
    except Exception as e: print(f"Error generating Relational Retail: {e}")

if __name__ == "__main__":
    fetch_datasets()
