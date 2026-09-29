import os
import sqlite3
import pandas as pd

def create_sqlite_dbs():
    base_dir = "test_datasets"
    
    # 1. Relational HR Database
    hr_db_path = os.path.join(base_dir, "relational_hr.db")
    if os.path.exists(hr_db_path):
        os.remove(hr_db_path)
        
    conn_hr = sqlite3.connect(hr_db_path)
    pd.read_csv(f"{base_dir}/relational_hr/departments.csv").to_sql('departments', conn_hr, index=False)
    pd.read_csv(f"{base_dir}/relational_hr/employees.csv").to_sql('employees', conn_hr, index=False)
    conn_hr.close()
    print(f"Created SQLite DB: {hr_db_path}")

    # 2. Relational Retail Database
    retail_db_path = os.path.join(base_dir, "relational_retail.db")
    if os.path.exists(retail_db_path):
        os.remove(retail_db_path)
        
    conn_retail = sqlite3.connect(retail_db_path)
    pd.read_csv(f"{base_dir}/relational_retail/users.csv").to_sql('users', conn_retail, index=False)
    pd.read_csv(f"{base_dir}/relational_retail/products.csv").to_sql('products', conn_retail, index=False)
    pd.read_csv(f"{base_dir}/relational_retail/reviews.csv").to_sql('reviews', conn_retail, index=False)
    conn_retail.close()
    print(f"Created SQLite DB: {retail_db_path}")

if __name__ == "__main__":
    create_sqlite_dbs()
