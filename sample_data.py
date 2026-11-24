import pandas as pd
import numpy as np


def generate_messy_sales_data(n_rows=500):
    np.random.seed(42)
    products = ["Organic Milk 1L", "Potato Chips 150g", "Whole Wheat Bread", "Arabica Coffee 250g", "Olive Oil 500ml"]
    base_prices = {"Organic Milk 1L": 65, "Potato Chips 150g": 40, "Whole Wheat Bread": 50, "Arabica Coffee 250g": 350, "Olive Oil 500ml": 600}
    stores = ["Store_Delhi_01", "Store_Gurugram_02", "Store_Mumbai_03", "Store_Bengaluru_04"]

    chosen_products = np.random.choice(products, n_rows)
    prices = [base_prices[p] * np.random.uniform(0.95, 1.05) for p in chosen_products]
    quantities = np.random.randint(1, 15, n_rows)

    df = pd.DataFrame({
        "Order_ID": [f"ORD-{1000 + i}" for i in range(n_rows)],
        "Store_ID": np.random.choice(stores, n_rows),
        "Product": chosen_products,
        "Unit_Price": np.round(prices, 2),
        "Quantity": quantities,
    })
    df["Total_Sales"] = np.round(df["Unit_Price"] * df["Quantity"], 2)

    # --- INJECT REALISTIC INDUSTRIAL DATA ERRORS ---
    # 1. Price spike anomalies (e.g., ₹45,000 bag of chips)
    df.loc[[12, 87, 214, 390], "Unit_Price"] = [45000.0, 89000.0, 32000.0, 61000.0]
    df["Total_Sales"] = np.round(df["Unit_Price"] * df["Quantity"], 2)

    # 2. Negative quantities (corrupted POS returns)
    df.loc[[45, 150, 310], "Quantity"] = [-5, -12, -3]

    # 3. Missing values (null prices & store IDs)
    df.loc[20:32, "Unit_Price"] = np.nan
    df.loc[100:108, "Store_ID"] = np.nan

    # 4. Duplicate order rows
    duplicates = df.iloc[5:12].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df
