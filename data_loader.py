import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def load_olist_data(orders_path: str, items_path: str):
    """Load real Olist CSVs from disk."""
    orders = pd.read_csv(orders_path)
    items  = pd.read_csv(items_path)
    return orders, items


def load_sample_data(n_customers: int = 500, seed: int = 42) -> tuple:
    """
    Generate a realistic synthetic Olist-schema dataset for demo purposes.
    Returns (orders_df, items_df).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # ── Customer pool ─────────────────────────────────────────────────────────
    customer_ids = [f"CUST_{i:05d}" for i in range(n_customers)]

    # Assign behavioural archetypes to mimic real RFM distribution
    archetypes = rng.choice(
        ["champion", "loyal", "promising", "at_risk", "lost", "new"],
        size=n_customers,
        p=[0.12, 0.18, 0.15, 0.20, 0.20, 0.15],
    )

    end_date   = datetime(2018, 9, 1)
    start_date = datetime(2016, 9, 4)

    orders_rows = []
    items_rows  = []
    order_counter = 1

    for cid, archetype in zip(customer_ids, archetypes):
        # Number of orders & recency profile per archetype
        if archetype == "champion":
            n_orders = int(rng.integers(8, 20))
            days_since = int(rng.integers(1, 30))
        elif archetype == "loyal":
            n_orders = int(rng.integers(4, 10))
            days_since = int(rng.integers(20, 90))
        elif archetype == "promising":
            n_orders = int(rng.integers(2, 5))
            days_since = int(rng.integers(10, 60))
        elif archetype == "at_risk":
            n_orders = int(rng.integers(3, 8))
            days_since = int(rng.integers(180, 365))
        elif archetype == "lost":
            n_orders = int(rng.integers(1, 3))
            days_since = int(rng.integers(300, 720))
        else:  # new
            n_orders = 1
            days_since = int(rng.integers(1, 45))

        # Generate order timestamps spread across history
        last_purchase = end_date - timedelta(days=days_since)
        order_dates = sorted([
            last_purchase - timedelta(days=int(rng.integers(0, 730)))
            for _ in range(n_orders - 1)
        ] + [last_purchase])
        order_dates = [d for d in order_dates if d >= start_date]
        if not order_dates:
            order_dates = [last_purchase]

        for od in order_dates:
            oid = f"ORD_{order_counter:07d}"
            order_counter += 1

            orders_rows.append({
                "order_id":                  oid,
                "customer_id":               cid,
                "order_status":              "delivered",
                "order_purchase_timestamp":  od.strftime("%Y-%m-%d %H:%M:%S"),
            })

            # 1–4 items per order
            n_items = int(rng.integers(1, 5))
            for _ in range(n_items):
                if archetype == "champion":
                    price = round(float(rng.uniform(80, 600)), 2)
                elif archetype == "loyal":
                    price = round(float(rng.uniform(50, 300)), 2)
                else:
                    price = round(float(rng.uniform(20, 200)), 2)

                items_rows.append({
                    "order_id":          oid,
                    "order_item_id":     1,
                    "product_id":        f"PROD_{rng.integers(1, 200):04d}",
                    "seller_id":         f"SELL_{rng.integers(1, 50):03d}",
                    "price":             price,
                    "freight_value":     round(float(rng.uniform(5, 40)), 2),
                })

    orders_df = pd.DataFrame(orders_rows)
    items_df  = pd.DataFrame(items_rows)

    return orders_df, items_df
