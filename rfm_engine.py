import pandas as pd
import numpy as np
from datetime import datetime


def compute_rfm(orders_df: pd.DataFrame, items_df: pd.DataFrame,
                recency_bins: int = 4, frequency_bins: int = 4,
                monetary_bins: int = 4) -> pd.DataFrame:
    """
    Compute RFM scores from Olist orders + order_items DataFrames.
    Handles both the real Kaggle schema and the synthetic sample schema.
    """
    # ── Normalise column names (real Kaggle vs sample) ───────────────────────
    orders = orders_df.copy()
    items  = items_df.copy()

    # Real Kaggle schema uses 'customer_unique_id' after merge with customers;
    # we use 'customer_id' as the customer key throughout.
    if "customer_unique_id" in orders.columns:
        orders["customer_id"] = orders["customer_unique_id"]

    # Determine date column
    date_col = None
    for c in ["order_purchase_timestamp", "order_date", "purchase_date"]:
        if c in orders.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Cannot find a date column in orders DataFrame.")

    orders[date_col] = pd.to_datetime(orders[date_col], errors="coerce")
    orders = orders.dropna(subset=[date_col, "customer_id"])

    # Keep only delivered orders if status column exists
    if "order_status" in orders.columns:
        orders = orders[orders["order_status"] == "delivered"]

    # ── Revenue per order ────────────────────────────────────────────────────
    order_id_col = "order_id" if "order_id" in items.columns else items.columns[0]
    price_col    = "price" if "price" in items.columns else "payment_value"

    if price_col not in items.columns:
        # fallback: use payment_value from orders if available
        if "payment_value" in orders.columns:
            order_revenue = orders[["order_id", "payment_value"]].rename(
                columns={"payment_value": "revenue"})
        else:
            orders["revenue"] = 1
            order_revenue = orders[["order_id", "revenue"]]
    else:
        order_revenue = (
            items.groupby(order_id_col)[price_col]
            .sum()
            .reset_index()
            .rename(columns={order_id_col: "order_id", price_col: "revenue"})
        )

    orders = orders.merge(order_revenue, on="order_id", how="left")
    orders["revenue"] = orders["revenue"].fillna(0)

    # ── Aggregate per customer ────────────────────────────────────────────────
    snapshot = orders[date_col].max() + pd.Timedelta(days=1)

    rfm = (
        orders.groupby("customer_id")
        .agg(
            last_purchase=(date_col, "max"),
            frequency=("order_id", "nunique"),
            monetary_value=("revenue", "sum"),
        )
        .reset_index()
    )
    rfm["recency_days"] = (snapshot - rfm["last_purchase"]).dt.days

    # ── Quantile scoring ─────────────────────────────────────────────────────
    def safe_qcut(series, q, labels, ascending=True):
        """qcut with duplicate-edge handling."""
        try:
            result = pd.qcut(series.rank(method="first"), q=q, labels=labels)
        except Exception:
            result = pd.cut(series, bins=q,
                            labels=labels if ascending else list(reversed(labels)))
        return result.astype(int)

    r_labels = list(range(recency_bins, 0, -1))   # lower recency = better
    f_labels = list(range(1, frequency_bins + 1))
    m_labels = list(range(1, monetary_bins + 1))

    rfm["r_score"] = safe_qcut(rfm["recency_days"],  recency_bins,  r_labels)
    rfm["f_score"] = safe_qcut(rfm["frequency"],     frequency_bins, f_labels)
    rfm["m_score"] = safe_qcut(rfm["monetary_value"], monetary_bins, m_labels)

    rfm["rfm_total_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # ── Segmentation ─────────────────────────────────────────────────────────
    rfm["customer_segment"] = rfm.apply(segment_customer, axis=1)

    rfm = rfm.set_index("customer_id")
    return rfm


def segment_customer(row) -> str:
    r, f, m = row["r_score"], row["f_score"], row["m_score"]

    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    elif r >= 3 and f >= 2:
        return "Loyal Customers"
    elif r >= 3 and f == 1:
        return "Promising"
    elif r == 2 and f >= 2:
        return "Potential Loyalists"
    elif r == 2 and f == 1:
        return "Needs Attention"
    elif r == 1 and f >= 3:
        return "At Risk"
    elif r == 1 and f == 2:
        return "Hibernating"
    else:
        return "Lost"


def get_recommendations() -> dict:
    return {
        "Champions": {
            "recommendation": "Reward them. They are your best customers.",
            "suggested_action": "VIP perks, early access, loyalty rewards, exclusive offers",
        },
        "Loyal Customers": {
            "recommendation": "Upsell higher-value products and deepen relationship.",
            "suggested_action": "Premium offerings, bundles, referral programs",
        },
        "Promising": {
            "recommendation": "Build relationship and increase purchase frequency.",
            "suggested_action": "Onboarding series, product education, limited-time deals",
        },
        "Potential Loyalists": {
            "recommendation": "Offer membership or loyalty programs.",
            "suggested_action": "Free trials, product demos, personalised recommendations",
        },
        "Needs Attention": {
            "recommendation": "Offer relevant recommendations before they slip away.",
            "suggested_action": "Special discounts, surveys, re-engagement campaigns",
        },
        "At Risk": {
            "recommendation": "Re-engage with win-back campaigns immediately.",
            "suggested_action": "Personalised emails, product updates, reactivation incentives",
        },
        "Hibernating": {
            "recommendation": "Revive with aggressive offers.",
            "suggested_action": "Deep discounts, reactivation emails, urgency messaging",
        },
        "Lost": {
            "recommendation": "Make limited time offers — last chance to recover.",
            "suggested_action": "Time-sensitive deals, flash sales, sunset or remove",
        },
    }
