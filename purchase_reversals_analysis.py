#!/usr/bin/env python3
"""
========================================================================================
Project: Understanding Purchase Reversals in Online Retail Through Customer Behaviour
         and Predictive Analytics
Dataset: Online Retail (UCI Machine Learning Repository, 2010 - 2011)
Goal:    Identify patterns associated with cancelled purchases, analyze product & customer
         dynamics, build an association network, segment customers via K-Means, and
         predict order cancellation risk at order placement time using 5 ML models.
========================================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ML & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Association Rules
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Set aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Online Retail.xlsx')
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'online_retail_cache.parquet')


def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


# ======================================================================================
# STEP 1: DATA LOADING, INSPECTION (HEAD, INFO, DESCRIBE)
# ======================================================================================
def load_and_inspect_data():
    print_section("Step 1: Data Loading & Initial Inspection")
    
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH}...")
        df = pd.read_parquet(CACHE_PATH)
    else:
        print(f"Loading raw dataset from {DATA_PATH} (this may take ~20 seconds)...")
        df = pd.read_excel(DATA_PATH)
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        try:
            df.to_parquet(CACHE_PATH)
            print(f"Saved optimized parquet cache to {CACHE_PATH} for fast subsequent loads.")
        except Exception as e:
            print(f"Parquet caching skipped: {e}")

    print(f"\nInitial Dataset Dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    print("\n--- DataFrame .head(5) ---")
    print(df.head())

    print("\n--- DataFrame .info() ---")
    df.info()

    print("\n--- DataFrame .describe() ---")
    print(df.describe())

    print("\n--- Missing Value Audit ---")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing Count': null_counts, 'Percentage (%)': null_pct})
    print(missing_summary[missing_summary['Missing Count'] > 0])

    return df


# ======================================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ======================================================================================
def clean_and_preprocess_data(df):
    print_section("Step 2: Data Cleaning & Preprocessing")

    initial_len = len(df)
    # Strip whitespace from descriptions and stock codes
    df['Description'] = df['Description'].astype(str).str.strip()
    df['StockCode'] = df['StockCode'].astype(str).str.strip()

    # Identify cancellation records by 'C' prefix in InvoiceNo
    df['IsCancellation'] = df['InvoiceNo'].astype(str).str.startswith('C')

    # Remove non-cancellation anomalies where UnitPrice <= 0 or Quantity <= 0 (damaged stock, fees, etc.)
    clean_mask = (df['CustomerID'].notnull()) & (df['UnitPrice'] > 0)
    cleaned_df = df[clean_mask].copy()
    cleaned_df['CustomerID'] = cleaned_df['CustomerID'].astype(int)

    # Distinguish positive purchases from cancellation events
    purchases = cleaned_df[(~cleaned_df['IsCancellation']) & (cleaned_df['Quantity'] > 0)].copy()
    cancellations = cleaned_df[cleaned_df['IsCancellation'] & (cleaned_df['Quantity'] < 0)].copy()
    cancellations['AbsQuantity'] = cancellations['Quantity'].abs()

    print(f"Total raw records:                     {initial_len:,}")
    print(f"Records with missing CustomerID removed: {df['CustomerID'].isnull().sum():,}")
    print(f"Valid cleaned records:                 {len(cleaned_df):,}")
    print(f"  -> Valid Completed Purchases:        {len(purchases):,} items across {purchases['InvoiceNo'].nunique():,} invoices")
    print(f"  -> Cancellation/Reversal Events:     {len(cancellations):,} items across {cancellations['InvoiceNo'].nunique():,} invoices")
    print(f"  -> Unique active customers:          {cleaned_df['CustomerID'].nunique():,}")

    return cleaned_df, purchases, cancellations


# ======================================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & REVERSAL PATTERNS
# ======================================================================================
def conduct_eda(purchases, cancellations):
    print_section("Step 3: Exploratory Data Analysis & Business Questions")

    # 1. Product Cancellation Analysis & Revenue vs. Cancellation Paradox
    purchases['PurchRevenue'] = purchases['Quantity'] * purchases['UnitPrice']
    cancellations['CancRevenue'] = cancellations['AbsQuantity'] * cancellations['UnitPrice']

    prod_p = purchases.groupby(['StockCode', 'Description']).agg(
        SoldQty=('Quantity', 'sum'),
        GrossRevenue=('PurchRevenue', 'sum'),
        OrderCount=('InvoiceNo', 'nunique')
    ).reset_index()

    prod_c = cancellations.groupby('StockCode').agg(
        CancQty=('AbsQuantity', 'sum'),
        LostRevenue=('CancRevenue', 'sum'),
        CancOrderCount=('InvoiceNo', 'nunique')
    ).reset_index()

    prod_metrics = pd.merge(prod_p, prod_c, on='StockCode', how='left').fillna(0)
    prod_metrics['NetRevenue'] = prod_metrics['GrossRevenue'] - prod_metrics['LostRevenue']
    prod_metrics['CancellationRate_Qty'] = prod_metrics['CancQty'] / (prod_metrics['SoldQty'] + prod_metrics['CancQty'])
    prod_metrics['RevenueLossRate'] = np.where(prod_metrics['GrossRevenue'] > 0,
                                               prod_metrics['LostRevenue'] / prod_metrics['GrossRevenue'], 0.0)

    # Top frequently cancelled products by volume
    top_cancelled_vol = prod_metrics.sort_values(by='CancQty', ascending=False).head(10)
    print("\n--- Top 10 Products by Total Cancelled Quantity ---")
    print(top_cancelled_vol[['StockCode', 'Description', 'SoldQty', 'CancQty', 'CancellationRate_Qty', 'LostRevenue']])

    # Surprising Insight: High Gross Revenue with High Cancellation Rate
    paradox_products = prod_metrics[(prod_metrics['GrossRevenue'] > 5000) & (prod_metrics['RevenueLossRate'] > 0.15)].sort_values(by='RevenueLossRate', ascending=False)
    print("\n--- The Revenue Paradox: High Revenue Products with Disproportionate Reversals ---")
    print(paradox_products[['StockCode', 'Description', 'GrossRevenue', 'LostRevenue', 'RevenueLossRate', 'CancellationRate_Qty']].head(8))

    # 2. Monthly Trends
    purchases['YearMonth'] = purchases['InvoiceDate'].dt.to_period('M')
    cancellations['YearMonth'] = cancellations['InvoiceDate'].dt.to_period('M')

    monthly_p = purchases.groupby('YearMonth').agg(Purchases=('InvoiceNo', 'nunique'), Revenue=('PurchRevenue', 'sum'))
    monthly_c = cancellations.groupby('YearMonth').agg(Cancellations=('InvoiceNo', 'nunique'), LostRevenue=('CancRevenue', 'sum'))
    monthly_trends = pd.concat([monthly_p, monthly_c], axis=1).fillna(0)
    monthly_trends['CancellationRate'] = monthly_trends['Cancellations'] / (monthly_trends['Purchases'] + monthly_trends['Cancellations'])
    monthly_trends.index = monthly_trends.index.astype(str)

    print("\n--- Monthly Transaction & Reversal Trends ---")
    print(monthly_trends)

    # 3. Country-Wise Comparisons
    country_p = purchases.groupby('Country').agg(Orders=('InvoiceNo', 'nunique'), Revenue=('PurchRevenue', 'sum'))
    country_c = cancellations.groupby('Country').agg(Cancellations=('InvoiceNo', 'nunique'), LostRev=('CancRevenue', 'sum'))
    country_comp = pd.concat([country_p, country_c], axis=1).fillna(0)
    country_comp['TotalOrders'] = country_comp['Orders'] + country_comp['Cancellations']
    country_comp['CancelRate'] = country_comp['Cancellations'] / country_comp['TotalOrders']
    top_countries = country_comp.sort_values(by='TotalOrders', ascending=False).head(10)

    print("\n--- Country-Wise Transaction & Reversal Comparisons (Top 10) ---")
    print(top_countries[['TotalOrders', 'Orders', 'Cancellations', 'CancelRate', 'Revenue', 'LostRev']])

    # ------------------ VISUALIZATION 1: Product Cancellation Frequency ------------------
    plt.figure(figsize=(12, 6))
    top_canc_plot = top_cancelled_vol.head(8).copy()
    top_canc_plot['ShortDesc'] = top_canc_plot['Description'].apply(lambda x: (x[:26] + '..') if len(x) > 28 else x)
    
    x = np.arange(len(top_canc_plot))
    width = 0.38
    plt.bar(x - width/2, top_canc_plot['SoldQty'], width=width, label='Sold Quantity', color='#2b5c8f')
    plt.bar(x + width/2, top_canc_plot['CancQty'], width=width, label='Cancelled Quantity', color='#d9534f')
    plt.xticks(x, top_canc_plot['ShortDesc'], rotation=30, ha='right', fontsize=9)
    plt.ylabel('Total Units')
    plt.title('Product Cancellation Frequency: Sold vs Cancelled Units (Top Impacted SKUs)', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, '01_product_cancellation_frequency.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # ------------------ VISUALIZATION 2: Monthly Transaction Trends ------------------
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x_months = np.arange(len(monthly_trends))
    ax1.bar(x_months - 0.15, monthly_trends['Purchases'], width=0.3, label='Completed Purchases', color='#4a90e2')
    ax1.bar(x_months + 0.15, monthly_trends['Cancellations'], width=0.3, label='Cancellation Invoices', color='#e74c3c')
    ax1.set_xticks(x_months)
    ax1.set_xticklabels(monthly_trends.index, rotation=35, ha='right')
    ax1.set_ylabel('Invoice Count', fontsize=11)
    
    ax2 = ax1.twinx()
    ax2.plot(x_months, monthly_trends['CancellationRate'] * 100, color='#2c3e50', marker='o', linewidth=2.2, label='Cancellation Rate (%)')
    ax2.set_ylabel('Cancellation Rate (%)', fontsize=11, color='#2c3e50')
    ax2.grid(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Monthly Transaction Trends & Reversal Rate Dynamics (Dec 2010 - Dec 2011)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, '03_monthly_transaction_trends.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # ------------------ VISUALIZATION 3: Country-Wise Sales & Cancellation Comparisons ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    top_countries_sub = top_countries.head(8)
    
    ax1.barh(top_countries_sub.index[::-1], top_countries_sub['Revenue'][::-1] / 1000, color='#1abc9c')
    ax1.set_xlabel('Gross Revenue (k£)', fontsize=10)
    ax1.set_title('Top 8 Countries by Gross Revenue', fontsize=11, fontweight='bold')

    ax2.barh(top_countries_sub.index[::-1], top_countries_sub['CancelRate'][::-1] * 100, color='#e67e22')
    ax2.set_xlabel('Cancellation Rate (%)', fontsize=10)
    ax2.set_title('Cancellation Rate (%) Across Key Markets', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, '04_country_sales_comparison.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    return prod_metrics


# ======================================================================================
# STEP 4: MARKET BASKET ANALYSIS & PRODUCT ASSOCIATION NETWORK
# ======================================================================================
def conduct_market_basket_analysis(purchases):
    print_section("Step 4: Market Basket Analysis & Product Association Network")

    # Focus on top 200 items in UK to keep basket dense and association network readable
    uk_purchases = purchases[purchases['Country'] == 'United Kingdom']
    top_skus = uk_purchases['Description'].value_counts().head(200).index
    filtered_df = uk_purchases[uk_purchases['Description'].isin(top_skus)]

    print(f"Analyzing transaction baskets across {filtered_df['InvoiceNo'].nunique():,} UK orders...")
    basket = (filtered_df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    basket = (basket > 0).astype(bool)

    # Frequent itemsets using FP-Growth
    frequent_items = fpgrowth(basket, min_support=0.015, use_colnames=True)
    rules = association_rules(frequent_items, metric='lift', min_threshold=1.5)
    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

    print(f"Discovered {len(frequent_items):,} frequent itemsets and {len(rules):,} association rules.")
    print("\n--- Top 8 Association Rules by Lift ---")
    top_rules_disp = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(8).copy()
    top_rules_disp['antecedents'] = top_rules_disp['antecedents'].apply(lambda s: list(s)[0])
    top_rules_disp['consequents'] = top_rules_disp['consequents'].apply(lambda s: list(s)[0])
    print(top_rules_disp)

    # ------------------ VISUALIZATION: Product Association Network Graph ------------------
    plt.figure(figsize=(11, 8.5))
    G = nx.Graph()

    # Build network from top 25 strongest rules
    for _, row in rules.head(25).iterrows():
        ant = list(row['antecedents'])[0]
        con = list(row['consequents'])[0]
        # Shorten names for clean graph labels
        ant_short = ant[:18] + '..' if len(ant) > 20 else ant
        con_short = con[:18] + '..' if len(con) > 20 else con
        G.add_edge(ant_short, con_short, weight=round(row['lift'], 2), confidence=round(row['confidence'], 2))

    pos = nx.spring_layout(G, k=0.85, seed=42)
    degrees = dict(G.degree)
    node_sizes = [300 + degrees[node] * 280 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3498db', alpha=0.85, edgecolors='#2c3e50')
    nx.draw_networkx_edges(G, pos, width=1.8, alpha=0.6, edge_color='#7f8c8d')
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_family='sans-serif')

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='#c0392b')

    plt.title('Product Association Network Graph (Co-Purchasing Affinities & Lift)', fontsize=13, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    p_net = os.path.join(FIG_DIR, '05_product_association_network.png')
    plt.savefig(p_net, dpi=300)
    plt.close()
    print(f"Saved: {p_net}")

    return rules


# ======================================================================================
# STEP 5: CUSTOMER SEGMENTATION VIA K-MEANS CLUSTERING
# ======================================================================================
def conduct_customer_segmentation(purchases, cancellations):
    print_section("Step 5: Customer Segmentation via K-Means Clustering")

    purchases['TotalSpend'] = purchases['Quantity'] * purchases['UnitPrice']
    cancellations['Refund'] = cancellations['AbsQuantity'] * cancellations['UnitPrice']
    ref_date = purchases['InvoiceDate'].max() + pd.Timedelta(days=1)

    cust_rfm = purchases.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalSpend', 'sum'),
        TotalUnits=('Quantity', 'sum')
    ).reset_index()

    cust_canc = cancellations.groupby('CustomerID').agg(
        CancelCount=('InvoiceNo', 'nunique'),
        RefundTotal=('Refund', 'sum'),
        CancelledUnits=('AbsQuantity', 'sum')
    ).reset_index()

    cust_df = pd.merge(cust_rfm, cust_canc, on='CustomerID', how='left').fillna(0)
    cust_df['TotalInvoices'] = cust_df['Frequency'] + cust_df['CancelCount']
    cust_df['CancelRate'] = cust_df['CancelCount'] / cust_df['TotalInvoices']
    cust_df['UnitReversalRate'] = cust_df['CancelledUnits'] / (cust_df['TotalUnits'] + cust_df['CancelledUnits'])

    # Feature transformation for clustering (handling high skew)
    cust_feat = pd.DataFrame()
    cust_feat['Recency_log'] = np.log1p(cust_df['Recency'])
    cust_feat['Frequency_log'] = np.log1p(cust_df['Frequency'])
    cust_feat['Monetary_log'] = np.log1p(np.maximum(cust_df['Monetary'], 0))
    cust_feat['CancelRate'] = cust_df['CancelRate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cust_feat)

    # K-Means clustering (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cust_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Assign interpretable business segment labels based on cluster centroids
    cluster_means = cust_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary', 'CancelRate']].mean()
    print("\n--- K-Means Cluster Centroids (Raw Scale) ---")
    print(cluster_means)

    # Dynamic naming based on centroid traits
    segment_names = {}
    for c_id, row in cluster_means.iterrows():
        if row['CancelRate'] > 0.30:
            segment_names[c_id] = "High-Risk Reversal Segment"
        elif row['Frequency'] > 8 and row['Monetary'] > 3000:
            segment_names[c_id] = "VIP Wholesale Champions"
        elif row['Recency'] > 120 and row['Frequency'] < 3:
            segment_names[c_id] = "Dormant / One-Time Buyers"
        else:
            segment_names[c_id] = "Active Loyal Regulars"

    cust_df['Segment'] = cust_df['Cluster'].map(segment_names)

    segment_summary = cust_df.groupby('Segment').agg(
        Customers=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        AvgCancelRate=('CancelRate', 'mean'),
        TotalRefunds=('RefundTotal', 'sum')
    ).round(2)
    print("\n--- Customer Segments Profile Summary ---")
    print(segment_summary)

    # ------------------ VISUALIZATION: Customer Segmentation Plots ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'High-Risk Reversal Segment': '#e74c3c',
              'VIP Wholesale Champions': '#2ecc71',
              'Active Loyal Regulars': '#3498db',
              'Dormant / One-Time Buyers': '#95a5a6'}

    for seg, color in colors.items():
        subset = cust_df[cust_df['Segment'] == seg]
        ax1.scatter(subset['Frequency'], subset['Monetary'], label=seg, color=color, alpha=0.6, s=35)
        ax2.scatter(subset['Monetary'], subset['CancelRate'] * 100, label=seg, color=color, alpha=0.6, s=35)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Order Frequency (Log Scale)', fontsize=10)
    ax1.set_ylabel('Total Monetary Spend (Log £)', fontsize=10)
    ax1.set_title('Customer Segmentation: Frequency vs Monetary', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5)

    ax2.set_xscale('log')
    ax2.set_xlabel('Total Monetary Spend (Log £)', fontsize=10)
    ax2.set_ylabel('Cancellation Rate (%)', fontsize=10)
    ax2.set_title('Customer Cancellation Propensity vs Spend', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()
    p_seg = os.path.join(FIG_DIR, '06_customer_segmentation.png')
    plt.savefig(p_seg, dpi=300)
    plt.close()
    print(f"Saved: {p_seg}")

    # Also save customer purchase behaviour overview plot
    plt.figure(figsize=(10, 4.8))
    sns.histplot(cust_df['CancelRate'] * 100, bins=25, kde=True, color='#8e44ad')
    plt.xlabel('Customer-Level Cancellation Rate (%)', fontsize=10)
    plt.ylabel('Customer Count', fontsize=10)
    plt.title('Customer Purchase Behaviour: Distribution of Order Reversal Propensities', fontsize=12, fontweight='bold')
    plt.tight_layout()
    p_behav = os.path.join(FIG_DIR, '02_customer_purchase_behaviour.png')
    plt.savefig(p_behav, dpi=300)
    plt.close()
    print(f"Saved: {p_behav}")

    return cust_df


# ======================================================================================
# STEP 6: PREDICTIVE ANALYTICS - ORDER CANCELLATION RISK MODELING
# ======================================================================================
def conduct_predictive_modeling(purchases, cancellations):
    print_section("Step 6: Predictive Analytics - Order Cancellation Risk Modeling")

    print("Constructing leak-free invoice-level records strictly at order placement time...")

    # Aggregate purchases to invoice level
    purch_inv = purchases.groupby('InvoiceNo').agg(
        CustomerID=('CustomerID', 'first'),
        InvoiceDate=('InvoiceDate', 'min'),
        TotalAmount=('PurchRevenue', 'sum'),
        TotalQuantity=('Quantity', 'sum'),
        ItemCount=('StockCode', 'nunique'),
        AvgUnitPrice=('UnitPrice', 'mean'),
        MaxUnitPrice=('UnitPrice', 'max'),
        Country=('Country', 'first')
    ).reset_index()

    # Map cancellations: matched on CustomerID, StockCode, where cancellation date >= purchase date
    merged_items = pd.merge(
        purchases[['InvoiceNo', 'CustomerID', 'StockCode', 'Quantity', 'InvoiceDate']],
        cancellations[['CustomerID', 'StockCode', 'AbsQuantity', 'InvoiceDate']],
        on=['CustomerID', 'StockCode'],
        suffixes=('_purch', '_canc')
    )
    valid_canc = merged_items[merged_items['InvoiceDate_canc'] >= merged_items['InvoiceDate_purch']]
    cancelled_invoices = set(valid_canc['InvoiceNo'].unique())
    purch_inv['IsCancelled'] = purch_inv['InvoiceNo'].isin(cancelled_invoices).astype(int)

    # Feature engineering strictly at order placement
    purch_inv['Hour'] = purch_inv['InvoiceDate'].dt.hour
    purch_inv['DayOfWeek'] = purch_inv['InvoiceDate'].dt.dayofweek
    purch_inv['Month'] = purch_inv['InvoiceDate'].dt.month
    purch_inv['IsUK'] = (purch_inv['Country'] == 'United Kingdom').astype(int)

    # Customer cumulative prior purchase count up to this order (strictly past history, no lookahead)
    purch_inv = purch_inv.sort_values(by=['CustomerID', 'InvoiceDate']).reset_index(drop=True)
    purch_inv['PriorPurchases'] = purch_inv.groupby('CustomerID').cumcount()

    feature_cols = ['TotalAmount', 'TotalQuantity', 'ItemCount', 'AvgUnitPrice', 'MaxUnitPrice',
                    'Hour', 'DayOfWeek', 'Month', 'IsUK', 'PriorPurchases']
    X = purch_inv[feature_cols]
    y = purch_inv['IsCancelled']

    print(f"Total labeled order transactions: {len(purch_inv):,}")
    print(f"Class Distribution: {y.value_counts().to_dict()} ({y.mean()*100:.2f}% positive cancellation rate)")

    # Train / Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the 5 ML models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=120, max_depth=8, class_weight='balanced', n_jobs=-1),
        'SVM': CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced', random_state=42)),
        'MLP': MLPClassifier(random_state=42, hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True)
    }

    results = []
    roc_data = {}
    cm_data = {}

    print("\nTraining and evaluating models:")
    for name, model in models.items():
        print(f"  -> Fitting {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Probabilities for ROC curve
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test_scaled)
        else:
            y_prob = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    perf_df = pd.DataFrame(results).set_index('Model').round(4)
    print("\n--- Predictive ML Models Performance Summary ---")
    print(perf_df)

    # ------------------ VISUALIZATION: Confusion Matrices ------------------
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    for ax, (name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 10, 'weight': 'bold'})
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('Actual Label', fontsize=9)
        ax.set_xticklabels(['Completed', 'Cancelled'])
        ax.set_yticklabels(['Completed', 'Cancelled'])

    plt.suptitle('Confusion Matrices Across 5 Predictive ML Models', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    p_cm = os.path.join(FIG_DIR, '07_confusion_matrix.png')
    plt.savefig(p_cm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_cm}")

    # ------------------ VISUALIZATION: ROC Curves ------------------
    plt.figure(figsize=(9, 6.5))
    colors = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9', '#8e44ad']
    for (name, (fpr, tpr, auc_val)), col in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=col, linewidth=2.2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    p_roc = os.path.join(FIG_DIR, '08_roc_curve.png')
    plt.savefig(p_roc, dpi=300)
    plt.close()
    print(f"Saved: {p_roc}")

    # Feature Importance from Random Forest
    rf_model = models['Random Forest']
    feat_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Order Cancellation Risk: Feature Importances (Random Forest) ---")
    print(feat_imp.round(4))

    return perf_df, purch_inv


# ======================================================================================
# MAIN EXECUTION ENTRYPOINT
# ======================================================================================
def main():
    print("=" * 80)
    print(" UNDERSTANDING PURCHASE REVERSALS IN ONLINE RETAIL ")
    print(" THROUGH CUSTOMER BEHAVIOUR AND PREDICTIVE ANALYTICS ")
    print("=" * 80)

    # 1. Load and inspect
    df = load_and_inspect_data()

    # 2. Clean and preprocess
    cleaned_df, purchases, cancellations = clean_and_preprocess_data(df)

    # 3. Exploratory Data Analysis & Paradox Insights
    conduct_eda(purchases, cancellations)

    # 4. Market Basket Analysis & Association Network
    conduct_market_basket_analysis(purchases)

    # 5. Customer Segmentation with K-Means
    conduct_customer_segmentation(purchases, cancellations)

    # 6. Predictive Modeling (LR, DT, RF, SVM, MLP)
    conduct_predictive_modeling(purchases, cancellations)

    print_section("Execution Completed Successfully")
    print(f"All 8 visualization figures have been exported to: {FIG_DIR}")
    print("Project executed end-to-end within a single, modular Python script.\n")


if __name__ == '__main__':
    main()
