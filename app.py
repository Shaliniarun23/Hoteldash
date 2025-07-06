import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px

# Streamlit App Setup
st.set_page_config(layout="wide")
st.title("🏨 Hospitality Analytics Dashboard")

# Load Excel File
@st.cache_data
def load_data():
    df = pd.read_excel("hotel_bookings_cleaned.xlsx")
    return df

df = load_data()

# Sidebar Tabs
tabs = st.tabs(["📊 Data Overview", "🔍 Classification", "📈 Regression", "🔍 Clustering", "🧾 Association Rules"])

# ======= TAB 1: DATA OVERVIEW =======
with tabs[0]:
    st.header("📊 Dataset Overview")
    st.write("### First Few Rows")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe(include='all'))

    st.write("### Null Values")
    st.write(df.isnull().sum())

# ======= TAB 2: CLASSIFICATION =======
with tabs[1]:
    st.header("🔍 Predict Booking Cancellation")
    df_cls = df.copy().dropna()
    
    if 'is_canceled' not in df_cls.columns:
        st.error("Missing target column: 'is_canceled'")
    else:
        X = df_cls.drop(columns=['is_canceled'])
        y = df_cls['is_canceled']
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
                "Precision": round(precision_score(y_test, y_pred)*100, 2),
                "Recall": round(recall_score(y_test, y_pred)*100, 2),
                "F1 Score": round(f1_score(y_test, y_pred)*100, 2)
            })

        st.subheader("📋 Model Performance Comparison")
        st.dataframe(pd.DataFrame(results))

        selected_model = st.selectbox("🔀 Show Confusion Matrix for:", list(models.keys()))
        if selected_model:
            model = models[selected_model]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay", "Cancel"], yticklabels=["Stay", "Cancel"])
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            st.pyplot(fig)

# ======= TAB 3: REGRESSION =======
with tabs[2]:
    st.header("📈 Predict Average Daily Rate (ADR)")
    df_reg = df.copy().dropna()
    
    if 'adr' not in df_reg.columns:
        st.error("Missing target column: 'adr'")
    else:
        X = df_reg.drop(columns=['adr'])
        y = df_reg['adr']
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Regression Results")
        st.write(f"**Mean Squared Error (MSE):** {round(mse, 2)}")
        st.write(f"**R-squared (R²):** {round(r2 * 100, 2)}%")

        st.subheader("📉 Residual Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True)
        plt.title("Distribution of Residuals")
        st.pyplot(fig)

# ======= TAB 4: CLUSTERING =======
with tabs[3]:
    st.header("🔍 Customer Segmentation")
    df_cluster = df.copy().dropna()
    cluster_data = df_cluster.select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    k = st.slider("Select Number of Clusters", min_value=2, max_value=5, value=4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_cluster['cluster'] = kmeans.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    df_cluster['PC1'] = components[:, 0]
    df_cluster['PC2'] = components[:, 1]

    st.subheader("📌 PCA Cluster Visualization")
    fig = px.scatter(df_cluster, x='PC1', y='PC2', color='cluster', title="Customer Segments")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧾 Cluster Summary")
    st.dataframe(df_cluster.groupby('cluster').mean().round(2))

# ======= TAB 5: ASSOCIATION RULES =======
with tabs[4]:
    st.header("🧾 Association Rule Mining")
    df_ar = df[['meal', 'deposit_type', 'customer_type', 'market_segment']].dropna().astype(str)
    records = df_ar.apply(lambda row: list(row), axis=1).tolist()

    te = TransactionEncoder()
    te_array = te.fit(records).transform(records)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    st.subheader("📋 Top Association Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    st.markdown("Use these rules to create bundles, upsell services, and tailor offers.")
