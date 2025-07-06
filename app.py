# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Hospitality Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/Hospitality_Synthetic_Survey.csv")

df = load_data()
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Go to", ["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"])

# ---------------------- 1. DATA VISUALIZATION ----------------------
if tabs == "Data Visualization":
    st.title("📊 Data Visualization")
    st.subheader("1. Age Group Distribution")
    st.bar_chart(df['Age Group'].value_counts())

    st.subheader("2. Income vs Comfort Spend Range")
    fig1 = px.histogram(df, x="Monthly Income", color="Comfort Spend Range", barmode='group')
    st.plotly_chart(fig1)

    st.subheader("3. Loyalty Tier Distribution")
    st.bar_chart(df['Loyalty Tier'].value_counts())

    st.subheader("4. Service Usage")
    service_counts = pd.Series(', '.join(df['Services Used']).split(', ')).value_counts()
    st.bar_chart(service_counts)

    st.subheader("5. Booking Channel vs Room Type")
    fig2 = px.histogram(df, x="Booking Channel", color="Preferred Room Type", barmode='group')
    st.plotly_chart(fig2)

    st.subheader("6. Overall Experience Rating")
    st.bar_chart(df['Overall Experience Rating'].value_counts())

    st.subheader("7. Correlation Heatmap (Numerical Ratings Only)")
    numeric_cols = df.select_dtypes(include=np.number)
    fig3, ax = plt.subplots()
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

    st.subheader("8. Willingness to Try Offers vs Loyalty Program")
    fig4 = px.histogram(df, x="Willing to Try Offers", color="In Loyalty Program", barmode='group')
    st.plotly_chart(fig4)

    st.subheader("9. Frequent Issues Faced by Guests")
    issues_counts = pd.Series(', '.join(df['Issues Faced']).split(', ')).value_counts()
    st.bar_chart(issues_counts.head(10))

    st.subheader("10. Service Time Preferences")
    st.bar_chart(df['Service Time'].value_counts())

# ---------------------- 2. CLASSIFICATION ----------------------
elif tabs == "Classification":
    st.title("🤖 Classification Models")
    df_clf = df.copy()
    df_clf = df_clf[df_clf['Willing to Try Offers'].isin(['Yes', 'No'])]
    label_cols = df_clf.select_dtypes(include='object').columns.drop(['Willing to Try Offers', 'Services Used', 'Preferred Bundles', 'Issues Faced', 'Nationality'])

    for col in label_cols:
        df_clf[col] = LabelEncoder().fit_transform(df_clf[col])

    X = df_clf.drop(columns=['Willing to Try Offers', 'Services Used', 'Preferred Bundles', 'Issues Faced', 'Nationality'])
    y = LabelEncoder().fit_transform(df_clf['Willing to Try Offers'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    st.subheader("Model Evaluation Table")
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        })

    st.dataframe(pd.DataFrame(results))

    st.subheader("Confusion Matrix")
    model_choice = st.selectbox("Choose model", list(models.keys()))
    cm = confusion_matrix(y_test, models[model_choice].predict(X_test))
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fig_roc = go.Figure()
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig_roc)

    st.subheader("Upload New Data for Prediction")
    uploaded_file = st.file_uploader("Upload CSV without target variable")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        new_data_encoded = new_data.copy()
        for col in label_cols:
            if col in new_data_encoded.columns:
                new_data_encoded[col] = LabelEncoder().fit_transform(new_data_encoded[col])
        prediction = models[model_choice].predict(new_data_encoded)
        new_data['Prediction'] = prediction
        st.dataframe(new_data)
        csv_out = new_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv_out, file_name="predictions.csv", mime="text/csv")

# ---------------------- 3. CLUSTERING ----------------------
elif tabs == "Clustering":
    st.title("🔍 Customer Clustering")
    cluster_df = df.copy()
    features = ['Spa Rating', 'Dining Rating', 'Room Service Rating', 'Cleanliness Rating', 'Overall Experience Rating']
    X = cluster_df[features]

    st.subheader("Elbow Method for Optimal Clusters")
    distortions = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(kmeans.inertia_)

    fig_elbow = px.line(x=list(K_range), y=distortions, labels={'x': 'k', 'y': 'Inertia'}, title="Elbow Curve")
    st.plotly_chart(fig_elbow)

    k = st.slider("Select Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    cluster_df['Cluster'] = kmeans.labels_

    st.subheader("Customer Segmentation Table")
    persona = cluster_df.groupby('Cluster')[features].mean().round(2)
    st.dataframe(persona)

    st.subheader("Download Cluster-Labelled Data")
    csv_cluster = cluster_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", data=csv_cluster, file_name="clustered_data.csv", mime="text/csv")

# ---------------------- 4. ASSOCIATION RULES ----------------------
elif tabs == "Association Rules":
    st.title("🔗 Association Rule Mining")
    ar_df = df.copy()
    transactions = [x.split(', ') for x in ar_df['Services Used'].dropna()]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    ar_data = pd.DataFrame(te_ary, columns=te.columns_)
    freq_items = apriori(ar_data, min_support=0.05, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
    rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ---------------------- 5. REGRESSION ----------------------
elif tabs == "Regression":
    st.title("📈 Regression Models")
    reg_df = df.copy()
    reg_df['Monthly Income'] = reg_df['Monthly Income'].map({
        '<25K': 20000, '25K–50K': 37500, '51K–75K': 62500,
        '76K–1L': 88000, '>1L': 120000
    })
    X = reg_df[['Spa Rating', 'Dining Rating', 'Room Service Rating', 'Cleanliness Rating', 'Overall Experience Rating']]
    y = reg_df['Monthly Income']

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    st.subheader("Model Performance Summary")
    reg_results = []
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        reg_results.append({
            "Model": name,
            "R2 Score": model.score(X, y),
            "MAE": np.mean(abs(y - y_pred)),
            "RMSE": np.sqrt(np.mean((y - y_pred)**2))
        })
    st.dataframe(pd.DataFrame(reg_results))
