# 🏨 Hospitality Analytics Dashboard

This Streamlit dashboard is built to demonstrate the use of machine learning and data analytics for the hospitality industry. It helps analyze customer behavior, segment users, predict preferences, and uncover associations between services — all using a synthetic yet realistic dataset.

---

## 📁 Project Structure

📦 hospitality_dashboard/
├── app.py
├── data/
│ └── Hospitality_Synthetic_Survey.csv
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 💡 Features

### 1. 📊 Data Visualization
- Demographic insights
- Booking behavior patterns
- Service usage trends
- Loyalty tier distribution
- Issue frequency and NPS scores

### 2. 🤖 Classification
- Algorithms: KNN, Decision Tree, Random Forest, Gradient Boosting
- Evaluation: Accuracy, Precision, Recall, F1-Score
- Confusion matrix and ROC curve visualization
- Upload your new data to get predictions
- Download prediction results

### 3. 🔍 Clustering
- K-means with adjustable cluster slider (2 to 10)
- Elbow method visualization
- Customer persona definitions per cluster
- Download clustered dataset

### 4. 🔗 Association Rule Mining
- Apriori algorithm applied to multi-select service data
- Filter top-10 service associations by confidence

### 5. 📈 Regression
- Models: Linear, Ridge, Lasso, Decision Tree Regressor
- Predict income or comfort-spend ranges
- View MAE, RMSE, and R² scores

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hospitality_dashboard.git
   cd hospitality_dashboard
