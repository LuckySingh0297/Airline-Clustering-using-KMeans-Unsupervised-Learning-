# Airline-Clustering-using-KMeans-Unsupervised-Learning-
Built a KMeans clustering model to segment airlines based on passenger traffic and flight frequency. Performed EDA, outlier removal, and cluster evaluation using Silhouette Score, Calinski-Harabasz, and Davies-Bouldin metrics. Identified optimal clusters and generated actionable insights on airline operations.
# 🚀 Airline Clustering using KMeans

## 📌 Overview
This project applies K-Means Clustering (Unsupervised Learning) to analyze airline traffic data and group airlines based on operational scale using passenger count and number of flights.

The main goal is to identify patterns in airline operations and segment them into meaningful clusters.

---

## 📂 Dataset
- Source: Air Traffic Passenger Statistics
- Features used:
  - Operating Airline
  - Passenger Count
  - Flights held (derived)

---

## 🧠 Problem Statement
Airlines operate at different scales — some handle massive traffic while others operate at a smaller level.

This project aims to:
- Segment airlines into clusters
- Identify high vs low traffic airlines
- Detect outliers affecting clustering

---

## ⚙️ Tech Stack
- Python
- Pandas
- Matplotlib
- Scikit-learn
- SQLAlchemy
- Kneedle

---

## 🔍 Project Workflow

### 1. Data Collection & Storage
- Loaded dataset using Pandas
- Stored data into MySQL database

### 2. Data Preprocessing
- Checked missing values and duplicates
- Selected relevant features
- Aggregated:
  - Total passengers per airline
  - Total flights per airline

### 3. Outlier Detection
- Used scatter plot visualization
- Identified extreme outliers:
  - United Airlines
  - United Airlines (Pre-2013)
- Removed outliers to improve clustering performance

### 4. Finding Optimal Clusters

Elbow Method:
- Plotted Inertia vs Number of Clusters
- No clear elbow detected (gradual decrease)

Silhouette Score:
- Evaluated K from 2 to 8
- Best K = 2

---

## 🤖 Model Building
Applied KMeans clustering:

KMeans(n_clusters=2)

---

## 📈 Model Evaluation
- Silhouette Score → Measures cluster separation
- Calinski-Harabasz Index → Higher is better
- Davies-Bouldin Index → Lower is better

---

## 📊 Results
The model segmented airlines into 2 clusters:

- Cluster 0: Low traffic airlines (fewer flights & passengers)
- Cluster 1: High traffic airlines (large-scale operations)

---

## 💾 Outputs
- Clustered dataset: Air.csv
- Trained model: Clust_.pkl

---

## 📌 Key Insights
- Clear separation between small and large airlines
- Outliers significantly affect clustering
- KMeans works well after preprocessing

---

## ⚠️ Limitations
- No feature scaling applied
- Limited features used
- GEO Region not used

---

## 🔥 Future Improvements
- Apply StandardScaler
- Use more features
- Try DBSCAN and Hierarchical Clustering
- Improve visualization using PCA
