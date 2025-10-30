# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# === CẤU HÌNH TRANG ===
st.set_page_config(page_title="Customer Clustering", layout="wide")
st.title("Customer Clustering Dashboard")
st.markdown("### Phân cụm khách hàng bằng K-Means (k=5)")

# === ĐƯỜNG DẪN AN TOÀN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Mall_Customers.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.joblib')

# === KIỂM TRA FILE ===
if not os.path.exists(DATA_PATH):
    st.error(f"Không tìm thấy file: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Không tìm thấy model: {MODEL_PATH}")
    st.stop()

# === ĐỌC DỮ LIỆU ===
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df

df = load_data()

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# === DỰ ĐOÁN CỤM (nếu chưa có) ===
if 'Cluster' not in df.columns:
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    X_scaled = scaler.transform(X)
    df['Cluster'] = model.predict(X_scaled)

# === SIDEBAR: LỌC DỮ LIỆU ===
st.sidebar.header("Bộ lọc")
selected_clusters = st.sidebar.multiselect(
    "Chọn cụm để hiển thị:",
    options=sorted(df['Cluster'].unique()),
    default=sorted(df['Cluster'].unique())
)

df_filtered = df[df['Cluster'].isin(selected_clusters)]

# === HIỂN THỊ THỐNG KÊ ===
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tổng khách hàng", len(df))
with col2:
    st.metric("Số cụm", df['Cluster'].nunique())
with col3:
    st.metric("Silhouette Score", "~0.55")

# === VẼ BIỂU ĐỒ 2D ===
st.subheader("Phân cụm: Thu nhập vs Điểm chi tiêu")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df_filtered['Annual Income (k$)'],
    df_filtered['Spending Score (1-100)'],
    c=df_filtered['Cluster'],
    cmap='viridis',
    s=100,
    edgecolors='k',
    alpha=0.8
)
ax.set_xlabel('Thu nhập hàng năm (k$)')
ax.set_ylabel('Điểm chi tiêu (1-100)')
ax.set_title('Phân cụm khách hàng (K-Means)')
plt.colorbar(scatter, label='Cụm', ax=ax)
st.pyplot(fig)

# === VẼ BIỂU ĐỒ 3D ===
st.subheader("3D: Tuổi vs Thu nhập vs Điểm chi tiêu")
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
scatter3d = ax3d.scatter(
    df_filtered['Age'],
    df_filtered['Annual Income (k$)'],
    df_filtered['Spending Score (1-100)'],
    c=df_filtered['Cluster'],
    cmap='viridis',
    s=60,
    edgecolors='k'
)
ax3d.set_xlabel('Tuổi')
ax3d.set_ylabel('Thu nhập (k$)')
ax3d.set_zlabel('Điểm chi tiêu')
plt.title('3D Visualization')
legend = plt.legend(*scatter3d.legend_elements(), title="Cụm", loc="upper right")
ax3d.add_artist(legend)
st.pyplot(fig3d)

# === PHÂN TÍCH TỪNG CỤM ===
st.subheader("Trung bình đặc trưng theo cụm")
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
st.dataframe(cluster_summary, use_container_width=True)

# === CHART BAR ===
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
cluster_summary.plot(kind='bar', ax=ax_bar)
ax_bar.set_title('Đặc trưng trung bình theo cụm')
ax_bar.set_ylabel('Giá trị trung bình')
ax_bar.set_xlabel('Cụm')
plt.xticks(rotation=0)
st.pyplot(fig_bar)

# === FOOTER ===
st.caption("Dự án phân cụm khách hàng - Local Python, không dùng AWS")