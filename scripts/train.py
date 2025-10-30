# scripts/train.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Đường dẫn tuyệt đối, không phụ thuộc vào nơi chạy lệnh
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Mall_Customers.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

print(f"Đang đọc file: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"KHÔNG TÌM THẤY FILE: {DATA_PATH}\nVui lòng đặt file vào thư mục data/")

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Huấn luyện
model = KMeans(n_clusters=5, random_state=42, n_init=10)
model.fit(X_scaled)

# Tạo thư mục model nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)

# Lưu model và scaler
model_path = os.path.join(MODEL_DIR, 'model.joblib')
scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Mô hình đã lưu: {model_path}")
print(f"Scaler đã lưu: {scaler_path}")