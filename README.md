# Phân cụm khách hàng cho Marketing (Unsupervised Learning)

## Mục tiêu
- Phân cụm khách hàng theo hành vi mua sắm
- Tối ưu chiến lược marketing cá nhân hóa
- Tăng tỷ lệ phản hồi marketing lên 20%

## Dataset
- Nguồn: [Kaggle - Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- 200 khách hàng, 5 đặc trưng

## Phương pháp
- **Mô hình**: K-Means
- **Số cụm tối ưu**: 5 (Elbow Method + Silhouette Score = ~0.55)
- **Đặc trưng**: Age, Annual Income, Spending Score

## Kết quả
- 5 nhóm khách hàng rõ rệt
- Visualize 2D & 3D
- Mô hình lưu tại `model/model.joblib`

## Cách chạy
```bash
pip install -r requirements.txt
jupyter notebook notebooks/clustering.ipynb