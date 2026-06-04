import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Đọc dữ liệu (Vì file csv nằm cùng thư mục src với main.py)
data = pd.read_csv("creditcard.csv")

# 2. Chia dữ liệu Input / Output
X = data.drop(columns=['Class'])
y = data['Class']

# 3. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Tạo và huấn luyện mô hình AI
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Dự đoán và in độ chính xác
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("--- Kết quả huấn luyện ---")
print("Accuracy:", acc)
print("--------------------------")

# 6. Tách dữ liệu để chuẩn bị vẽ biểu đồ
normal = data[data['Class'] == 0] # Giao dịch bình thường (Tím)
fraud = data[data['Class'] == 1]  # Giao dịch gian lận (Vàng)

# 7. Tiến hành vẽ biểu đồ tối ưu hiển thị
plt.figure(figsize=(10, 6))

# Vẽ nhóm Bình Thường TRƯỚC làm nền mờ (Tím)
plt.scatter(
    normal['Time'], 
    normal['Amount'], 
    color='#440154', 
    s=5, 
    alpha=0.1, 
    label='Normal Transactions'
)

# Vẽ nhóm Gian Lận SAU đè lên trên (Vàng có viền đen sắc nét)
plt.scatter(
    fraud['Time'], 
    fraud['Amount'], 
    color='#FDE725', 
    s=70, 
    alpha=1.0, 
    edgecolor='black', 
    label='🚨 Fraud Transactions'
)

# Thêm thắt tiêu đề, nhãn
plt.xlabel("Time (Seconds)", fontsize=12, fontweight='bold')
plt.ylabel("Amount (USD)", fontsize=12, fontweight='bold')
plt.title("Credit Card Fraud Detection - Optimized Visualization", fontsize=14, fontweight='bold')
plt.legend(markerscale=1.5, fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Hiển thị biểu đồ
plt.show()