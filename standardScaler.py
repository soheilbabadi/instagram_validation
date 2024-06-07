from sklearn.preprocessing import MinMaxScaler,StandardScaler

import numpy as np

# داده‌های نمونه
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# ایجاد یک StandardScaler
scaler = StandardScaler()

# استانداردسازی داده‌ها
standardized_data = scaler.fit_transform(data)

print("Standardized Data:")
print(standardized_data)


# داده‌های نمونه
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# ایجاد یک MinMaxScaler
scaler = MinMaxScaler()

# نرمال‌سازی داده‌ها
normalized_data = scaler.fit_transform(data)

print("Normalized Data:")
print(normalized_data)
