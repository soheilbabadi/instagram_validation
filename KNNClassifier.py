from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            most_common = Counter(nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

# بارگذاری داده‌ها
data = load_iris()
X = data.data
y = data.target

# تقسیم داده‌ها به بخش‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تعریف و آموزش مدل KNN
knn = KNNClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# پیش‌بینی و ارزیابی مدل
y_pred = knn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test accuracy: {accuracy}')
