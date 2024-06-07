import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class WeightedKNN:
    def __init__(self, n_neighbors=5, epsilon=1e-5):
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_distances = distances[nearest_indices]
            nearest_labels = self.y_train[nearest_indices]

            weights = 1 / (nearest_distances + self.epsilon)
            weighted_votes = np.zeros(np.max(self.y_train) + 1)
            for i, label in enumerate(nearest_labels):
                weighted_votes[label] += weights[i]

            predictions.append(np.argmax(weighted_votes))
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

# تعریف و آموزش مدل WeightedKNN
knn = WeightedKNN(n_neighbors=5)
knn.fit(X_train, y_train)

# پیش‌بینی و ارزیابی مدل
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')
