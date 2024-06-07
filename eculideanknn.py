from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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

# تعریف مدل KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# آموزش مدل
knn.fit(X_train, y_train)

# ارزیابی مدل
accuracy = knn.score(X_test, y_test)
print(f'Test accuracy: {accuracy}')
