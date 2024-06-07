from scipy.stats import pearsonr

# داده‌های نمونه
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# محاسبه ضریب همبستگی پیرسون
r, _ = pearsonr(x, y)
print("Pearson correlation coefficient:", r)
