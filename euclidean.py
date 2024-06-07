from scipy.spatial.distance import euclidean

# تعریف دو نقطه
A = [1, 2]
B = [4, 6]

# محاسبه فاصله اقلیدسی
distance = euclidean(A, B)
print(f'Euclidean Distance: {distance}')
