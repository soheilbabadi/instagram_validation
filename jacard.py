import pandas as pd

# بارگذاری داده‌ها از فایل اکسل
file_path = '/path/to/your/excel/file.xlsx'
df = pd.read_excel(file_path)

# استخراج مقادیر TP، FP و FN از فایل اکسل
TP = df['TP'].sum()
FP = df['FP'].sum()
FN = df['FN'].sum()

# محاسبه اندیس ژاکارد
Jaccard_Index = TP / (TP + FP + FN)

print(f'Jaccard Index: {Jaccard_Index}')
