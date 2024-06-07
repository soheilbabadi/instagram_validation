# مقادیر TP، FP و FN
TP = 100
FP = 10
FN = 20

# محاسبه Precision و Recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)

# محاسبه F1-Score
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1_score}')
