import numpy as np
from sklearn import metrics

y_pred = [0, 2, 1, 3, 9, 9, 8, 5, 8]
y_true = [0, 1, 2, 3, 2, 6,3 , 5, 9]

metrics.accuracy_score(y_true, y_pred)
metrics.accuracy_score(y_true, y_pred, normalize=False)  # 类似海明距离，每个类别求准确后，再求微平均

metrics.precision_score(y_true, y_pred, average='micro')  # 微平均，精确率
metrics.precision_score(y_true, y_pred, average='macro')  # 宏平均，精确率
metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')  # 指定特定分类标签的精确率

metrics.recall_score(y_true, y_pred, average='micro')
metrics.recall_score(y_true, y_pred, average='macro')

metrics.f1_score(y_true, y_pred, average='weighted')

metrics.confusion_matrix(y_true, y_pred)

target_names = ['class 0', 'class 1', 'class 2']
print(metrics.classification_report(y_true, y_pred, target_names=target_names))

# kappa score是一个介于(-1, 1)之间的数. score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
metrics.cohen_kappa_score(y_true, y_pred)

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
metrics.roc_auc_score(y_true, y_scores)

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


