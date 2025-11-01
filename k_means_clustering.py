from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成一个简单的二维数据集
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 训练 K-means 模型
model = KMeans(n_clusters=4)
model.fit(X)

# 预测聚类结果
y_kmeans = model.predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.show()