import numpy as np

d = 6  # 向量维度
nb = 20  # index向量库的数据量
nq = 5  # 待检索query的数目
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
print("xb:", xb)
xb[:, 0] += np.arange(nb) / 10.  # index向量库的向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 10.  # 待检索的query向量
print("xb:", xb)
print("xq:", xq)
