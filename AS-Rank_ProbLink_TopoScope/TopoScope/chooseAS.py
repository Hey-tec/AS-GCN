from basicAtts import BasicAtts
import math
from sklearn.cluster import KMeans

ba = BasicAtts('./File/asrel.txt')
fullVP = set(open('./Infer_File/fullVP.txt', 'r').read().split('\n'))

X = []
asn = []
for node in ba.graph.nodes():
    tmp = []
    asn.append(node)
    tmp.append(math.log10(ba.graph.degree(node)))
    tmp.append(ba.distance[node])
    tmp.append(ba.getHierarchy(node))
    tmp.extend([math.log10(len(ba.customer[node])+1), math.log10(len(ba.peer[node])+1), math.log10(len(ba.provider[node])+1)])
    X.append(tmp)

cluster = 10
estimator = KMeans(n_clusters=cluster)                      # 生成10个簇
estimator.fit(X)                                            # 计算K-means聚类
pred = estimator.labels_                                    # 每个点的标签    0 ~ 9
res = [set() for _ in range(cluster)]
for i in range(len(pred)):
    for j in range(cluster):
        if pred[i] == j:
            res[j].add(asn[i])                              # 根据标签，分为是个ASN集合
fout = open('./Infer_File/chooseAS.txt', 'w')
for i in range(cluster):
    print(len(res[i]), len(fullVP), len(res[i] & fullVP) / len(res[i]))
    if len(res[i] & fullVP) / len(res[i]) > 0.06:
        fout.write(str(res[i]))
fout.close()