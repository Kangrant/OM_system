
from text_analysis_tools.api.text_cluster.kmeans import KmeansClustering
from text_analysis_tools.api.text_cluster.dbscan import DbscanClustering


"""
文本聚类：
kmeans_cluster
dbscan_cluster
"""


def kmeans_cluster(data_path="./test_data/test_data_cluster.txt",
                   n_clusters=5):
    """
    KMEANS文本聚类
    :param data_path: 需要聚类的文本路径，每条文本存放一行
    :param n_clusters: 聚类个数
    :return: {'cluster_0': [0, 1, 2, 3, 4], 'cluster_1': [5, 6, 7, 8, 9]}   0,1,2....为文本的行号
    """
    Kmeans = KmeansClustering()
    result = Kmeans.kmeans(data_path, n_clusters=n_clusters)
    print("keams result: {}\n".format(result))


def dbscan_cluster(data_path="./test_data/test_data_cluster.txt",
                   eps=0.05, min_samples=3, fig=False):
    """
    基于DBSCAN进行文本聚类
    :param data_path: 文本路径，每行一条
    :param eps: DBSCA中半径参数
    :param min_samples: DBSCAN中半径eps内最小样本数目
    :param fig: 是否对降维后的样本进行画图显示，默认False
    :return: {'cluster_0': [0, 1, 2, 3, 4], 'cluster_1': [5, 6, 7, 8, 9]}   0,1,2....为文本的行号
    """
    dbscan = DbscanClustering()
    result = dbscan.dbscan(corpus_path=data_path, eps=eps, min_samples=min_samples, fig=fig)
    print("dbscan result: {}\n".format(result))

if __name__ == '__main__':
    data_path = 'cluster_data.txt'
    # kmeans_cluster(data_path,n_clusters=4)
    data = ['位置','地理位置','位置']
    dbscan_cluster(data,eps=0.005,min_samples=1)