# coding:utf-8
import numpy as np


"""
输入数据result_list和list_len
result_list：列表类型 [user_id item_id pred_score real_rating]
list_len：NDCG预测的列表长度
"""
def get_ndcg(result_list, list_len):
    dcg_list = sorted(result_list, key = lambda x : x[2], reverse = True)
    temp_list = []
    for i in range(list_len):
        temp_list.append(dcg_list[i])
    # dcg_list = sorted(temp_list, key = lambda x:x[2], reverse = True)
    dcg_list = temp_list
    dcg = 0.0
    for i in range(len(dcg_list)):
        dcg = dcg + float(dcg_list[i][3]) / np.log(i + 2.0)
    idcg_list = sorted(temp_list, key = lambda x : x[3], reverse = True)
    idcg = 0.0
    for i in range(len(idcg_list)):
        idcg = idcg + float(idcg_list[i][3]) / np.log(i + 2.0)
    ndcg = dcg / idcg
    return ndcg

# 参考资料如下：
# github网址： https://github.com/wubinzzu/NeuRec/blob/master/evaluator/backend/python/metric.py
# 资料：http://www.bubuko.com/infodetail-3592693.html
# https://blog.csdn.net/lanyuelvyun/article/details/102585513
# https://zhuanlan.zhihu.com/p/84206752