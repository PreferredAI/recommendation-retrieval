import numpy as np
from .lsh import * 

def evaluate_topK(test_data, data, queries, topK=10):
    inv_list = np.unique(test_data[:, 0]).astype(int)
    inv_list = [x for x in inv_list if x <= queries.shape[0]]
    rec_list = {}
    for inv in inv_list:
        if rec_list.get(inv, None) is None:
            rec_list[inv] = np.argsort(np.dot(data, queries[int(inv-1), :]))[-topK:]
    
    intersection_cnt = {}         
    for i in range(test_data.shape[0]):
        id = int(i)
        if int(test_data[id, 0]) in inv_list:
            if int(test_data[id, 1]) in rec_list[int(test_data[id, 0])]:
                intersection_cnt[test_data[id, 0]] = intersection_cnt.get(test_data[id, 0], 0) + 1
                
    invPairs_cnt = np.bincount(np.array(test_data[:, 0], dtype='int32'))

    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_list:
        precision_acc += intersection_cnt.get(inv, 0) / float(topK)
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

    return precision_acc / len(inv_list), recall_acc / len(inv_list)


def evaluate_LSHTopK(test_data, data, queries, lsh_index, metric, topK):          
    #build index
    lsh_index.index(data.tolist())

    inv_list = np.unique(test_data[:, 0]).astype(int)
    inv_list = [x for x in inv_list if x <= queries.shape[0]]

    rec_list = {}
    for inv in inv_list:
        if rec_list.get(inv, None) is None:
            rec_list[inv] = list(map(itemgetter(0), lsh_index.query(queries[inv-1, :], metric, topK)))
            
    intersection_cnt = {}         
    for i in range(test_data.shape[0]):
        id = int(i)
        if int(test_data[id, 0]) in inv_list:
            if int(test_data[id, 1]) in rec_list[int(test_data[id, 0])]:
                intersection_cnt[test_data[id, 0]] = intersection_cnt.get(test_data[id, 0], 0) + 1
                
    invPairs_cnt = np.bincount(np.array(test_data[:, 0], dtype='int32'))

    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_list:
        precision_acc += intersection_cnt.get(inv, 0) / float(topK)
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
    
    touched = float(lsh_index.get_avg_touched())/data.shape[0]
    
    return precision_acc / len(inv_list), recall_acc / len(inv_list), touched