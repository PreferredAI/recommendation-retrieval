import numpy as np
from .lsh import * 

def evaluate_topK(test_data, rec_list, k=10):
    inv_lst = np.unique(test_data[:, 0])

    intersection_cnt = {}         
    for i in range(test_data.shape[0]):
        id = int(i)
        if int(test_data[id, 1]) in rec_list[int(test_data[id, 0])]:
            intersection_cnt[test_data[id, 0]] = intersection_cnt.get(test_data[id, 0], 0) + 1
    invPairs_cnt = np.bincount(np.array(test_data[:, 0], dtype='int32'))

    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_lst:
        precision_acc += intersection_cnt.get(inv, 0) / float(k)
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

    return precision_acc / len(inv_lst), recall_acc / len(inv_lst)


def evalate_LSHTopK(test_vec, data, queries, hashFamily, num_table, num_bit, metric, topK):
    lsh_index = LSHIndex(hash_family = hashFamily(data.shape[1]), k = num_bit, L=num_table)
              
    #build index
    lsh_index.index(data.tolist())

    inv_lst = np.unique(test_vec[:, 0]).astype(int)

    lsh_rec_list = {}
    for inv in inv_lst:
        if lsh_rec_list.get(inv, None) is None:
            lsh_rec_list[inv] = list(map(itemgetter(0), lsh_index.query(queries[inv-1, :], metric, topK)))

    lsh_prec,lsh_recall = evaluate_topK(test_vec, lsh_rec_list, topK)
    touched = float(lsh_index.get_avg_touched())/data.shape[0]
    
    return lsh_prec, lsh_recall, touched