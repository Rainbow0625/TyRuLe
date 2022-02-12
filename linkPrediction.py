import numpy as np
import sys
import time
from scipy import sparse
import gc
import util as util


def get_onehot_matrix(p, fact_dic, ent_size, rel_size):
    # sparse matrix
    if p >= rel_size:
        inverse_flag = True
        pfacts = fact_dic.get(p - rel_size)
    else:
        inverse_flag = False
        pfacts = fact_dic.get(p)
    pmatrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    if inverse_flag:
        for f in pfacts:
            pmatrix[f[1], f[0]] = 1
    else:
        for f in pfacts:
            pmatrix[f[0], f[1]] = 1
    return pmatrix


def get_onehot_matrix_filter_by_type(p, fact_dic, ent_size, rel_size, h_t, t_t, type_entities_dic):
    h_e = set()
    t_e = set()
    for t in h_t:
        h_e.update(type_entities_dic.get(t))
    for t in t_t:
        t_e.update(type_entities_dic.get(t))
    # sparse matrix
    if p >= rel_size:
        inverse_flag = True
        pfacts = fact_dic.get(p - rel_size)
    else:
        inverse_flag = False
        pfacts = fact_dic.get(p)
    pmatrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    if inverse_flag:
        for f in pfacts:
            # len==0: no constrain
            if len(h_e) == 0:
                if len(t_e) == 0:
                    pmatrix[f[1], f[0]] = 1
                else:
                    if f[0] in t_e:
                        pmatrix[f[1], f[0]] = 1
            else:
                if len(t_e) == 0:
                    if f[1] in h_e:
                        pmatrix[f[1], f[0]] = 1
                else:
                    if f[1] in h_e and f[0] in t_e:
                        pmatrix[f[1], f[0]] = 1
    else:
        for f in pfacts:
            # len==0: no constrain
            if len(h_e) == 0:
                if len(t_e) == 0:
                    pmatrix[f[0], f[1]] = 1
                else:
                    if f[1] in t_e:
                        pmatrix[f[0], f[1]] = 1
            else:
                if len(t_e) == 0:
                    if f[0] in h_e:
                        pmatrix[f[0], f[1]] = 1
                else:
                    if f[0] in h_e and f[1] in t_e:
                        pmatrix[f[0], f[1]] = 1
    return pmatrix


def filter_fb15k237(test_facts, pt):
    # Delete the other pre in test case.
    fetch_list = list(np.where(test_facts[:, 2] == pt)[0])
    test_facts = test_facts[fetch_list]
    return test_facts


# 1
def predict(lp_save_path, pt, fact_dic, rules, ent_size, rel_size):
    # rules: [index, flag={1:Rule, 2:Quality Rule}, degree=[SC, HC]]
    predict_fact_num = 0
    # predict_Qfact_num = 0

    # predict_matrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule = {}  # key: fact  value: rule_SC list
    i = 1
    for rule in rules:
        i += 1
        index = rule[0]
        degree = rule[2]  # Todo: why not use degree?

        pmatrix = get_onehot_matrix(index[0], fact_dic, ent_size, rel_size)
        for i in range(1, len(index)):
            pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size, rel_size))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule:
                temp_SC_list = predict_facts_by_rule.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(degree[0])
            predict_facts_by_rule[key] = temp_SC_list

        # predict_matrix += pmatrix
        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (i, len(rules)))
    sys.stdout.flush()

    # return predict_matrix, predict_fact_num, predict_facts_by_rule
    return None, predict_fact_num, predict_facts_by_rule


# 1.2 THIS
def predict_SC_type(lp_save_path, pt, fact_dic, rules, ent_size, rel_size):
    # rules: [index, flag={1:Rule, 2:Quality Rule}, degree=[SC, HC]]
    predict_fact_num = 0
    # predict_Qfact_num = 0

    # predict_matrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule = {}  # key: fact  value: rule_SC list
    j = 1
    for rule in rules:
        j += 1
        index = rule[0]
        # full process of RL and LP
        SC_final = 0.0
        if len(rule) == 5:
            degree = rule[2]
            if int(rule[3]) != 0:
                degree_type = rule[4]
                if degree[0] >= degree_type[0]:
                    SC_final = degree[0]
                else:
                    # if the SC_type>SC, the SC_type works.
                    SC_final = degree_type[0]
            else:
                SC_final = degree[0]
        # part process of LP, read rules from files.
        if len(rule) == 3:
            degree = rule[1]
            degree_type = rule[2]
            if degree[0] >= degree_type[0]:
                SC_final = degree[0]
            else:
                # if the SC_type>SC, the SC_type works.
                SC_final = degree_type[0]
        if len(rule) == 2:
            degree = rule[1]
            SC_final = degree[0]

        pmatrix = get_onehot_matrix(index[0], fact_dic, ent_size, rel_size)
        for i in range(1, len(index)):
            pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size, rel_size))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule:
                temp_SC_list = predict_facts_by_rule.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(SC_final)
            predict_facts_by_rule[key] = temp_SC_list

        # predict_matrix += pmatrix
        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (j, len(rules)))
    sys.stdout.flush()

    # return predict_matrix, predict_fact_num, predict_facts_by_rule
    return None, predict_fact_num, predict_facts_by_rule


# 1.3 only type rules
def predict_SC_type_only(lp_save_path, pt, fact_dic, rules, ent_size, rel_size):
    # rules: [index, flag={1:Rule, 2:Quality Rule}, degree=[SC, HC]]
    predict_fact_num = 0
    # predict_Qfact_num = 0

    # predict_matrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule = {}  # key: fact  value: rule_SC list
    j = 1
    for rule in rules:
        j += 1
        index = rule[0]
        # full process of RL and LP
        SC_final = 0.0
        if len(rule) == 5:
            degree = rule[2]
            if int(rule[3]) != 0:
                degree_type = rule[4]
                if degree[0] >= degree_type[0]:
                    SC_final = degree[0]
                else:
                    # if the SC_type>SC, the SC_type works.
                    SC_final = degree_type[0]
            else:
                SC_final = degree[0]
        # part process of LP, read rules from files.
        if len(rule) == 3:
            degree = rule[1]
            degree_type = rule[2]
            # if degree[0] >= degree_type[0]:
            #     SC_final = degree[0]
            # else:
            #     # if the SC_type>SC, the SC_type works.
            #     SC_final = degree_type[0]
            SC_final = degree_type[0]
        if len(rule) == 2:
            degree = rule[1]
            SC_final = degree[0]
            continue

        pmatrix = get_onehot_matrix(index[0], fact_dic, ent_size, rel_size)
        for i in range(1, len(index)):
            pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size, rel_size))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule:
                temp_SC_list = predict_facts_by_rule.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(SC_final)
            predict_facts_by_rule[key] = temp_SC_list

        # predict_matrix += pmatrix
        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (j, len(rules)))
    sys.stdout.flush()

    # return predict_matrix, predict_fact_num, predict_facts_by_rule
    return None, predict_fact_num, predict_facts_by_rule


# 2
def predictByType_Zn_Y(lp_save_path, pt, fact_dic, rulesWithType, ent_size, rel_size, BENCHMARK):
    # rulesWithType: [index(p1, p2, ...), arg_type(x, y, z1, z2 ...) ]
    predict_fact_num = 0
    # predict_Qfact_num = 0
    # Get type2entity key:type_index value:entities_index
    type_entities_dic = util.get_type_entities(filename="./benchmarks/{0}/".format(BENCHMARK))

    '''predict right'''
    # predict_matrix_right = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule_right = {}  # key: fact  value: rule_SC list
    i = 1
    for rule in rulesWithType:
        i += 1
        index = rule[0]
        degree = rule[1]  # degree = [SC, HC]
        arg_type = rule[2]

        # 2. Only one change!!!  x z1 y
        h_t = []
        t_t = arg_type[1]
        pmatrix = get_onehot_matrix_filter_by_type(index[0], fact_dic, ent_size, rel_size, h_t, t_t, type_entities_dic)
        for i in range(1, len(index)):
            # Todo check, and change arg_type's order!
            h_t = arg_type[i]
            t_t = arg_type[i + 1]
            pmatrix = pmatrix.dot(get_onehot_matrix_filter_by_type(index[i], fact_dic, ent_size, rel_size,
                                                                   h_t, t_t, type_entities_dic))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        # predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule_right:
                temp_SC_list = predict_facts_by_rule_right.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(degree[0])
            predict_facts_by_rule_right[key] = temp_SC_list

        # predict_matrix_right += pmatrix

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (i, len(rulesWithType)))
    sys.stdout.flush()

    '''predict left'''
    # predict_matrix_left = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule_left = {}  # key: fact  value: rule_SC list
    i = 1
    for rule in rulesWithType:
        i += 1
        index = rule[0]
        degree = rule[1]  # degree = [SC, HC]
        arg_type = rule[2]

        # 2. Only one change!!!   x z1 z2 y
        h_t = arg_type[0]
        t_t = arg_type[1]
        pmatrix = get_onehot_matrix_filter_by_type(index[0], fact_dic, ent_size, rel_size, h_t, t_t, type_entities_dic)
        for i in range(1, len(index)):
            if i == len(index) - 1:
                h_t = arg_type[i]
                t_t = []
                pmatrix = pmatrix.dot(get_onehot_matrix_filter_by_type(index[i], fact_dic, ent_size, rel_size,
                                                                       h_t, t_t, type_entities_dic))
            else:
                h_t = arg_type[i]
                t_t = arg_type[i + 1]
                pmatrix = pmatrix.dot(get_onehot_matrix_filter_by_type(index[i], fact_dic, ent_size, rel_size,
                                                                       h_t, t_t, type_entities_dic))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        # predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule_left:
                temp_SC_list = predict_facts_by_rule_left.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(degree[0])
            predict_facts_by_rule_left[key] = temp_SC_list

        # predict_matrix_left += pmatrix

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (i, len(rulesWithType)))
    sys.stdout.flush()

    return None, None, \
           predict_fact_num, \
           predict_facts_by_rule_right, predict_facts_by_rule_left


# 3
def predictByType_Y(lp_save_path, pt, fact_dic, rulesWithType, ent_size, rel_size, BENCHMARK):
    # rulesWithType: [index(p1, p2, ...), arg_type(x, y, z1, z2 ...) ]
    predict_fact_num = 0
    # predict_Qfact_num = 0
    # Get type2entity key:type_index value:entities_index
    type_entities_dic = util.get_type_entities(filename="./benchmarks/{0}/".format(BENCHMARK))

    '''predict right'''
    # predict_matrix_right = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule_right = {}  # key: fact  value: rule_SC list
    i = 1
    for rule in rulesWithType:
        i += 1
        index = rule[0]
        degree = rule[1]  # degree = [SC, HC]
        arg_type = rule[2]

        pmatrix = get_onehot_matrix(index[0], fact_dic, ent_size, rel_size)
        for i in range(1, len(index)):
            if i == len(index) - 1:
                h_t = []
                t_t = arg_type[i + 1]
                pmatrix = pmatrix.dot(get_onehot_matrix_filter_by_type(index[i], fact_dic, ent_size, rel_size,
                                                                       h_t, t_t, type_entities_dic))
            else:
                pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size, rel_size))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule_right:
                temp_SC_list = predict_facts_by_rule_right.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(degree[0])
            predict_facts_by_rule_right[key] = temp_SC_list

        # predict_matrix_right += pmatrix

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (i, len(rulesWithType)))
    sys.stdout.flush()

    '''predict left'''
    # predict_matrix_left = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule_left = {}  # key: fact  value: rule_SC list
    i = 1
    for rule in rulesWithType:
        i += 1
        index = rule[0]
        degree = rule[1]  # degree = [SC, HC]
        arg_type = rule[2]

        h_t = arg_type[0]
        t_t = []
        pmatrix = get_onehot_matrix_filter_by_type(index[0], fact_dic, ent_size, rel_size, h_t, t_t, type_entities_dic)
        for i in range(1, len(index)):
            pmatrix = pmatrix.dot(get_onehot_matrix(index[i], fact_dic, ent_size, rel_size))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule_left:
                temp_SC_list = predict_facts_by_rule_left.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(degree[0])
            predict_facts_by_rule_left[key] = temp_SC_list

        # predict_matrix_left += pmatrix

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (i, len(rulesWithType)))
    sys.stdout.flush()

    return None, None, \
           predict_fact_num, \
           predict_facts_by_rule_right, predict_facts_by_rule_left


# 4
def predictByType(lp_save_path, pt, fact_dic, rulesWithType, ent_size, rel_size, BENCHMARK):
    # rulesWithType: [index(p1, p2, ...), arg_type(x, y, z1, z2 ...) ]
    predict_fact_num = 0
    # predict_Qfact_num = 0
    # predict_matrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int32)
    predict_facts_by_rule = {}  # key: fact  value: rule_SC list
    # Get type2entity key:type_index value:entities_index
    type_entities_dic = util.get_type_entities(filename="./benchmarks/{0}/".format(BENCHMARK))
    i = 1
    for rule in rulesWithType:
        i += 1
        index = rule[0]
        degree = rule[1]  # degree = [SC, HC]
        arg_type = rule[2]

        # 2. Only one change!!!
        # h_t = []
        # 4. Only one change!!!
        h_t = arg_type[0]
        t_t = arg_type[1]
        pmatrix = get_onehot_matrix_filter_by_type(index[0], fact_dic, ent_size, rel_size, h_t, t_t, type_entities_dic)
        for i in range(1, len(index)):
            # Todo check, and change arg_type's order!
            h_t = arg_type[i]
            t_t = arg_type[i + 1]
            pmatrix = pmatrix.dot(get_onehot_matrix_filter_by_type(index[i], fact_dic, ent_size, rel_size,
                                                                   h_t, t_t, type_entities_dic))
        pmatrix = pmatrix.todok()
        # Predict num of facts:
        predict_fact_num += len(pmatrix)

        for key in pmatrix.keys():
            if key in predict_facts_by_rule:
                temp_SC_list = predict_facts_by_rule.get(key)
            else:
                temp_SC_list = []
            temp_SC_list.append(degree[0])
            predict_facts_by_rule[key] = temp_SC_list

        # predict_matrix += pmatrix

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()
    sys.stdout.write('\rProgress: %d - %d ' % (i, len(rulesWithType)))
    sys.stdout.flush()

    return None, predict_fact_num, predict_facts_by_rule


# 1
def testByNoisyORPath(lp_save_path, pt, predict_matrix, test_facts):
    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    # Filter the head entity for 'predict_matrix'.
    test_head_entity = test_facts[:, 0]
    test_entity_dic = {}
    for key in predict_matrix.keys():
        if list(key)[0] in test_head_entity:
            # test_entity_dic -- key: test_head_entity  value:[[tail_entity, count], ...]
            if list(key)[0] in test_entity_dic.keys():
                temp_list = test_entity_dic.get(list(key)[0])
            else:
                temp_list = []
            temp_list.append([list(key)[1], predict_matrix[key]])
            test_entity_dic[list(key)[0]] = temp_list
    if len(test_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".format(len(test_head_entity), len(test_entity_dic)))

    # Todo: Rank the predicted facts by rule number! Not CD!
    for head in test_entity_dic.keys():
        test_entity_dic.get(head).sort(key=lambda x: x[1], reverse=True)
    # Calculate the MRR and Hit@10.
    test_result_for_tail_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_entity_dic.get(test_fact[0]) is not None:
            tail_list = [row[0] for row in test_entity_dic.get(test_fact[0])]
            test_number += 1
        else:
            tail_list = []
        if test_fact[1] in tail_list:
            top = tail_list.index(test_fact[1]) + 1
            right_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        print(right_Hits_10)
        print("test_number: {0}".format(str(test_number)))
        print("test_facts: {0}".format(str(len(test_facts))))
        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10 = right_Hits_10 / float(test_number)
        right_Hits_3 = right_Hits_3 / float(test_number)
        right_Hits_1 = right_Hits_1 / float(test_number)
        right_MRR = right_MRR / float(test_number)
    else:
        right_Hits_10 = 0.0
        right_Hits_3 = 0.0
        right_Hits_1 = 0.0
        right_MRR = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0
    # Save the results in file.
    with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
        for item in test_result_for_tail_prediction:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
        f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
        f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
        f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))

        f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
        f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
        f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
        f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    # Filter the tail entity for 'predict_matrix'.
    test_tail_entity = test_facts[:, 1]
    test_entity_dic = {}
    for key in predict_matrix.keys():
        if list(key)[1] in test_tail_entity:
            # test_entity_dic -- key: test_tail_entity  value:[[head_entity, count], ...]
            if list(key)[1] in test_entity_dic.keys():
                temp_list = test_entity_dic.get(list(key)[1])
            else:
                temp_list = []
            temp_list.append([list(key)[0], predict_matrix[key]])
            test_entity_dic[list(key)[1]] = temp_list
    if len(test_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".format(len(test_tail_entity), len(test_entity_dic)))

    # Rank the predicted facts by rule number! Not CD!
    for tail in test_entity_dic.keys():
        test_entity_dic.get(tail).sort(key=lambda x: x[1], reverse=True)

    # Calculate the MRR and Hit@10.
    test_result_for_head_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_entity_dic.get(test_fact[1]) is not None:
            head_list = [row[0] for row in test_entity_dic.get(test_fact[1])]
            test_number += 1
        else:
            head_list = []
        if test_fact[0] in head_list:
            top = head_list.index(test_fact[0]) + 1
            left_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10 = left_Hits_10 / float(test_number)
        left_Hits_3 = left_Hits_3 / float(test_number)
        left_Hits_1 = left_Hits_1 / float(test_number)
        left_MRR = left_MRR / float(test_number)
    else:
        left_Hits_10 = 0.0
        left_Hits_3 = 0.0
        left_Hits_1 = 0.0
        left_MRR = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

    # Save the results in file.
    with open("{0}test_Pt_{1}_left.txt".format(lp_save_path, str(pt)), 'w') as f:
        for item in test_result_for_head_prediction:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, Hits@10: %f\n" % (pt, left_Hits_10))
        f.write("For Pt: %d, Hits@3: %f\n" % (pt, left_Hits_3))
        f.write("For Pt: %d, Hits@1: %f\n" % (pt, left_Hits_1))
        f.write("For Pt: %d, MRR: %f\n" % (pt, left_MRR))

        f.write("For Pt: %d, Hits@10_all: %f\n" % (pt, left_Hits_10_all))
        f.write("For Pt: %d, Hits@3_all: %f\n" % (pt, left_Hits_3_all))
        f.write("For Pt: %d, Hits@1_all: %f\n" % (pt, left_Hits_1_all))
        f.write("For Pt: %d, MRR_all: %f\n" % (pt, left_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, left_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, left_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, left_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, left_MRR))

    print("For Pt: %d, Hits@10_all: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3_all: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1_all: %f" % (pt, left_Hits_1_all))
    print("For Pt: %d, MRR_all: %f" % (pt, left_MRR_all))

    Hit_10 = (left_Hits_10 + right_Hits_10) / 2
    Hit_3 = (left_Hits_3 + right_Hits_3) / 2
    Hit_1 = (left_Hits_1 + right_Hits_1) / 2
    MRR = (left_MRR + right_MRR) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    # return MRR, Hit_1, Hit_3, Hit_10
    return MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all


def testByNoisyORPath_2(lp_save_path, pt, tail_predict_matrix, head_predict_matrix, test_facts):
    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    # Filter the head entity for 'predict_matrix'.
    test_head_entity = test_facts[:, 0]
    test_entity_dic = {}
    for key in tail_predict_matrix.keys():
        if list(key)[0] in test_head_entity:
            # test_entity_dic -- key: test_head_entity  value:[[tail_entity, count], ...]
            if list(key)[0] in test_entity_dic.keys():
                temp_list = test_entity_dic.get(list(key)[0])
            else:
                temp_list = []
            temp_list.append([list(key)[1], tail_predict_matrix[key]])
            test_entity_dic[list(key)[0]] = temp_list
    if len(test_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".format(len(test_head_entity), len(test_entity_dic)))

    # Todo: Rank the predicted facts by rule number! Not CD!
    for head in test_entity_dic.keys():
        test_entity_dic.get(head).sort(key=lambda x: x[1], reverse=True)
    # Calculate the MRR and Hit@10.
    test_result_for_tail_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_entity_dic.get(test_fact[0]) is not None:
            tail_list = [row[0] for row in test_entity_dic.get(test_fact[0])]
            test_number += 1
        else:
            tail_list = []
        if test_fact[1] in tail_list:
            top = tail_list.index([test_fact[1]]) + 1
            right_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        print(right_Hits_10)
        print("test_number: {0}".format(str(test_number)))
        print("test_facts: {0}".format(str(len(test_facts))))
        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10 = right_Hits_10 / float(test_number)
        right_Hits_3 = right_Hits_3 / float(test_number)
        right_Hits_1 = right_Hits_1 / float(test_number)
        right_MRR = right_MRR / float(test_number)
    else:
        right_Hits_10 = 0.0
        right_Hits_3 = 0.0
        right_Hits_1 = 0.0
        right_MRR = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0
    # Save the results in file.
    with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
        for item in test_result_for_tail_prediction:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
        f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
        f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
        f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))

        f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
        f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
        f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
        f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    # Filter the tail entity for 'predict_matrix'.
    test_tail_entity = test_facts[:, 1]
    test_entity_dic = {}
    for key in head_predict_matrix.keys():
        if list(key)[1] in test_tail_entity:
            # test_entity_dic -- key: test_tail_entity  value:[[head_entity, count], ...]
            if list(key)[1] in test_entity_dic.keys():
                temp_list = test_entity_dic.get(list(key)[1])
            else:
                temp_list = []
            temp_list.append([list(key)[0], head_predict_matrix[key]])
            test_entity_dic[list(key)[1]] = temp_list
    if len(test_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".format(len(test_tail_entity), len(test_entity_dic)))

    # Rank the predicted facts by rule number! Not CD!
    for tail in test_entity_dic.keys():
        test_entity_dic.get(tail).sort(key=lambda x: x[1], reverse=True)

    # Calculate the MRR and Hit@10.
    test_result_for_head_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_entity_dic.get(test_fact[1]) is not None:
            head_list = [row[0] for row in test_entity_dic.get(test_fact[1])]
            test_number += 1
        else:
            head_list = []
        if test_fact[0] in head_list:
            top = head_list.index([test_fact[0]]) + 1
            left_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10 = left_Hits_10 / float(test_number)
        left_Hits_3 = left_Hits_3 / float(test_number)
        left_Hits_1 = left_Hits_1 / float(test_number)
        left_MRR = left_MRR / float(test_number)
    else:
        left_Hits_10 = 0.0
        left_Hits_3 = 0.0
        left_Hits_1 = 0.0
        left_MRR = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

    # Save the results in file.
    with open("{0}test_Pt_{1}_left.txt".format(lp_save_path, str(pt)), 'w') as f:
        for item in test_result_for_head_prediction:
            f.write(str(item) + '\n')
        f.write("For Pt: %d, Hits@10: %f\n" % (pt, left_Hits_10))
        f.write("For Pt: %d, Hits@3: %f\n" % (pt, left_Hits_3))
        f.write("For Pt: %d, Hits@1: %f\n" % (pt, left_Hits_1))
        f.write("For Pt: %d, MRR: %f\n" % (pt, left_MRR))

        f.write("For Pt: %d, Hits@10_all: %f\n" % (pt, left_Hits_10_all))
        f.write("For Pt: %d, Hits@3_all: %f\n" % (pt, left_Hits_3_all))
        f.write("For Pt: %d, Hits@1_all: %f\n" % (pt, left_Hits_1_all))
        f.write("For Pt: %d, MRR_all: %f\n" % (pt, left_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, left_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, left_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, left_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, left_MRR))

    print("For Pt: %d, Hits@10_all: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3_all: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1_all: %f" % (pt, left_Hits_1_all))
    print("For Pt: %d, MRR_all: %f" % (pt, left_MRR_all))

    Hit_10 = (left_Hits_10 + right_Hits_10) / 2
    Hit_3 = (left_Hits_3 + right_Hits_3) / 2
    Hit_1 = (left_Hits_1 + right_Hits_1) / 2
    MRR = (left_MRR + right_MRR) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    # return MRR, Hit_1, Hit_3, Hit_10
    return MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all


# 2
def testByNoisyOR(lp_save_path, pt, test_facts, predict_facts_by_rule):
    test_head_entity = test_facts[:, 0]
    test_head_entity_dic = {}  # key: h  value:[t, t_SC]
    test_tail_entity = test_facts[:, 1]
    test_tail_entity_dic = {}  # key: t  value:[h, h_SC]
    for key in predict_facts_by_rule.keys():
        SC_list = predict_facts_by_rule.get(key)
        tmp = 1
        for x in SC_list:
            tmp *= 1 - x
        key_SC = 1 - tmp
        if list(key)[0] in test_head_entity:
            if list(key)[0] in test_head_entity_dic:
                templist = test_head_entity_dic.get(list(key)[0])
            else:
                templist = []
            templist.append([list(key)[1], key_SC])
            test_head_entity_dic[list(key)[0]] = templist
        if list(key)[1] in test_tail_entity:
            if list(key)[1] in test_tail_entity_dic:
                templist = test_tail_entity_dic.get(list(key)[1])
            else:
                templist = []
            templist.append([list(key)[0], key_SC])
            test_tail_entity_dic[list(key)[1]] = templist
    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    if len(test_head_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".
          format(len(test_head_entity), len(test_head_entity_dic)))

    # Rank the predicted facts by CD
    for head in test_head_entity_dic.keys():
        test_head_entity_dic.get(head).sort(key=lambda x: x[1], reverse=True)
    # Calculate the MRR and Hit@10.
    # test_result_for_tail_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_head_entity_dic.get(test_fact[0]) is not None:
            tail_list = [row[0] for row in test_head_entity_dic.get(test_fact[0])]
            test_number += 1
        else:
            tail_list = []
        if test_fact[1] in tail_list:
            top = tail_list.index(test_fact[1]) + 1
            right_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        # test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        print(right_Hits_10)
        print("test_number: {0}".format(str(test_number)))
        print("test_facts: {0}".format(str(len(test_facts))))
        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10 = right_Hits_10 / float(test_number)
        right_Hits_3 = right_Hits_3 / float(test_number)
        right_Hits_1 = right_Hits_1 / float(test_number)
        right_MRR = right_MRR / float(test_number)
    else:
        right_Hits_10 = 0.0
        right_Hits_3 = 0.0
        right_Hits_1 = 0.0
        right_MRR = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0
    # Save the results in file.
    # with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_tail_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))
    #
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    if len(test_tail_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".
          format(len(test_tail_entity), len(test_tail_entity_dic)))

    # Rank the predicted facts by rule number! Not CD!
    for tail in test_tail_entity_dic.keys():
        test_tail_entity_dic.get(tail).sort(key=lambda x: x[1], reverse=True)

    # Calculate the MRR and Hit@10.
    # test_result_for_head_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_tail_entity_dic.get(test_fact[1]) is not None:
            head_list = [row[0] for row in test_tail_entity_dic.get(test_fact[1])]
            test_number += 1
        else:
            head_list = []
        if test_fact[0] in head_list:
            top = head_list.index(test_fact[0]) + 1
            left_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        # test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10 = left_Hits_10 / float(test_number)
        left_Hits_3 = left_Hits_3 / float(test_number)
        left_Hits_1 = left_Hits_1 / float(test_number)
        left_MRR = left_MRR / float(test_number)
    else:
        left_Hits_10 = 0.0
        left_Hits_3 = 0.0
        left_Hits_1 = 0.0
        left_MRR = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

    # Save the results in file.
    # with open("{0}test_Pt_{1}_left.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_head_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, left_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, left_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, left_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, left_MRR))
    #
    #     f.write("For Pt: %d, Hits@10_all: %f\n" % (pt, left_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3_all: %f\n" % (pt, left_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1_all: %f\n" % (pt, left_Hits_1_all))
    #     f.write("For Pt: %d, MRR_all: %f\n" % (pt, left_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, left_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, left_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, left_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, left_MRR))

    print("For Pt: %d, Hits@10_all: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3_all: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1_all: %f" % (pt, left_Hits_1_all))
    print("For Pt: %d, MRR_all: %f" % (pt, left_MRR_all))

    Hit_10 = (left_Hits_10 + right_Hits_10) / 2
    Hit_3 = (left_Hits_3 + right_Hits_3) / 2
    Hit_1 = (left_Hits_1 + right_Hits_1) / 2
    MRR = (left_MRR + right_MRR) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    # return MRR, Hit_1, Hit_3, Hit_10
    return MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all


def testByNoisyOR_2(lp_save_path, pt, test_facts, predict_facts_by_rule_right, predict_facts_by_rule_left):
    test_head_entity = test_facts[:, 0]
    test_head_entity_dic = {}  # key: h  value:[t, t_SC]
    test_tail_entity = test_facts[:, 1]
    test_tail_entity_dic = {}  # key: t  value:[h, h_SC]
    for key in predict_facts_by_rule_right.keys():
        SC_list = predict_facts_by_rule_right.get(key)
        tmp = 1
        for x in SC_list:
            tmp *= 1 - x
        key_SC = 1 - tmp
        if list(key)[0] in test_head_entity:
            if list(key)[0] in test_head_entity_dic:
                templist = test_head_entity_dic.get(list(key)[0])
            else:
                templist = []
            templist.append([list(key)[1], key_SC])
            test_head_entity_dic[list(key)[0]] = templist
    for key in predict_facts_by_rule_left.keys():
        SC_list = predict_facts_by_rule_left.get(key)
        tmp = 1
        for x in SC_list:
            tmp *= 1 - x
        key_SC = 1 - tmp
        if list(key)[1] in test_tail_entity:
            if list(key)[1] in test_tail_entity_dic:
                templist = test_tail_entity_dic.get(list(key)[1])
            else:
                templist = []
            templist.append([list(key)[0], key_SC])
            test_tail_entity_dic[list(key)[1]] = templist

    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    if len(test_head_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".
          format(len(test_head_entity), len(test_head_entity_dic)))

    # Rank the predicted facts by CD
    for head in test_head_entity_dic.keys():
        test_head_entity_dic.get(head).sort(key=lambda x: x[1], reverse=True)
    # Calculate the MRR and Hit@10.
    # test_result_for_tail_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_head_entity_dic.get(test_fact[0]) is not None:
            tail_list = [row[0] for row in test_head_entity_dic.get(test_fact[0])]
            test_number += 1
        else:
            tail_list = []
        if test_fact[1] in tail_list:
            top = tail_list.index(test_fact[1]) + 1
            right_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        # test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        print(right_Hits_10)
        print("test_number: {0}".format(str(test_number)))
        print("test_facts: {0}".format(str(len(test_facts))))
        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10 = right_Hits_10 / float(test_number)
        right_Hits_3 = right_Hits_3 / float(test_number)
        right_Hits_1 = right_Hits_1 / float(test_number)
        right_MRR = right_MRR / float(test_number)
    else:
        right_Hits_10 = 0.0
        right_Hits_3 = 0.0
        right_Hits_1 = 0.0
        right_MRR = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0
    # Save the results in file.
    # with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_tail_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))
    #
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    if len(test_tail_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".
          format(len(test_tail_entity), len(test_tail_entity_dic)))

    # Rank the predicted facts by rule number! Not CD!
    for tail in test_tail_entity_dic.keys():
        test_tail_entity_dic.get(tail).sort(key=lambda x: x[1], reverse=True)

    # Calculate the MRR and Hit@10.
    # test_result_for_head_prediction = []
    test_number = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_tail_entity_dic.get(test_fact[1]) is not None:
            head_list = [row[0] for row in test_tail_entity_dic.get(test_fact[1])]
            test_number += 1
        else:
            head_list = []
        if test_fact[0] in head_list:
            top = head_list.index(test_fact[0]) + 1
            left_MRR += 1 / float(top)
        else:
            top = -1
        Hit = 0
        if 0 < top <= 10:
            Hit = 1
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        # test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10 = left_Hits_10 / float(test_number)
        left_Hits_3 = left_Hits_3 / float(test_number)
        left_Hits_1 = left_Hits_1 / float(test_number)
        left_MRR = left_MRR / float(test_number)
    else:
        left_Hits_10 = 0.0
        left_Hits_3 = 0.0
        left_Hits_1 = 0.0
        left_MRR = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

    # Save the results in file.
    # with open("{0}test_Pt_{1}_left.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_head_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, left_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, left_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, left_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, left_MRR))
    #
    #     f.write("For Pt: %d, Hits@10_all: %f\n" % (pt, left_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3_all: %f\n" % (pt, left_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1_all: %f\n" % (pt, left_Hits_1_all))
    #     f.write("For Pt: %d, MRR_all: %f\n" % (pt, left_MRR_all))
    print("For Pt: %d, Hits@10: %f" % (pt, left_Hits_10))
    print("For Pt: %d, Hits@3: %f" % (pt, left_Hits_3))
    print("For Pt: %d, Hits@1: %f" % (pt, left_Hits_1))
    print("For Pt: %d, MRR: %f" % (pt, left_MRR))

    print("For Pt: %d, Hits@10_all: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3_all: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1_all: %f" % (pt, left_Hits_1_all))
    print("For Pt: %d, MRR_all: %f" % (pt, left_MRR_all))

    Hit_10 = (left_Hits_10 + right_Hits_10) / 2
    Hit_3 = (left_Hits_3 + right_Hits_3) / 2
    Hit_1 = (left_Hits_1 + right_Hits_1) / 2
    MRR = (left_MRR + right_MRR) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    # return MRR, Hit_1, Hit_3, Hit_10
    return MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all


# 3 THIS
def testByMaxAggregation(lp_save_path, pt, test_facts, predict_facts_by_rule, h_r_dic_t, r_t_dic_h):
    # notPredictedTestFacts_only1 = []  # cannot predict test facts
    # notPredictedTestFacts_only12 = []  # cannot predict test facts
    test_head_entity = test_facts[:, 0]
    test_head_entity_dic = {}  # key: h  value:[t, t_SC]
    test_tail_entity = test_facts[:, 1]
    test_tail_entity_dic = {}  # key: t  value:[h, h_SC]
    for key in predict_facts_by_rule.keys():
        # Ranked by SC
        SC_list = predict_facts_by_rule.get(key)
        SC_list.sort(reverse=True)
        if list(key)[0] in test_head_entity:
            if list(key)[0] in test_head_entity_dic:
                templist = test_head_entity_dic.get(list(key)[0])
            else:
                templist = []
            templist.append([list(key)[1], SC_list])
            test_head_entity_dic[list(key)[0]] = templist
        if list(key)[1] in test_tail_entity:
            if list(key)[1] in test_tail_entity_dic:
                templist = test_tail_entity_dic.get(list(key)[1])
            else:
                templist = []
            templist.append([list(key)[0], SC_list])
            test_tail_entity_dic[list(key)[1]] = templist

    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    if len(test_head_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".
          format(len(test_head_entity), len(test_head_entity_dic)))

    # Calculate the MRR and Hit@10.
    # test_result_for_tail_prediction = []
    test_number_23 = 0
    test_number_3 = 0
    for test_fact in test_facts:
        # print(test_fact)
        # t = [test_fact[0], test_fact[1]]
        if test_head_entity_dic.get(test_fact[0]) is not None:
            test_number_23 += 1
            _temp = test_head_entity_dic.get(test_fact[0])
            # e.g. _temp = [["t1", [0.9, 0.1]], ["t2", [0.9, 0.8]], ["t3", [0.8, 0.8]]]
            filter_entities = h_r_dic_t.get((test_fact[0], pt))
            if filter_entities is not None:
                # print(filter_entities)
                for row in _temp:
                    if row[0] in filter_entities:
                        # print(row[0])
                        _temp.remove(row)
                        # print(_temp)
            _temp.sort(key=lambda x: x[1][0], reverse=True)
            tail = [i[0] for i in _temp]
            t = [i[1][0] for i in _temp]
            rank_list = [t.index(i) + 1 for i in t]
            if test_fact[1] in tail:
                test_number_3 += 1
                top = rank_list[tail.index(test_fact[1])]
                right_MRR += 1 / float(top)
            else:
                # notPredictedTestFacts_only12.append([1, test_fact])
                top = -1
        else:
            # notPredictedTestFacts_only1.append([1, test_fact])
            # notPredictedTestFacts_only12.append([1, test_fact])
            top = -1
        if 0 < top <= 10:
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        # test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number_23 != 0:
        # print(right_Hits_10)
        # print("test_number: {0}".format(str(test_number_23)))
        # print("test_facts: {0}".format(str(len(test_facts))))

        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10_part23 = right_Hits_10 / float(test_number_23)
        right_Hits_3_part23 = right_Hits_3 / float(test_number_23)
        right_Hits_1_part23 = right_Hits_1 / float(test_number_23)
        right_MRR_part23 = right_MRR / float(test_number_23)

        if test_number_3 != 0:
            right_Hits_10_part3 = right_Hits_10 / float(test_number_3)
            right_Hits_3_part3 = right_Hits_3 / float(test_number_3)
            right_Hits_1_part3 = right_Hits_1 / float(test_number_3)
            right_MRR_part3 = right_MRR / float(test_number_3)
        else:
            right_Hits_10_part3 = 0.0
            right_Hits_3_part3 = 0.0
            right_Hits_1_part3 = 0.0
            right_MRR_part3 = 0.0
    else:
        right_Hits_10_part23 = 0.0
        right_Hits_3_part23 = 0.0
        right_Hits_1_part23 = 0.0
        right_MRR_part23 = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0

        right_Hits_10_part3 = 0.0
        right_Hits_3_part3 = 0.0
        right_Hits_1_part3 = 0.0
        right_MRR_part3 = 0.0
    # Save the results in file.
    # with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_tail_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))
    #
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10_part3: %f" % (pt, right_Hits_10_part3))
    print("For Pt: %d, Hits@3_part3: %f" % (pt, right_Hits_3_part3))
    print("For Pt: %d, Hits@1_part3: %f" % (pt, right_Hits_1_part3))
    print("For Pt: %d, MRR_part3: %f" % (pt, right_MRR_part3))

    print("For Pt: %d, Hits@10_part23: %f" % (pt, right_Hits_10_part23))
    print("For Pt: %d, Hits@3_part23: %f" % (pt, right_Hits_3_part23))
    print("For Pt: %d, Hits@1_part23: %f" % (pt, right_Hits_1_part23))
    print("For Pt: %d, MRR_part23: %f" % (pt, right_MRR_part23))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    if len(test_tail_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".
          format(len(test_tail_entity), len(test_tail_entity_dic)))

    # Calculate the MRR and Hit@10.
    # test_result_for_head_prediction = []
    test_number_23 = 0
    test_number_3 = 0
    for test_fact in test_facts:
        # print(test_fact)
        # t = [test_fact[0], test_fact[1]]
        if test_tail_entity_dic.get(test_fact[1]) is not None:
            test_number_23 += 1
            _temp = test_tail_entity_dic.get(test_fact[1])
            # e.g. _temp = [["t1", [0.9, 0.1]], ["t2", [0.9, 0.8]], ["t3", [0.8, 0.8]]]
            filter_entities = r_t_dic_h.get((pt, test_fact[1]))
            if filter_entities is not None:
                for row in _temp:
                    if row[0] in filter_entities:
                        _temp.remove(row)
            _temp.sort(key=lambda x: x[1][0], reverse=True)
            head = [i[0] for i in _temp]
            t = [i[1][0] for i in _temp]
            rank_list = [t.index(i) + 1 for i in t]
            if test_fact[0] in head:
                test_number_3 += 1
                top = rank_list[head.index(test_fact[0])]
                left_MRR += 1 / float(top)
            else:
                # 0: predict left, 1: predict right.
                # notPredictedTestFacts_only12.append([0, test_fact])
                top = -1
        else:
            # notPredictedTestFacts_only1.append([0, test_fact])
            # notPredictedTestFacts_only12.append([0, test_fact])
            top = -1
        if 0 < top <= 10:
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        # test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number_23 != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10_part23 = left_Hits_10 / float(test_number_23)
        left_Hits_3_part23 = left_Hits_3 / float(test_number_23)
        left_Hits_1_part23 = left_Hits_1 / float(test_number_23)
        left_MRR_part23 = left_MRR / float(test_number_23)
        if test_number_3 != 0:
            left_Hits_10_part3 = left_Hits_10 / float(test_number_3)
            left_Hits_3_part3 = left_Hits_3 / float(test_number_3)
            left_Hits_1_part3 = left_Hits_1 / float(test_number_3)
            left_MRR_part3 = left_MRR / float(test_number_3)
        else:
            left_Hits_10_part3 = 0.0
            left_Hits_3_part3 = 0.0
            left_Hits_1_part3 = 0.0
            left_MRR_part3 = 0.0
    else:
        left_Hits_10_part23 = 0.0
        left_Hits_3_part23 = 0.0
        left_Hits_1_part23 = 0.0
        left_MRR_part23 = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

        left_Hits_10_part3 = 0.0
        left_Hits_3_part3 = 0.0
        left_Hits_1_part3 = 0.0
        left_MRR_part3 = 0.0

    print("For Pt: %d, Hits@10_part3: %f" % (pt, left_Hits_10_part3))
    print("For Pt: %d, Hits@3_part3: %f" % (pt, left_Hits_3_part3))
    print("For Pt: %d, Hits@1_part3: %f" % (pt, left_Hits_1_part3))
    print("For Pt: %d, MRR_part3: %f" % (pt, left_MRR_part3))

    print("For Pt: %d, Hits@10_part23: %f" % (pt, left_Hits_10_part23))
    print("For Pt: %d, Hits@3_part23: %f" % (pt, left_Hits_3_part23))
    print("For Pt: %d, Hits@1_part23: %f" % (pt, left_Hits_1_part23))
    print("For Pt: %d, MRR_part23: %f" % (pt, left_MRR_part23))

    print("For Pt: %d, Hits@10: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, left_MRR_all))

    # # Save the notPredictedTestFacts in file.
    # with open("{0}notPredictedTestFacts_only1.txt".format(lp_save_path), 'a') as f:
    #     for item in notPredictedTestFacts_only1:
    #         f.write("{0}\t{1}\t{2}\t{3}\n".format(str(item[0]), str(item[1][0]), str(item[1][1]), str(item[1][2])))
    #     f.flush()
    # with open("{0}notPredictedTestFacts_only12.txt".format(lp_save_path), 'a') as f:
    #     for item in notPredictedTestFacts_only12:
    #         f.write("{0}\t{1}\t{2}\t{3}\n".format(str(item[0]), str(item[1][0]), str(item[1][1]), str(item[1][2])))
    #     f.flush()

    Hit_10_part3 = (left_Hits_10_part3 + right_Hits_10_part3) / 2
    Hit_3_part3 = (left_Hits_3_part3 + right_Hits_3_part3) / 2
    Hit_1_part3 = (left_Hits_1_part3 + right_Hits_1_part3) / 2
    MRR_part3 = (left_MRR_part3 + right_MRR_part3) / 2

    Hit_10_part23 = (left_Hits_10_part23 + right_Hits_10_part23) / 2
    Hit_3_part23 = (left_Hits_3_part23 + right_Hits_3_part23) / 2
    Hit_1_part23 = (left_Hits_1_part23 + right_Hits_1_part23) / 2
    MRR_part23 = (left_MRR_part23 + right_MRR_part23) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    return [right_MRR_part3, right_Hits_1_part3, right_Hits_3_part3, right_Hits_10_part3,
            left_MRR_part3, left_Hits_1_part3, left_Hits_3_part3, left_Hits_10_part3,
            MRR_part3, Hit_1_part3, Hit_3_part3, Hit_10_part3,

            right_MRR_part23, right_Hits_1_part23, right_Hits_3_part23, right_Hits_10_part23,
            left_MRR_part23, left_Hits_1_part23, left_Hits_3_part23, left_Hits_10_part23,
            MRR_part23, Hit_1_part23, Hit_3_part23, Hit_10_part23,

            right_MRR_all, right_Hits_1_all, right_Hits_3_all, right_Hits_10_all,
            left_MRR_all, left_Hits_1_all, left_Hits_3_all, left_Hits_10_all,
            MRR_all, Hit_1_all, Hit_3_all, Hit_10_all]


# 3.1 THIS
def testByMaxAggregation_save_middle_results(save_path, pt, test_facts, predict_facts_by_rule, h_r_dic_t, r_t_dic_h):
    test_head_entity = test_facts[:, 0]
    test_head_entity_dic = {}  # key: h  value:[[t, t_SC],...]
    test_tail_entity = test_facts[:, 1]
    test_tail_entity_dic = {}  # key: t  value:[[h, h_SC],...]
    for key in predict_facts_by_rule.keys():
        # Ranked by SC
        SC_list = predict_facts_by_rule.get(key)
        SC_list.sort(reverse=True)
        if list(key)[0] in test_head_entity:
            if list(key)[0] in test_head_entity_dic.keys():
                templist = test_head_entity_dic.get(list(key)[0])
            else:
                templist = []
            filter_entities = h_r_dic_t.get((list(key)[0], pt))
            if filter_entities is not None:
                if list(key)[1] not in filter_entities:
                    templist.append([list(key)[1], SC_list[0]])  # max SC:SC_list[0]
            else:
                templist.append([list(key)[1], SC_list[0]])
            test_head_entity_dic[list(key)[0]] = templist
        if list(key)[1] in test_tail_entity:
            if list(key)[1] in test_tail_entity_dic.keys():
                templist = test_tail_entity_dic.get(list(key)[1])
            else:
                templist = []
            filter_entities = r_t_dic_h.get((pt, list(key)[1]))
            if filter_entities is not None:
                if list(key)[0] not in filter_entities:
                    templist.append([list(key)[0], SC_list[0]])
            else:
                templist.append([list(key)[0], SC_list[0]])
            test_tail_entity_dic[list(key)[1]] = templist

    # filter for top 100 entities.
    for key in test_head_entity_dic.keys():
        templist = test_head_entity_dic.get(key)
        templist.sort(reverse=True)
        if len(templist) <= 100:
            test_head_entity_dic[key] = templist
        else:
            test_head_entity_dic[key] = templist[0:100]
    for key in test_tail_entity_dic.keys():
        templist = test_tail_entity_dic.get(key)
        templist.sort(reverse=True)
        if len(templist) <= 100:
            test_tail_entity_dic[key] = templist
        else:
            test_tail_entity_dic[key] = templist[0:100]

    for head in test_head_entity:
        if head not in test_head_entity_dic.keys():
            test_head_entity_dic[head] = []
    for tail in test_tail_entity:
        if tail not in test_tail_entity_dic.keys():
            test_tail_entity_dic[tail] = []

    # if len(test_head_entity_dic) == len(test_head_entity) and len(test_tail_entity) == len(test_tail_entity_dic):
    #     print("Equally. If dic == head entity, it's right.")
    # else:
    #     print("NOT equally. If dic < head entity, it's wrong.")

    # Save the middle results.
    with open("{0}/r_{1}_right.txt".format(save_path, str(pt)), 'w') as f:
        for key in test_head_entity_dic.keys():
            ent_lines = ""
            if len(test_head_entity_dic.get(key)) == 0:
                ent_lines = "None"
            else:
                for e in test_head_entity_dic.get(key):
                    ent_lines += "{0}:{1},".format(e[0], e[1])
                ent_lines = ent_lines.rstrip(",")
            f.write("{0}\t{1}\n".format(key, ent_lines))
        f.flush()
    with open("{0}/r_{1}_left.txt".format(save_path, str(pt)), 'w') as f:
        for key in test_tail_entity_dic.keys():
            ent_lines = ""
            if len(test_tail_entity_dic.get(key)) == 0:
                ent_lines = "None"
            else:
                for e in test_tail_entity_dic.get(key):
                    ent_lines += "{0}:{1},".format(e[0], e[1])
                ent_lines = ent_lines.rstrip(",")
            f.write("{0}\t{1}\n".format(key, ent_lines))
        f.flush()


def testByMaxAggregation_2(lp_save_path, pt, test_facts,
                           predict_facts_by_rule_right, predict_facts_by_rule_left,
                           h_r_dic_t, r_t_dic_h):
    notPredictedTestFacts_only1 = []  # cannot predict test facts
    notPredictedTestFacts_only12 = []  # cannot predict test facts
    test_head_entity = test_facts[:, 0]
    test_head_entity_dic = {}  # key: h  value:[t, t_SC]
    test_tail_entity = test_facts[:, 1]
    test_tail_entity_dic = {}  # key: t  value:[h, h_SC]
    for key in predict_facts_by_rule_right.keys():
        # Ranked by SC
        SC_list = predict_facts_by_rule_right.get(key)
        SC_list.sort(reverse=True)
        if list(key)[0] in test_head_entity:
            if list(key)[0] in test_head_entity_dic:
                templist = test_head_entity_dic.get(list(key)[0])
            else:
                templist = []
            templist.append([list(key)[1], SC_list])
            test_head_entity_dic[list(key)[0]] = templist
    for key in predict_facts_by_rule_left.keys():
        # Ranked by SC
        SC_list = predict_facts_by_rule_left.get(key)
        SC_list.sort(reverse=True)
        if list(key)[1] in test_tail_entity:
            if list(key)[1] in test_tail_entity_dic:
                templist = test_tail_entity_dic.get(list(key)[1])
            else:
                templist = []
            templist.append([list(key)[0], SC_list])
            test_tail_entity_dic[list(key)[1]] = templist

    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    if len(test_head_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".
          format(len(test_head_entity), len(test_head_entity_dic)))

    # Calculate the MRR and Hit@10.
    # test_result_for_tail_prediction = []
    test_number_23 = 0
    test_number_3 = 0
    for test_fact in test_facts:
        # print(test_fact)
        # t = [test_fact[0], test_fact[1]]
        if test_head_entity_dic.get(test_fact[0]) is not None:
            test_number_23 += 1
            _temp = test_head_entity_dic.get(test_fact[0])
            # e.g. _temp = [["t1", [0.9, 0.1]], ["t2", [0.9, 0.8]], ["t3", [0.8, 0.8]]]
            filter_entities = h_r_dic_t.get((test_fact[0], pt))
            if filter_entities is not None:
                # print(filter_entities)
                for row in _temp:
                    if row[0] in filter_entities:
                        # print(row[0])
                        _temp.remove(row)
                        # print(_temp)
            _temp.sort(key=lambda x: x[1][0], reverse=True)
            tail = [i[0] for i in _temp]
            t = [i[1][0] for i in _temp]
            rank_list = [t.index(i) + 1 for i in t]
            if test_fact[1] in tail:
                test_number_3 += 1
                top = rank_list[tail.index(test_fact[1])]
                right_MRR += 1 / float(top)
            else:
                notPredictedTestFacts_only12.append([1, test_fact])
                top = -1
        else:
            notPredictedTestFacts_only1.append([1, test_fact])
            notPredictedTestFacts_only12.append([1, test_fact])
            top = -1
        if 0 < top <= 10:
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        # test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number_23 != 0:

        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10_part23 = right_Hits_10 / float(test_number_23)
        right_Hits_3_part23 = right_Hits_3 / float(test_number_23)
        right_Hits_1_part23 = right_Hits_1 / float(test_number_23)
        right_MRR_part23 = right_MRR / float(test_number_23)

        if test_number_3 != 0:
            right_Hits_10_part3 = right_Hits_10 / float(test_number_3)
            right_Hits_3_part3 = right_Hits_3 / float(test_number_3)
            right_Hits_1_part3 = right_Hits_1 / float(test_number_3)
            right_MRR_part3 = right_MRR / float(test_number_3)
        else:
            right_Hits_10_part3 = 0.0
            right_Hits_3_part3 = 0.0
            right_Hits_1_part3 = 0.0
            right_MRR_part3 = 0.0
    else:
        right_Hits_10_part23 = 0.0
        right_Hits_3_part23 = 0.0
        right_Hits_1_part23 = 0.0
        right_MRR_part23 = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0

        right_Hits_10_part3 = 0.0
        right_Hits_3_part3 = 0.0
        right_Hits_1_part3 = 0.0
        right_MRR_part3 = 0.0
    # Save the results in file.
    # with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_tail_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))
    #
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10_part3: %f" % (pt, right_Hits_10_part3))
    print("For Pt: %d, Hits@3_part3: %f" % (pt, right_Hits_3_part3))
    print("For Pt: %d, Hits@1_part3: %f" % (pt, right_Hits_1_part3))
    print("For Pt: %d, MRR_part3: %f" % (pt, right_MRR_part3))

    print("For Pt: %d, Hits@10_part23: %f" % (pt, right_Hits_10_part23))
    print("For Pt: %d, Hits@3_part23: %f" % (pt, right_Hits_3_part23))
    print("For Pt: %d, Hits@1_part23: %f" % (pt, right_Hits_1_part23))
    print("For Pt: %d, MRR_part23: %f" % (pt, right_MRR_part23))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    if len(test_tail_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".
          format(len(test_tail_entity), len(test_tail_entity_dic)))

    # Calculate the MRR and Hit@10.
    # test_result_for_head_prediction = []
    test_number_23 = 0
    test_number_3 = 0
    for test_fact in test_facts:
        # print(test_fact)
        t = [test_fact[0], test_fact[1]]
        if test_tail_entity_dic.get(test_fact[1]) is not None:
            test_number_23 += 1
            _temp = test_tail_entity_dic.get(test_fact[1])
            # e.g. _temp = [["t1", [0.9, 0.1]], ["t2", [0.9, 0.8]], ["t3", [0.8, 0.8]]]
            filter_entities = r_t_dic_h.get((pt, test_fact[1]))
            if filter_entities is not None:
                for row in _temp:
                    if row[0] in filter_entities:
                        _temp.remove(row)
            _temp.sort(key=lambda x: x[1][0], reverse=True)
            head = [i[0] for i in _temp]
            t = [i[1][0] for i in _temp]
            rank_list = [t.index(i) + 1 for i in t]
            if test_fact[0] in head:
                test_number_3 += 1
                top = rank_list[head.index(test_fact[0])]
                left_MRR += 1 / float(top)
            else:
                # 0: predict left, 1: predict right.
                notPredictedTestFacts_only12.append([0, test_fact])
                top = -1
        else:
            notPredictedTestFacts_only1.append([0, test_fact])
            notPredictedTestFacts_only12.append([0, test_fact])
            top = -1
        if 0 < top <= 10:
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        # test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number_23 != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10_part23 = left_Hits_10 / float(test_number_23)
        left_Hits_3_part23 = left_Hits_3 / float(test_number_23)
        left_Hits_1_part23 = left_Hits_1 / float(test_number_23)
        left_MRR_part23 = left_MRR / float(test_number_23)
        if test_number_3 != 0:
            left_Hits_10_part3 = left_Hits_10 / float(test_number_3)
            left_Hits_3_part3 = left_Hits_3 / float(test_number_3)
            left_Hits_1_part3 = left_Hits_1 / float(test_number_3)
            left_MRR_part3 = left_MRR / float(test_number_3)
        else:
            left_Hits_10_part3 = 0.0
            left_Hits_3_part3 = 0.0
            left_Hits_1_part3 = 0.0
            left_MRR_part3 = 0.0
    else:
        left_Hits_10_part23 = 0.0
        left_Hits_3_part23 = 0.0
        left_Hits_1_part23 = 0.0
        left_MRR_part23 = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

        left_Hits_10_part3 = 0.0
        left_Hits_3_part3 = 0.0
        left_Hits_1_part3 = 0.0
        left_MRR_part3 = 0.0

    print("For Pt: %d, Hits@10_part3: %f" % (pt, left_Hits_10_part3))
    print("For Pt: %d, Hits@3_part3: %f" % (pt, left_Hits_3_part3))
    print("For Pt: %d, Hits@1_part3: %f" % (pt, left_Hits_1_part3))
    print("For Pt: %d, MRR_part3: %f" % (pt, left_MRR_part3))

    print("For Pt: %d, Hits@10_part23: %f" % (pt, left_Hits_10_part23))
    print("For Pt: %d, Hits@3_part23: %f" % (pt, left_Hits_3_part23))
    print("For Pt: %d, Hits@1_part23: %f" % (pt, left_Hits_1_part23))
    print("For Pt: %d, MRR_part23: %f" % (pt, left_MRR_part23))

    print("For Pt: %d, Hits@10_all: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3_all: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1_all: %f" % (pt, left_Hits_1_all))
    print("For Pt: %d, MRR_all: %f" % (pt, left_MRR_all))

    # Save the notPredictedTestFacts in file.
    with open("{0}notPredictedTestFacts_only1.txt".format(lp_save_path), 'a') as f:
        for item in notPredictedTestFacts_only1:
            f.write("{0}\t{1}\t{2}\t{3}\n".format(str(item[0]), str(item[1][0]), str(item[1][1]), str(item[1][2])))
        f.flush()
    with open("{0}notPredictedTestFacts_only12.txt".format(lp_save_path), 'a') as f:
        for item in notPredictedTestFacts_only12:
            f.write("{0}\t{1}\t{2}\t{3}\n".format(str(item[0]), str(item[1][0]), str(item[1][1]), str(item[1][2])))
        f.flush()

    Hit_10_part3 = (left_Hits_10_part3 + right_Hits_10_part3) / 2
    Hit_3_part3 = (left_Hits_3_part3 + right_Hits_3_part3) / 2
    Hit_1_part3 = (left_Hits_1_part3 + right_Hits_1_part3) / 2
    MRR_part3 = (left_MRR_part3 + right_MRR_part3) / 2

    Hit_10_part23 = (left_Hits_10_part23 + right_Hits_10_part23) / 2
    Hit_3_part23 = (left_Hits_3_part23 + right_Hits_3_part23) / 2
    Hit_1_part23 = (left_Hits_1_part23 + right_Hits_1_part23) / 2
    MRR_part23 = (left_MRR_part23 + right_MRR_part23) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    # return MRR, Hit_1, Hit_3, Hit_10
    return [right_MRR_part3, right_Hits_1_part3, right_Hits_3_part3, right_Hits_10_part3,
            left_MRR_part3, left_Hits_1_part3, left_Hits_3_part3, left_Hits_10_part3,
            MRR_part3, Hit_1_part3, Hit_3_part3, Hit_10_part3,

            right_MRR_part23, right_Hits_1_part23, right_Hits_3_part23, right_Hits_10_part23,
            left_MRR_part23, left_Hits_1_part23, left_Hits_3_part23, left_Hits_10_part23,
            MRR_part23, Hit_1_part23, Hit_3_part23, Hit_10_part23,

            right_MRR_all, right_Hits_1_all, right_Hits_3_all, right_Hits_10_all,
            left_MRR_all, left_Hits_1_all, left_Hits_3_all, left_Hits_10_all,
            MRR_all, Hit_1_all, Hit_3_all, Hit_10_all]


def testByMaxAggregation_filterType(lp_save_path, pt, test_facts, predict_facts_by_rule,
                                    h_r_dic_t, r_t_dic_h, x_entity_set, y_entity_set):
    # notPredictedTestFacts_only1 = []  # cannot predict test facts
    # notPredictedTestFacts_only12 = []  # cannot predict test facts
    test_head_entity = test_facts[:, 0]
    test_head_entity_dic = {}  # key: h  value:[t, t_SC]
    test_tail_entity = test_facts[:, 1]
    test_tail_entity_dic = {}  # key: t  value:[h, h_SC]
    for key in predict_facts_by_rule.keys():
        # Ranked by SC
        SC_list = predict_facts_by_rule.get(key)
        SC_list.sort(reverse=True)
        if list(key)[0] in test_head_entity:
            if list(key)[0] in test_head_entity_dic:
                templist = test_head_entity_dic.get(list(key)[0])
            else:
                templist = []
            templist.append([list(key)[1], SC_list])
            test_head_entity_dic[list(key)[0]] = templist
        if list(key)[1] in test_tail_entity:
            if list(key)[1] in test_tail_entity_dic:
                templist = test_tail_entity_dic.get(list(key)[1])
            else:
                templist = []
            templist.append([list(key)[0], SC_list])
            test_tail_entity_dic[list(key)[1]] = templist

    """ Predict the tail entity given head entity """
    print("Step 2.1: Begin to test the tail entity given head entity.")
    right_Hits_10 = 0
    right_Hits_3 = 0
    right_Hits_1 = 0
    right_MRR = 0
    if len(test_head_entity_dic) == len(test_head_entity):
        print("Equally. If dic == head entity, it's right.")
    else:
        print("NOT equally. If dic < head entity, it's right.")
        # print("dic size: %d" % len(test_entity_dic))
        # print("head entity size: %d" % len(test_head_entity))
    print("For {0} head entities, there are {1} can be predicted".
          format(len(test_head_entity), len(test_head_entity_dic)))

    # Calculate the MRR and Hit@10.
    # test_result_for_tail_prediction = []
    test_number_23 = 0
    test_number_3 = 0
    for test_fact in test_facts:
        # print(test_fact)
        # t = [test_fact[0], test_fact[1]]
        if test_head_entity_dic.get(test_fact[0]) is not None:
            test_number_23 += 1
            _temp = test_head_entity_dic.get(test_fact[0])
            # e.g. _temp = [["t1", [0.9, 0.1]], ["t2", [0.9, 0.8]], ["t3", [0.8, 0.8]]]
            # 1. Filter setting (original)
            filter_entities = h_r_dic_t.get((test_fact[0], pt))
            if filter_entities is not None:
                # print(filter_entities)
                for row in _temp:
                    if row[0] in filter_entities:
                        # print(row[0])
                        _temp.remove(row)
                        # print(_temp)
            # 2. Filter setting (type)
            if y_entity_set is not None:
                for row in _temp:
                    if row[0] not in y_entity_set:
                        # print(row[0])
                        _temp.remove(row)
            _temp.sort(key=lambda x: x[1][0], reverse=True)
            tail = [i[0] for i in _temp]
            t = [i[1][0] for i in _temp]
            rank_list = [t.index(i) + 1 for i in t]
            if test_fact[1] in tail:
                test_number_3 += 1
                top = rank_list[tail.index(test_fact[1])]
                right_MRR += 1 / float(top)
            else:
                # notPredictedTestFacts_only12.append([1, test_fact])
                top = -1
        else:
            # notPredictedTestFacts_only1.append([1, test_fact])
            # notPredictedTestFacts_only12.append([1, test_fact])
            top = -1
        if 0 < top <= 10:
            right_Hits_10 += 1
        if 0 < top <= 3:
            right_Hits_3 += 1
        if top == 1:
            right_Hits_1 += 1
        # test_result_for_tail_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number_23 != 0:

        right_Hits_10_all = right_Hits_10 / float(len(test_facts))
        right_Hits_3_all = right_Hits_3 / float(len(test_facts))
        right_Hits_1_all = right_Hits_1 / float(len(test_facts))
        right_MRR_all = right_MRR / float(len(test_facts))

        right_Hits_10_part23 = right_Hits_10 / float(test_number_23)
        right_Hits_3_part23 = right_Hits_3 / float(test_number_23)
        right_Hits_1_part23 = right_Hits_1 / float(test_number_23)
        right_MRR_part23 = right_MRR / float(test_number_23)

        if test_number_3 != 0:
            right_Hits_10_part3 = right_Hits_10 / float(test_number_3)
            right_Hits_3_part3 = right_Hits_3 / float(test_number_3)
            right_Hits_1_part3 = right_Hits_1 / float(test_number_3)
            right_MRR_part3 = right_MRR / float(test_number_3)
        else:
            right_Hits_10_part3 = 0.0
            right_Hits_3_part3 = 0.0
            right_Hits_1_part3 = 0.0
            right_MRR_part3 = 0.0
    else:
        right_Hits_10_part23 = 0.0
        right_Hits_3_part23 = 0.0
        right_Hits_1_part23 = 0.0
        right_MRR_part23 = 0.0

        right_Hits_10_all = 0.0
        right_Hits_3_all = 0.0
        right_Hits_1_all = 0.0
        right_MRR_all = 0.0

        right_Hits_10_part3 = 0.0
        right_Hits_3_part3 = 0.0
        right_Hits_1_part3 = 0.0
        right_MRR_part3 = 0.0
    # Save the results in file.
    # with open("{0}test_Pt_{1}_right.txt".format(lp_save_path, str(pt)), 'w') as f:
    #     for item in test_result_for_tail_prediction:
    #         f.write(str(item) + '\n')
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR))
    #
    #     f.write("For Pt: %d, Hits@10: %f\n" % (pt, right_Hits_10_all))
    #     f.write("For Pt: %d, Hits@3: %f\n" % (pt, right_Hits_3_all))
    #     f.write("For Pt: %d, Hits@1: %f\n" % (pt, right_Hits_1_all))
    #     f.write("For Pt: %d, MRR: %f\n" % (pt, right_MRR_all))
    print("For Pt: %d, Hits@10_part3: %f" % (pt, right_Hits_10_part3))
    print("For Pt: %d, Hits@3_part3: %f" % (pt, right_Hits_3_part3))
    print("For Pt: %d, Hits@1_part3: %f" % (pt, right_Hits_1_part3))
    print("For Pt: %d, MRR_part3: %f" % (pt, right_MRR_part3))

    print("For Pt: %d, Hits@10_part23: %f" % (pt, right_Hits_10_part23))
    print("For Pt: %d, Hits@3_part23: %f" % (pt, right_Hits_3_part23))
    print("For Pt: %d, Hits@1_part23: %f" % (pt, right_Hits_1_part23))
    print("For Pt: %d, MRR_part23: %f" % (pt, right_MRR_part23))

    print("For Pt: %d, Hits@10: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, Hits@3: %f" % (pt, right_Hits_3_all))
    print("For Pt: %d, Hits@1: %f" % (pt, right_Hits_10_all))
    print("For Pt: %d, MRR: %f" % (pt, right_MRR_all))

    """Predict the head entity given tail entity"""
    print("Step 2.2: Begin to test the head entity given tail entity.")
    left_Hits_10 = 0
    left_Hits_3 = 0
    left_Hits_1 = 0
    left_MRR = 0
    if len(test_tail_entity_dic) == len(test_tail_entity):
        print("Filter successfully.")
    else:
        print("NOT equally. If dic < tail entity, it's right.")
        # print("dic: %d" % len(test_entity_dic))
        # print("tail entity: %d" % len(test_tail_entity))
    print("For {0} tail entities, there are {1} can be predicted".
          format(len(test_tail_entity), len(test_tail_entity_dic)))

    # Calculate the MRR and Hit@10.
    # test_result_for_head_prediction = []
    test_number_23 = 0
    test_number_3 = 0
    for test_fact in test_facts:
        # print(test_fact)
        # t = [test_fact[0], test_fact[1]]
        if test_tail_entity_dic.get(test_fact[1]) is not None:
            test_number_23 += 1
            _temp = test_tail_entity_dic.get(test_fact[1])
            # e.g. _temp = [["t1", [0.9, 0.1]], ["t2", [0.9, 0.8]], ["t3", [0.8, 0.8]]]
            # 1. Filter setting (original)
            filter_entities = r_t_dic_h.get((pt, test_fact[1]))
            if filter_entities is not None:
                for row in _temp:
                    if row[0] in filter_entities:
                        _temp.remove(row)
            # 2. Filter setting (type)
            if x_entity_set is not None:
                for row in _temp:
                    if row[0] not in x_entity_set:
                        _temp.remove(row)
            _temp.sort(key=lambda x: x[1][0], reverse=True)
            head = [i[0] for i in _temp]
            t = [i[1][0] for i in _temp]
            rank_list = [t.index(i) + 1 for i in t]
            if test_fact[0] in head:
                test_number_3 += 1
                top = rank_list[head.index(test_fact[0])]
                left_MRR += 1 / float(top)
            else:
                # 0: predict left, 1: predict right.
                # notPredictedTestFacts_only12.append([0, test_fact])
                top = -1
        else:
            # notPredictedTestFacts_only1.append([0, test_fact])
            # notPredictedTestFacts_only12.append([0, test_fact])
            top = -1
        if 0 < top <= 10:
            left_Hits_10 += 1
        if 0 < top <= 3:
            left_Hits_3 += 1
        if top == 1:
            left_Hits_1 += 1
        # test_result_for_head_prediction.append([t, top, Hit, 1 / float(top)])
    if test_number_23 != 0:
        left_Hits_10_all = left_Hits_10 / float(len(test_facts))
        left_Hits_3_all = left_Hits_3 / float(len(test_facts))
        left_Hits_1_all = left_Hits_1 / float(len(test_facts))
        left_MRR_all = left_MRR / float(len(test_facts))

        left_Hits_10_part23 = left_Hits_10 / float(test_number_23)
        left_Hits_3_part23 = left_Hits_3 / float(test_number_23)
        left_Hits_1_part23 = left_Hits_1 / float(test_number_23)
        left_MRR_part23 = left_MRR / float(test_number_23)
        if test_number_3 != 0:
            left_Hits_10_part3 = left_Hits_10 / float(test_number_3)
            left_Hits_3_part3 = left_Hits_3 / float(test_number_3)
            left_Hits_1_part3 = left_Hits_1 / float(test_number_3)
            left_MRR_part3 = left_MRR / float(test_number_3)
        else:
            left_Hits_10_part3 = 0.0
            left_Hits_3_part3 = 0.0
            left_Hits_1_part3 = 0.0
            left_MRR_part3 = 0.0
    else:
        left_Hits_10_part23 = 0.0
        left_Hits_3_part23 = 0.0
        left_Hits_1_part23 = 0.0
        left_MRR_part23 = 0.0

        left_Hits_10_all = 0.0
        left_Hits_3_all = 0.0
        left_Hits_1_all = 0.0
        left_MRR_all = 0.0

        left_Hits_10_part3 = 0.0
        left_Hits_3_part3 = 0.0
        left_Hits_1_part3 = 0.0
        left_MRR_part3 = 0.0

    print("For Pt: %d, Hits@10_part3: %f" % (pt, left_Hits_10_part3))
    print("For Pt: %d, Hits@3_part3: %f" % (pt, left_Hits_3_part3))
    print("For Pt: %d, Hits@1_part3: %f" % (pt, left_Hits_1_part3))
    print("For Pt: %d, MRR_part3: %f" % (pt, left_MRR_part3))

    print("For Pt: %d, Hits@10_part23: %f" % (pt, left_Hits_10_part23))
    print("For Pt: %d, Hits@3_part23: %f" % (pt, left_Hits_3_part23))
    print("For Pt: %d, Hits@1_part23: %f" % (pt, left_Hits_1_part23))
    print("For Pt: %d, MRR_part23: %f" % (pt, left_MRR_part23))

    print("For Pt: %d, Hits@10_all: %f" % (pt, left_Hits_10_all))
    print("For Pt: %d, Hits@3_all: %f" % (pt, left_Hits_3_all))
    print("For Pt: %d, Hits@1_all: %f" % (pt, left_Hits_1_all))
    print("For Pt: %d, MRR_all: %f" % (pt, left_MRR_all))

    # # Save the notPredictedTestFacts in file.
    # with open("{0}notPredictedTestFacts_only1.txt".format(lp_save_path), 'a') as f:
    #     for item in notPredictedTestFacts_only1:
    #         f.write("{0}\t{1}\t{2}\t{3}\n".format(str(item[0]), str(item[1][0]), str(item[1][1]), str(item[1][2])))
    #     f.flush()
    # with open("{0}notPredictedTestFacts_only12.txt".format(lp_save_path), 'a') as f:
    #     for item in notPredictedTestFacts_only12:
    #         f.write("{0}\t{1}\t{2}\t{3}\n".format(str(item[0]), str(item[1][0]), str(item[1][1]), str(item[1][2])))
    #     f.flush()

    Hit_10_part3 = (left_Hits_10_part3 + right_Hits_10_part3) / 2
    Hit_3_part3 = (left_Hits_3_part3 + right_Hits_3_part3) / 2
    Hit_1_part3 = (left_Hits_1_part3 + right_Hits_1_part3) / 2
    MRR_part3 = (left_MRR_part3 + right_MRR_part3) / 2

    Hit_10_part23 = (left_Hits_10_part23 + right_Hits_10_part23) / 2
    Hit_3_part23 = (left_Hits_3_part23 + right_Hits_3_part23) / 2
    Hit_1_part23 = (left_Hits_1_part23 + right_Hits_1_part23) / 2
    MRR_part23 = (left_MRR_part23 + right_MRR_part23) / 2

    Hit_10_all = (left_Hits_10_all + right_Hits_10_all) / 2
    Hit_3_all = (left_Hits_3_all + right_Hits_3_all) / 2
    Hit_1_all = (left_Hits_1_all + right_Hits_1_all) / 2
    MRR_all = (left_MRR_all + right_MRR_all) / 2

    # return MRR, Hit_1, Hit_3, Hit_10
    return [right_MRR_part3, right_Hits_1_part3, right_Hits_3_part3, right_Hits_10_part3,
            left_MRR_part3, left_Hits_1_part3, left_Hits_3_part3, left_Hits_10_part3,
            MRR_part3, Hit_1_part3, Hit_3_part3, Hit_10_part3,

            right_MRR_part23, right_Hits_1_part23, right_Hits_3_part23, right_Hits_10_part23,
            left_MRR_part23, left_Hits_1_part23, left_Hits_3_part23, left_Hits_10_part23,
            MRR_part23, Hit_1_part23, Hit_3_part23, Hit_10_part23,

            right_MRR_all, right_Hits_1_all, right_Hits_3_all, right_Hits_10_all,
            left_MRR_all, left_Hits_1_all, left_Hits_3_all, left_Hits_10_all,
            MRR_all, Hit_1_all, Hit_3_all, Hit_10_all]
