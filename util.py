import os
import numpy as np
import sys


def read_data(filename, file_type=""):  # index from 0
    # read the Fact.txt: h t r
    if file_type == "train":
        with open("{0}train/Fact.txt".format(filename), 'r') as f:
            factSize = int(f.readline())
            facts = np.array([line.strip('\n').split('\t') for line in f.readlines()], dtype='int32')
            print("Total %s facts:%d" % (file_type, factSize))
        with open('{0}entity2id.txt'.format(filename), 'r') as f:
            entity_size = int(f.readline())
            ent = [line.strip('\n').split("\t")[0] for line in f.readlines()]
            print("Total entities:%d" % entity_size)
        with open("{0}relation2id.txt".format(filename)) as f:
            preSize = int(f.readline())
            pre = [line.strip('\n').split("\t")[0] for line in f.readlines()]
            print("Total predicates:%d" % preSize)
        # with open("{0}type2id.txt".format(filename)) as f:
        # with open("{0}type2id_one.txt".format(filename)) as f:
        # with open("{0}type2id_filter.txt".format(filename)) as f:
        if not os.path.exists("{0}type2id.txt".format(filename)):
            print("{0}type2id.txt".format(filename))
            typ = None
            print("No type information.")
        else:
            with open("{0}type2id.txt".format(filename)) as f:
                typeSize = int(f.readline())
                typ = [line.strip('\n').split("	")[0] for line in f.readlines()]
                print("Total types:%d" % typeSize)
        # if Max_body_rule_length > 2:
        #     with open("{0}type2id_one.txt".format(filename)) as f:
        #         typeSize_one = int(f.readline())
        #         typ_one = [line.strip('\n').split("	")[0] for line in f.readlines()]
        #         print("Total types:%d" % typeSize)
        #     return facts, ent, pre, typ, typ_one
        return facts, ent, pre, typ
    else:
        # file_type == "test" or "valid"
        with open("{0}{1}/Fact.txt".format(filename, file_type), 'r') as f:
            factSize = int(f.readline())
            facts = np.array([line.strip('\n').split('\t') for line in f.readlines()], dtype='int32')
            print("Total %s facts:%d\n" % (file_type, factSize))
        return facts


def read_relation_type(filename):
    pre_dom_type = {}
    pre_ran_type = {}
    pre_dom_type_one = {}
    pre_ran_type_one = {}
    # key: pre_index value:type_index list
    # with open("{0}relation_domain_type_one.txt".format(filename)) as f:
    # with open("{0}relation_domain_type.txt".format(filename)) as f:
    # with open("{0}relation_domain_type_filter.txt".format(filename)) as f:
    with open("{0}relation_domain_type.txt".format(filename)) as f:
        preSize = int(f.readline())
        for line in f.readlines():
            _tempList = line.strip('\n').split("\t")
            pre_dom_type[int(_tempList[0])] = list(map(int, _tempList[1:]))
    # with open("{0}relation_range_type_one.txt".format(filename)) as f:
    # with open("{0}relation_range_type.txt".format(filename)) as f:
    # with open("{0}relation_range_type_filter.txt".format(filename)) as f:
    with open("{0}relation_range_type.txt".format(filename)) as f:
        preSize = int(f.readline())
        for line in f.readlines():
            _tempList = line.strip('\n').split("\t")
            pre_ran_type[int(_tempList[0])] = list(map(int, _tempList[1:]))
    # if Max_body_rule_length > 2:
    #     with open("{0}relation_domain_type_one.txt".format(filename)) as f:
    #         preSize_one = int(f.readline())
    #         for line in f.readlines():
    #             _tempList = line.strip('\n').split("\t")
    #             pre_dom_type_one[int(_tempList[0])] = list(map(int, _tempList[1:]))
    #     with open("{0}relation_range_type_one.txt".format(filename)) as f:
    #         preSize_one = int(f.readline())
    #         for line in f.readlines():
    #             _tempList = line.strip('\n').split("\t")
    #             pre_ran_type_one[int(_tempList[0])] = list(map(int, _tempList[1:]))
    #     return pre_dom_type, pre_ran_type, pre_dom_type_one, pre_ran_type_one
    return pre_dom_type, pre_ran_type


def get_pre_fact_dic_all(facts_all):
    # fact_dic: key: P_index, value: all_fact_list
    fact_dic = {}
    for f in facts_all:
        if f[2] in fact_dic.keys():
            temp_list = fact_dic.get(f[2])
        else:
            temp_list = []
        temp_list.append([f[0], f[1]])
        fact_dic[f[2]] = temp_list
    return fact_dic


def get_filter_triple(facts):
    h_r_dic_t = {}
    r_t_dic_h = {}
    for f in facts:
        # h_r_dic_t
        if (f[0], f[2]) in h_r_dic_t.keys():
            tempList = h_r_dic_t.get((f[0], f[2]))
        else:
            tempList = []
        tempList.append(f[1])
        h_r_dic_t[(f[0], f[2])] = tempList
        # r_t_dic_h
        if (f[2], f[1]) in r_t_dic_h.keys():
            tempList = r_t_dic_h.get((f[2], f[1]))
        else:
            tempList = []
        tempList.append(f[0])
        r_t_dic_h[(f[2], f[1])] = tempList
    return h_r_dic_t, r_t_dic_h


def save_rules(save_path, Pt, candidate_of_Pt, pre_name):
    R_num = 0
    QR_num = 0
    typeR_num = 0
    if len(candidate_of_Pt) == 0:
        return R_num, QR_num, typeR_num

    with open('{0}rule_{1}.txt'.format(str(save_path), str(Pt)), 'w') as f:
        R_num = len(candidate_of_Pt)
        # [index, result, degree]
        for row in candidate_of_Pt:
            result = row[1]
            if result == 2:
                QR_num += 1
            index = row[0]
            index_line = ""
            for i in index:
                index_line += str(i)
                index_line += ','
            index_line.rstrip(',')
            # index_line = index_line.rstrip(',')
            degree = row[2]
            degree_line = ""
            for d in degree:
                degree_line += str(d)
                degree_line += ','
            degree_line.rstrip(',')
            # degree_line = degree_line.rstrip(',')

            if row[3] != 0:
                typeR_num += 1
                degree_type = row[4]
                degree_line_type = ""
                for d in degree_type:
                    degree_line_type += str(d)
                    degree_line_type += ','
                degree_line_type.rstrip(',')
                # degree_line_type = degree_line_type.rstrip(',')
                line = "{0}\t{1}\t{2}\n".format(index_line, degree_line, degree_line_type)
            else:
                line = "{0}\t{1}\n".format(index_line, degree_line)
            f.write(line)
        f.flush()
    return R_num, QR_num, typeR_num


def save_rules_for_names(save_path, Pt, candidate_of_Pt, pre_name):
    R_num = 0
    QR_num = 0
    typeR_num = 0
    if len(candidate_of_Pt) == 0:
        return R_num, QR_num, typeR_num

    with open('{0}rule_{1}.txt'.format(str(save_path), str(Pt)), 'w') as f:
        R_num = len(candidate_of_Pt)
        pre_size_all = len(pre_name)
        # [index, result, degree]
        for row in candidate_of_Pt:
            result = row[1]
            if result == 2:
                QR_num += 1
            index = row[0]
            index_line = ""
            for i in index:
                inverse_flag = False
                if i >= pre_size_all:
                    i = i - pre_size_all
                    inverse_flag = True
                if inverse_flag:
                    index_line += pre_name[i] + "^-1"
                else:
                    index_line += pre_name[i]
                index_line += ','
            index_line.rstrip(',')
            # index_line = index_line.rstrip(',')
            degree = row[2]
            degree_line = ""
            for d in degree:
                degree_line += str(d)
                degree_line += ','
            degree_line.rstrip(',')
            # degree_line = degree_line.rstrip(',')

            if row[3] != 0:
                typeR_num += 1
                degree_type = row[4]
                degree_line_type = ""
                for d in degree_type:
                    degree_line_type += str(d)
                    degree_line_type += ','
                degree_line_type.rstrip(',')
                # degree_line_type = degree_line_type.rstrip(',')
                line = "{0}\t{1}\t{2}\n".format(index_line, degree_line, degree_line_type)
            else:
                line = "{0}\t{1}\n".format(index_line, degree_line)
            f.write(line)
        f.flush()
    return R_num, QR_num, typeR_num


def read_rules(save_path, pt):
    candidate_of_Pt = []
    # [index:0,1 \t degree:0.9,0.8 \t degree_type:0.9,0.8 \n]
    if not os.path.exists('{0}rule_{1}.txt'.format(str(save_path), str(pt))):
        print("Sorry, the file doesn't exist.")
        return None
    with open('{0}rule_{1}.txt'.format(str(save_path), str(pt)), 'r') as f:
        for line in f.readlines():
            _tempList = line.strip('\n').split("\t")
            _index = _tempList[0].split(',')
            # _index.pop()
            index = [int(item) for item in _index]
            _degree = _tempList[1].split(',')
            _degree.pop()
            degree = [float(item) for item in _degree]
            if len(_tempList) > 2:
                _degree_type = _tempList[2].split(',')
                _degree_type.pop()
                degree_type = [float(item) for item in _degree_type]
                candidate_of_Pt.append([index, degree, degree_type])
            else:
                candidate_of_Pt.append([index, degree])
    return candidate_of_Pt


def read_middle_results(pt, model, BENCHMARK, filetype):
    middle_results_save_path = "./linkprediction/{0}/{1}".format(BENCHMARK, filetype)
    pt_test_head_dic = {}
    pt_test_tail_dic = {}
    if not os.path.exists("{0}/{1}/r_{2}_right.txt".format(middle_results_save_path, model, str(pt))) and \
            filetype == "valid":
        return pt_test_head_dic, pt_test_tail_dic
    with open("{0}/{1}/r_{2}_right.txt".format(middle_results_save_path, model, str(pt)), 'r') as f:
        for line in f.readlines():
            _tempList = line.strip('\n').split("\t")
            h = int(_tempList[0])
            if _tempList[1] != 'None':
                _t_s = _tempList[1].split(',')
                t_s = []
                for _pair in _t_s:
                    # print(_pair)
                    pair = _pair.split(':')
                    # print(pair)
                    t = int(pair[0])
                    s = float(pair[1])
                    t_s.append([t, s])
            else:
                t_s = None
            pt_test_head_dic[h] = t_s
    with open("{0}/{1}/r_{2}_left.txt".format(middle_results_save_path, model, str(pt)), 'r') as f:
        for line in f.readlines():
            _tempList = line.strip('\n').split("\t")
            t = int(_tempList[0])
            if _tempList[1] != 'None':
                _h_s = _tempList[1].split(',')
                h_s = []
                for _pair in _h_s:
                    pair = _pair.split(':')
                    h = int(pair[0])
                    s = float(pair[1])
                    h_s.append([h, s])
            else:
                h_s = None
            pt_test_tail_dic[t] = h_s
    return pt_test_head_dic, pt_test_tail_dic


def get_type_entities(filename):
    type_entities_dic = {}
    with open("{0}type_entity.txt".format(filename)) as f:
        typeSize = int(f.readline())
        for line in f.readlines():
            _tempList = line.strip('\n').split("\t")
            type_entities_dic[int(_tempList[0])] = list(map(int, _tempList[1:]))
    return type_entities_dic


def mark_and_count_type_of_rules(pt, candidate, pre_dom_type, pre_ran_type, allRelationSize, type_name):
    candidateWithType = []
    for rule in candidate:
        # candidate: [index, result, degree]
        index = rule[0]
        degree = rule[2]

        # print("For rule :{0} -> {1}".format(str(index), str(pt)))

        pt_dom_types = set(pre_dom_type.get(pt))
        pt_ran_types = set(pre_ran_type.get(pt))

        TYPE = set()
        if index[0] >= allRelationSize:
            p1_x = set(pre_ran_type.get(index[0]-allRelationSize))
        else:
            p1_x = set(pre_dom_type.get(index[0]))
        if index[-1] >= allRelationSize:
            pn_y = set(pre_dom_type.get(index[-1]-allRelationSize))
        else:
            pn_y = set(pre_ran_type.get(index[-1]))
        x = pt_dom_types & p1_x
        # print("X  %d:  %s" % (len(x), x))
        y = pt_ran_types & pn_y
        # print("Y  %d:  %s" % (len(y), y))
        arg_type = [list(x)]
        for t in x:
            TYPE.add(t)
        for t in y:
            TYPE.add(t)

        # for z_i
        if len(index) >= 2:
            for i in range(0, len(index)-1):
                if index[i] >= allRelationSize:
                    z_i_left = set(pre_dom_type.get(index[i]-allRelationSize))
                else:
                    z_i_left = set(pre_ran_type.get(index[i]))
                if index[i+1] >= allRelationSize:
                    z_i_right = set(pre_ran_type.get(index[i + 1]-allRelationSize))
                else:
                    z_i_right = set(pre_dom_type.get(index[i+1]))
                z_i = z_i_left & z_i_right
                # print("Z_%d  %d:  %s" % (i, len(z_i), z_i))
                arg_type.append(list(z_i))
                for t in z_i:
                    TYPE.add(t)
        arg_type.append(list(y))
        # print("The number of types: %d" % len(TYPE))
        # print("\n")
        candidateWithType.append([index, degree, arg_type])
    return candidateWithType


def mark_and_count_type_of_rules_2(BENCHMARK, pt, candidate, pre_dom_type, pre_ran_type, allRelationSize, type_name):
    x_type_set = set()
    y_type_set = set()
    for rule in candidate:
        # candidate: [index, result, degree]
        index = rule[0]

        # print("For rule :{0} -> {1}".format(str(index), str(pt)))

        pt_dom_types = set(pre_dom_type.get(pt))
        pt_ran_types = set(pre_ran_type.get(pt))

        # TYPE = set()
        if index[0] >= allRelationSize:
            p1_x = set(pre_ran_type.get(index[0]-allRelationSize))
        else:
            p1_x = set(pre_dom_type.get(index[0]))
        if index[-1] >= allRelationSize:
            pn_y = set(pre_dom_type.get(index[-1]-allRelationSize))
        else:
            pn_y = set(pre_ran_type.get(index[-1]))
        x = pt_dom_types & p1_x
        # print("X  %d:  %s" % (len(x), x))
        y = pt_ran_types & pn_y
        # print("Y  %d:  %s" % (len(y), y))

        for t_x in x:
            x_type_set.add(t_x)
        for t_y in y:
            y_type_set.add(t_y)
    # Get type2entity key:type_index value:entities_index
    type_entities_dic = get_type_entities(filename="./benchmarks/{0}/".format(BENCHMARK))
    x_entity_set = set()
    y_entity_set = set()
    for t_x in x_type_set:
        for e in type_entities_dic.get(t_x):
            x_entity_set.add(e)
    for t_y in y_type_set:
        for e in type_entities_dic.get(t_y):
            y_entity_set.add(e)
    return x_entity_set, y_entity_set
