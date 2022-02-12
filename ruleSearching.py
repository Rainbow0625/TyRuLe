import time
import numpy as np
from scipy import sparse
import itertools
import gc
import sys
import torch as torch
import util as util

import linkPrediction as lp

# from multiprocessing import Pool
import multiprocessing.dummy as mp


class RuleSearchingAndEvaluating(object):
    def __int__(self, DEGREE, pre_fact_dic_all, allEntitySize, allRelationSize,
                pre_dom_type, pre_ran_type, BENCHMARK):
        self.DEGREE = DEGREE
        self.length = None
        self.pt = None
        self.tempCandidate = None
        self.ent_size_all = allEntitySize
        self.pre_size_all = allRelationSize
        self.ent_size_sample = None
        self.pre_fact_dic_all = pre_fact_dic_all
        self.pre_fact_dic_sample = None

        # todo: ent_emb, rel_emb, _syn, _coocc, isfullKG, E_0
        self.ent_emb = None
        self.rel_emb = None
        self._syn = None
        self._coocc = None
        self.ptmatrix_part = None
        self.ptmatrix_full = None
        self.ptmatrix_sample = None
        self.E_0 = None

        self.pre_dom_type = pre_dom_type
        self.pre_ran_type = pre_ran_type
        self.BENCHMARK = BENCHMARK
    # def __int__(self, DEGREE, pre_fact_dic_all, allEntitySize, allRelationSize):
    #     self.DEGREE = DEGREE
    #     self.length = None
    #     self.pt = None
    #     self.tempCandidate = None
    #     self.ent_size_all = allEntitySize
    #     self.pre_size_all = allRelationSize
    #     self.ent_size_sample = None
    #     self.pre_fact_dic_all = pre_fact_dic_all
    #     self.pre_fact_dic_sample = None
    #
    #     # todo: ent_emb, rel_emb, _syn, _coocc, isfullKG, E_0
    #     self.ent_emb = None
    #     self.rel_emb = None
    #     self._syn = None
    #     self._coocc = None
    #     self.ptmatrix_part = None
    #     self.ptmatrix_full = None
    #     self.ptmatrix_sample = None
    #     self.E_0 = None

    @staticmethod
    def sim(para1, para2):  # similarity of vector or matrix
        return np.e**(-np.linalg.norm(para1 - para2, ord=2))

    @staticmethod
    def get_pre_fact_dic_sample(facts_sample):
        fact_dic = {}
        for f in facts_sample:
            if f[2] in fact_dic.keys():
                templist = fact_dic.get(f[2])
            else:
                templist = []
            templist.append([f[0], f[1]])
            fact_dic[f[2]] = templist
        return fact_dic

    def is_repeated(self, M_index):
        for i in range(1, self.length):
            if M_index[i] < M_index[0]:
                return True
        return False

    def get_sub_and_obj_dic(self):
        subdic = {}  # key:predicate value: head_set
        objdic = {}  # key:predicate value: tail_set
        for key in self.pre_fact_dic_sample.keys():
            tempsub = set()
            tempobj = set()
            facts_list = self.pre_fact_dic_sample.get(key)
            for f in facts_list:
                tempsub.add(f[0])
                tempobj.add(f[1])
            subdic[key] = tempsub
            objdic[key] = tempobj
        return subdic, objdic

    def getOneHotEncoder(self, integer_encoded):
        integer_encoded = torch.from_numpy(integer_encoded)
        onehot_encoded = torch.nn.functional.one_hot(integer_encoded, self.ent_size_all)
        onehot_encoded = onehot_encoded.numpy()
        sum_average_encoded = np.sum(onehot_encoded, axis=0) / len(integer_encoded)
        return sum_average_encoded

    # Todo
    def score_function1(self, flag, score_top_container):  # synonymy!
        relation = self.rel_emb
        for index in self.index_tuple:
            M = [relation[i] for i in index]
            # print(M)
            if flag == 0:  # matrix
                # array
                result = np.linalg.multi_dot(M)
            else:  # vector
                if self.is_repeated(index):
                    continue
                else:
                    result = sum(M)
            top_values = score_top_container[:, self.length]
            value = self.sim(result, relation[self.pt])

            if value > np.min(top_values):
                replace_index = np.argmin(top_values)
                for i in range(self.length):
                    score_top_container[replace_index][i] = index[i]
                score_top_container[replace_index][self.length] = value
                # print(score_top_container[replace_index])
        return score_top_container

    # Todo
    def score_function2(self, score_top_container, sub_dic, obj_dic):  # co-occurrence
        tt = time.time()
        entity = self.ent_emb
        # get the average vector of average predicate which is saved in the dictionary.
        average_vector = {}
        for key in sub_dic:
            # print(key)
            sub = sum([entity[item, :] for item in sub_dic[key]]) / len(sub_dic[key])
            obj = sum([entity[item, :] for item in obj_dic[key]]) / len(obj_dic[key])
            # For predicates: 0, 2, 4, ... [sub, obj]
            # For predicates: 1, 3, 5, ... [obj, sub]
            average_vector[key] = [sub, obj]
            average_vector[key + 1] = [obj, sub]
        # print("\n the dic's size is equal to the predicates' number! ")
        # print(len(average_vector))
        f = 0
        for index in self.index_tuple:
            f = f + 1
            if self.length == 2 and index[0] == self.pt and index[1] == index[0]:
                continue
            elif self.length == 2 and index[0] == self.pt + 1 and index[1] == index[0]:
                continue
            elif self.length == 2 and index[0] % 2 == 0 and index[1] == index[0] + 1:
                continue
            elif self.length == 2 and index[1] % 2 == 0 and index[0] == index[1] + 1:
                continue
            para_sum = float(0)
            for i in range(self.length - 1):
                para_sum = para_sum + self.sim(average_vector.get(index[i])[1], average_vector.get(index[i + 1])[0])
            value = para_sum + self.sim(average_vector.get(index[0])[0], average_vector.get(self.pt)[0]) \
                    + self.sim(average_vector.get(index[self.length - 1])[1],
                               average_vector.get(self.pt)[1])
            top_values = score_top_container[:, self.length]
            if value > np.min(top_values):
                replace_index = np.argmin(top_values)
                for i in range(self.length):
                    score_top_container[replace_index][i] = index[i]
                score_top_container[replace_index][self.length] = value
        print('Progress: %d - %d ' % (f, self.index_tuple_size))
        print("Time: %f." % (time.time() - tt))
        return score_top_container

    def scoreFunction3(self, score_top_container, sub_dic, obj_dic):
        tt = time.time()
        # Get the average sum vector of average argument entities which is saved in the dictionary.
        # key: pre_index  value:[sub_array, obj_array]
        average_vector = {}
        for key in sub_dic.keys():
            # print(key)
            sub = self.getOneHotEncoder(np.array(list(sub_dic[key]), dtype=np.int64))
            obj = self.getOneHotEncoder(np.array(list(obj_dic[key]), dtype=np.int64))
            average_vector[key] = [sub, obj]
            average_vector[key + self.pre_size_all] = [obj, sub]
        # Todo：For test
        # print("\n the dic's size is equal to the predicates' number! ")
        # print(len(average_vector))
        f = 0
        for index in self.tempCandidate:
            f = f + 1
            if self.length == 1 and index == self.pt:
                continue
            if self.length == 1:
                value = RuleSearchingAndEvaluating.sim(average_vector.get(index)[0], average_vector.get(self.pt)[0]) \
                        + RuleSearchingAndEvaluating.sim(average_vector.get(index)[1], average_vector.get(self.pt)[1])
            else:
                para_sum = float(0)
                for i in range(self.length - 1):
                    para_sum = para_sum + RuleSearchingAndEvaluating.sim(average_vector.get(index[i])[1], average_vector.get(index[i + 1])[0])
                value = para_sum + RuleSearchingAndEvaluating.sim(average_vector.get(index[0])[0], average_vector.get(self.pt)[0]) \
                        + RuleSearchingAndEvaluating.sim(average_vector.get(index[self.length - 1])[1], average_vector.get(self.pt)[1])
            top_values = score_top_container[:, self.length]
            if value > np.min(top_values):
                replace_index = np.argmin(top_values)
                if self.length == 1:
                    score_top_container[replace_index][0] = index
                else:
                    for i in range(self.length):
                        score_top_container[replace_index][i] = index[i]
                score_top_container[replace_index][self.length] = value
        print('Progress: %d - %d ' % (f, len(self.tempCandidate)))
        print("Time: %f." % (time.time() - tt))
        return score_top_container

    # def scoreFunction4(self, score_top_container):
    #     tt = time.time()
    #     # 1
    #     relation = self.rel_emb
    #     for index in self.index_tuple:
    #         M = [relation[i] for i in index]
    #         # print(M)
    #         if flag == 0:  # matrix
    #             # array
    #             result = np.linalg.multi_dot(M)
    #         else:  # vector
    #             if self.is_repeated(index):
    #                 continue
    #             else:
    #                 result = sum(M)
    #         top_values = score_top_container[:, self.length]
    #         value = self.sim(result, relation[self.pt])
    #
    #         if value > np.min(top_values):
    #             replace_index = np.argmin(top_values)
    #             for i in range(self.length):
    #                 score_top_container[replace_index][i] = index[i]
    #             score_top_container[replace_index][self.length] = value
    #             # print(score_top_container[replace_index])
    #
    #     # 2
    #     # Get the average sum vector of average argument entities which is saved in the dictionary.
    #     # key: pre_index  value:[sub_array, obj_array]
    #     average_vector = {}
    #     for key in sub_dic.keys():
    #         # print(key)
    #         sub = self.getOneHotEncoder(np.array(list(sub_dic[key]), dtype=np.int64))
    #         obj = self.getOneHotEncoder(np.array(list(obj_dic[key]), dtype=np.int64))
    #         average_vector[key] = [sub, obj]
    #         average_vector[key + self.pre_size_all] = [obj, sub]
    #     # Todo：For test
    #     # print("\n the dic's size is equal to the predicates' number! ")
    #     # print(len(average_vector))
    #     f = 0
    #     for index in self.tempCandidate:
    #         f = f + 1
    #         if self.length == 1 and index == self.pt:
    #             continue
    #         if self.length == 1:
    #             value = RuleSearchingAndEvaluating.sim(average_vector.get(index)[0], average_vector.get(self.pt)[0]) \
    #                     + RuleSearchingAndEvaluating.sim(average_vector.get(index)[1], average_vector.get(self.pt)[1])
    #         else:
    #             para_sum = float(0)
    #             for i in range(self.length - 1):
    #                 para_sum = para_sum + RuleSearchingAndEvaluating.sim(average_vector.get(index[i])[1], average_vector.get(index[i + 1])[0])
    #             value = para_sum + RuleSearchingAndEvaluating.sim(average_vector.get(index[0])[0], average_vector.get(self.pt)[0]) \
    #                     + RuleSearchingAndEvaluating.sim(average_vector.get(index[self.length - 1])[1], average_vector.get(self.pt)[1])
    #         top_values = score_top_container[:, self.length]
    #         if value > np.min(top_values):
    #             replace_index = np.argmin(top_values)
    #             if self.length == 1:
    #                 score_top_container[replace_index][0] = index
    #             else:
    #                 for i in range(self.length):
    #                     score_top_container[replace_index][i] = index[i]
    #             score_top_container[replace_index][self.length] = value
    #     print('Progress: %d - %d ' % (f, len(self.tempCandidate)))
    #     print("Time: %f." % (time.time() - tt))
    #     return score_top_container

    def getmatrix(self, p, isWhichKG):
        # sparse matrix
        inverse_flag = False
        # if p == self.pt:
        #     if isWhichKG == 0:
        #         if self.ptmatrix_sample != None:
        #             return self.ptmatrix_sample
        #     elif isWhichKG == 1:
        #         if self.ptmatrix_part != None:
        #             return self.ptmatrix_part
        #     elif isWhichKG == 2:
        #         if self.ptmatrix_full != None:
        #             return self.ptmatrix_full
        if p >= self.pre_size_all:
            p = p - self.pre_size_all
            inverse_flag = True
        # Pt: avoid cal it again.
        if isWhichKG == 0:
            pfacts = self.pre_fact_dic_sample.get(p)
            pmatrix = sparse.dok_matrix((self.ent_size_sample, self.ent_size_sample), dtype=np.int8)
            if inverse_flag:
                for f in pfacts:
                    pmatrix[f[1], f[0]] = 1
            else:
                for f in pfacts:
                    pmatrix[f[0], f[1]] = 1
        elif isWhichKG == 1:  # Evaluate on Pt's entity one-hot matrix.
            pfacts = self.pre_fact_dic_all.get(p)
            ent_size = len(self.E_0)
            E_0_list = list(self.E_0)
            pmatrix = sparse.dok_matrix((ent_size, ent_size), dtype=np.int8)
            for f in pfacts:
                if f[0] in self.E_0 and f[1] in self.E_0:
                    if inverse_flag:
                        pmatrix[E_0_list.index(f[1]), E_0_list.index(f[0])] = 1
                    else:
                        pmatrix[E_0_list.index(f[0]), E_0_list.index(f[1])] = 1
        else:
            pfacts = self.pre_fact_dic_all.get(p)
            pmatrix = sparse.dok_matrix((self.ent_size_all, self.ent_size_all), dtype=np.int8)
            if inverse_flag:
                for f in pfacts:
                    pmatrix[f[1], f[0]] = 1
            else:
                for f in pfacts:
                    pmatrix[f[0], f[1]] = 1
        return pmatrix

    def getmatrixByType(self, p, p_dom_types, p_ran_types):
        # sparse matrix
        inverse_flag = False
        if p >= self.pre_size_all:
            p = p - self.pre_size_all
            inverse_flag = True
        # Get type information for calculating the SC_type and HC_type.
        pfacts = self.pre_fact_dic_all.get(p)

        p_dom_entities = []
        for t in p_dom_types:
            if self.type_entities_dic.get(t) is not None:
                p_dom_entities.extend(self.type_entities_dic.get(t))

        p_ran_entities = []
        for t in p_ran_types:
            if self.type_entities_dic.get(t) is not None:
                p_ran_entities.extend(self.type_entities_dic.get(t))
        # Pt: avoid cal it again.
        pmatrix_type = sparse.dok_matrix((self.ent_size_all, self.ent_size_all), dtype=np.int8)
        if inverse_flag:
            for f in pfacts:
                if f[0] in p_dom_entities and f[1] in p_ran_entities:
                    pmatrix_type[f[1], f[0]] = 1
        else:
            for f in pfacts:
                if f[0] in p_dom_entities and f[1] in p_ran_entities:
                    pmatrix_type[f[0], f[1]] = 1
        return pmatrix_type

    # def calSupportGreaterThanOne(self, pmatrix, isSampleKG):
    #     if isSampleKG:
    #         ptmatrix = self.ptmatrix_sample
    #     else:
    #         ptmatrix = self.ptmatrix_part
    #     supp = 0
    #     head = len(ptmatrix)
    #     body = len(pmatrix)
    #     if head == 0 or body == 0:
    #         return False
    #     if head < body:
    #         for key in ptmatrix.keys():
    #             if pmatrix.get(key) > 0:
    #                 supp += 1
    #                 return True
    #     if head >= body:
    #         for key in pmatrix.keys():
    #             if ptmatrix.get(key) == 1:
    #                 supp += 1
    #                 return True
    #     return False

    # def calSCandHC_csr(self, pmatrix):
    #     print("\n---------------csr---------------\n")
    #     # calculate New SC
    #     # supp_score = 0.0
    #     # body_score = 0.0
    #     ptmatrix = self.ptmatrix_full
    #     head = len(ptmatrix)
    #     body = pmatrix.nnz
    #     supp = 0
    #     if head == 0 or body == 0:
    #         return 0, 0
    #     row_compress = pmatrix.indptr
    #     col = pmatrix.indices
    #     # print(pmatrix.nnz)
    #     # print(sys.getsizeof(pmatrix))
    #     flag = 0
    #     for i in range(pmatrix.shape[0]):
    #         row_num = row_compress[i + 1] - row_compress[i]
    #         if row_num == 0:
    #             continue
    #         row_col = col[flag: flag + row_num]
    #         for j in range(row_num):
    #             if ptmatrix.get(tuple([i, row_col[j]])) == 1:
    #                 supp = supp + 1
    #         flag += row_num
    #     # Judge by supp.
    #     if body == 0:
    #         SC = 0
    #     else:
    #         SC = supp / body
    #     if head == 0:
    #         HC = 0
    #     else:
    #         HC = supp / head
    #     return SC, HC
    def matrix_dot(self, index, isWhichKG):  # 0:sample 1:part 2:full;
        pmatrix = self.getmatrix(index[0], isWhichKG)
        for i in range(1, self.length):
            # Matrix distribution law
            a = self.getmatrix(index[i], isWhichKG)
            if len(a.keys()) >= 808073:
                print("pass")
                pmatrix = sparse.dok_matrix((self.ent_size_all, self.ent_size_all), dtype=np.int8)
            else:
                pmatrix = pmatrix.dot(a)
            if not gc.isenabled():
                gc.enable()
            gc.collect()
            gc.disable()
        return pmatrix

    def matrix_dot_type(self, index, arg_type):  # 0:sample 1:part 2:full;
        pmatrix = self.getmatrixByType(index[0], arg_type[0], arg_type[1])
        for i in range(1, self.length):
            # Matrix distribution law
            a = self.getmatrixByType(index[i], arg_type[i], arg_type[i+1])
            if len(a.keys()) >= 808073:
                print("pass")
                pmatrix = sparse.dok_matrix((self.ent_size_all, self.ent_size_all), dtype=np.int8)
            else:
                pmatrix = pmatrix.dot(a)
            if not gc.isenabled():
                gc.enable()
            gc.collect()
            gc.disable()
        pmatrix = pmatrix.todok()
        return pmatrix

    def calSCandHC_dok(self, pmatrix):
        ptmatrix = self.ptmatrix_full

        head = len(ptmatrix)
        body = len(pmatrix)
        supp = 0
        # calculate New SC
        # supp_score = 0.0
        # body_score = 0.0
        if head == 0 or body == 0:
            return 0, 0
        if head < body:
            for key in ptmatrix.keys():
                if pmatrix.get(key) > 0:
                    supp = supp + 1
        elif head >= body:
            for key in pmatrix.keys():
                if ptmatrix.get(key) == 1:
                    supp = supp + 1
        # Judge by supp.
        if body == 0:
            SC = 0
        else:
            SC = supp / body
        if head == 0:
            HC = 0
        else:
            HC = supp / head
        return [SC, HC]

    def get_rule_args_type(self, index):
        # index:[p1, p2] pt: pt
        pt_dom_types = set(self.pre_dom_type.get(self.pt))
        pt_ran_types = set(self.pre_ran_type.get(self.pt))

        if index[0] >= self.pre_size_all:
            p1_x = set(self.pre_ran_type.get(index[0] - self.pre_size_all))
        else:
            p1_x = set(self.pre_dom_type.get(index[0]))
        if index[-1] >= self.pre_size_all:
            pn_y = set(self.pre_dom_type.get(index[-1] - self.pre_size_all))
        else:
            pn_y = set(self.pre_ran_type.get(index[-1]))
        x = pt_dom_types & p1_x
        # print("X  %d:  %s" % (len(x), x))
        y = pt_ran_types & pn_y
        # print("Y  %d:  %s" % (len(y), y))
        arg_type = [list(x)]
        # for z_i
        if len(index) >= 2:
            for i in range(0, len(index) - 1):
                if index[i] >= self.pre_size_all:
                    z_i_left = set(self.pre_dom_type.get(index[i] - self.pre_size_all))
                else:
                    z_i_left = set(self.pre_ran_type.get(index[i]))
                if index[i + 1] >= self.pre_size_all:
                    z_i_right = set(self.pre_ran_type.get(index[i + 1] - self.pre_size_all))
                else:
                    z_i_right = set(self.pre_dom_type.get(index[i + 1]))
                z_i = z_i_left & z_i_right
                arg_type.append(list(z_i))
        arg_type.append(list(y))
        return arg_type

    def calSCandHC_dok_type(self, index, arg_type):

        ptmatrix = self.getmatrixByType(self.pt, arg_type[0], arg_type[-1])
        pmatrix = self.matrix_dot_type(index, arg_type)

        head = len(ptmatrix)
        body = len(pmatrix)
        supp = 0
        # calculate New SC
        # supp_score = 0.0
        # body_score = 0.0
        if head == 0 or body == 0:
            return 0, 0
        if head < body:
            for key in ptmatrix.keys():
                if pmatrix.get(key) > 0:
                    supp = supp + 1
        elif head >= body:
            for key in pmatrix.keys():
                if ptmatrix.get(key) == 1:
                    supp = supp + 1
        # Judge by supp.
        if body == 0:
            SC = 0
        else:
            SC = supp / body
        if head == 0:
            HC = 0
        else:
            HC = supp / head
        return [SC, HC]

    def evaluate_and_filter(self, index, isfullKG, type_flag=False):
        # if not isfullKG:
        #     if len(self.E_0) <= 50000:
        #         # On part. (Priority)
        #         pmatrix = self.matrix_dot(index, 1)
        #         pmatrix = pmatrix.todok()
        #         if not self.calSupportGreaterThanOne(pmatrix, False):
        #             return 0, None
        #     else:
        #         # On sample.
        #         pmatrix = self.matrix_dot(index, 0)
        #         pmatrix = pmatrix.todok()
        #         if not self.calSupportGreaterThanOne(pmatrix, True):
        #             return 0, None
        # On full.
        # calculate the temp SC and HC
        # if sys.getsizeof(pmatrix) > 10485760:  # 10M
        #     # Type of pmatrix:  csr_matrix!
        #     print(sys.getsizeof(pmatrix))
        #     print("Date size:")
        #     print("pmatrix len:%d" % pmatrix.nnz)
        #     if isfullKG:
        #         print("full:")
        #         print(pmatrix.nnz / self.ent_size_all ** 2)
        #     else:
        #         print(pmatrix.nnz / self.ent_size_sample ** 2)
        #     print("\n")
        #     SC, HC = self.calSCandHC_csr(pmatrix)
        #     degree = [SC, HC]
        # else:

        # Type of pmatrix:  dok_matrix!
        # print("Date size:")
        # print("pmatrix len:%d" % len(pmatrix))
        # if isfullKG:
        # print("full:")
        # print(len(pmatrix) / self.ent_size_all ** 2)
        # else:
        # print(len(pmatrix) / self.ent_size_sample ** 2)
        # print("\n")
        if type_flag:
            arg_type = self.get_rule_args_type(index)
            degree = self.calSCandHC_dok_type(index, arg_type)
        else:
            pmatrix = self.matrix_dot(index, 2)
            pmatrix = pmatrix.todok()
            degree = self.calSCandHC_dok(pmatrix)

        # print(degree)
        if degree[0] >= self.DEGREE[0] and degree[1] >= self.DEGREE[1]:
            # 1: quality rule
            # 2: high quality rule
            # print("\n%s - SC:%s, HC:%s." % (str(index), str(degree[0]), str(degree[1])))
            # print("\n%s - SC: %s, HC: %s; SC_type: %s, HC_type: %s." %
            #       (str(index), str(degree[0]), str(degree[1]), str(degree[2]), str(degree[3])))
            if type_flag:
                print("\n%s - SC_type: %s, HC_type: %s." % (str(index), str(degree[0]), str(degree[1])))
            else:
                print("\n%s - SC: %s, HC: %s." % (str(index), str(degree[0]), str(degree[1])))

            # print("The NEW Standard Confidence of this rule is " + str(NSC))
            if degree[0] >= self.DEGREE[2] and degree[1] >= self.DEGREE[3]:
                return 2, degree
            return 1, degree
        return 0, None

    def search_and_evaluate(self, length, Pt, tempCandidate, sampledFacts, sampledEntitySize, _coocc, isfullKG):
        self.length = length
        self.pt = Pt
        self.tempCandidate = tempCandidate
        self.pre_fact_dic_sample = RuleSearchingAndEvaluating.get_pre_fact_dic_sample(sampledFacts)
        self.ent_size_sample = sampledEntitySize

        # self._syn = _syn
        self._coocc = _coocc

        candidate = []

        """Calculate the Score Functions."""
        '''Calculate the f3.'''
        if len(self.tempCandidate) < _coocc:
            top_candidate_size = len(self.tempCandidate)
        else:
            top_candidate_size = _coocc
        # print("The number of New-COOCC Top Candidates is %d" % top_candidate_size)
        score_top_container = np.ones(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        score_top_container = - score_top_container
        subdic, objdic = self.get_sub_and_obj_dic()
        print("Begin to calculate the f3: New Co-occurrence")
        score_top_container = self.scoreFunction3(score_top_container, subdic, objdic)

        '''Calculate the f2.'''
        '''
        # top_candidate_size = int(_coocc * self.index_tuple_size)
        if len(self.tempCandidate) < _coocc:
            top_candidate_size = len(self.tempCandidate)
        else:
            top_candidate_size = _coocc
        print("The number of COOCC Top Candidates is %d" % top_candidate_size)
        score_top_container = np.zeros(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        subdic, objdic = self.get_sub_and_obj_dic()
        print("\nBegin to calculate the f2: Co-occurrence")
        score_top_container = self.score_function2(score_top_container, subdic, objdic)
        '''

        '''Calculate the f1.'''
        '''
        # top_candidate_size = int(_syn * self.index_tuple_size)
        if self.index_tuple_size < _syn:
            top_candidate_size = self.index_tuple_size
        else:
            top_candidate_size = _syn
        top_candidate_size = _syn
        score_top_container = np.zeros(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        print("The number of SYN Top Candidates is %d" % top_candidate_size)
        print("\nBegin to calculate the f1: synonymy")
        score_top_container = self.score_function1(f, score_top_container, rel_emb)
        # Method 1: Top ones until it reaches the 100th. OMIT!
        # Method 2: Use two matrices to catch rules.

        print("\n Begin to use syn to filter: ")
        for item in score_top_container:
            index = [int(item[i]) for i in range(self.length)]
            if f == 0:  # matrix
                result, degree = self.evaluate_and_filter(index, DEGREE)
                if result != 0 and index not in all_candidate_set:
                    all_candidate_set.append(index)
                    candidate.append([index, result, degree])
            elif f == 1:  # vector
                # It needs to evaluate for all arranges of index.
                for i in itertools.permutations(index, self.length):
                    # Deduplicate.
                    _index = list(np.array(i))
                    if _index in all_candidate_set:
                        continue
                    result, degree = self.evaluate_and_filter(_index, DEGREE, isfullKG)
                    if result != 0:
                        all_candidate_set.append(_index)
                        candidate.append([_index, result, degree])
        if not gc.isenabled():
            gc.enable()
        del rel_emb, score_top_container
        gc.collect()
        gc.disable()
        '''

        if not gc.isenabled():
            gc.enable()
        del subdic, objdic
        gc.collect()
        gc.disable()

        """Evaluate the rules in score_top_container."""
        # Get relation_type. pre_dom_type, pre_ran_type
        # Get type_entity.
        self.type_entities_dic = util.get_type_entities(filename="./benchmarks/{0}/".format(self.BENCHMARK))

        print("Get the Pt matrix before evaluating the rules:")
        # if not isfullKG:
        #     if len(self.E_0) > 50000:
        #         self.ptmatrix_sample = self.getmatrix(self.pt, 0)
        #         print(" Sample: len:%d  size:%d" % (len(self.ptmatrix_sample), sys.getsizeof(self.ptmatrix_sample)))
        #     else:
        #         self.ptmatrix_part = self.getmatrix(self.pt, 1)
        #         print(" Part: len:%d  size:%d" % (len(self.ptmatrix_part), sys.getsizeof(self.ptmatrix_part)))

        self.ptmatrix_full = self.getmatrix(self.pt, 2)
        # TODO!!!!!修改！！！！每个都是临时计算的！！！！
        # self.ptmatrix_full_type = self.getmatrixByType(self.pt, 2)
        print(" Full: len:%d  size:%d" % (len(self.ptmatrix_full), sys.getsizeof(self.ptmatrix_full)))

        print("Begin to evaluate the rules in score_top_container: ")
        count = 0
        tt = time.time()

        # # Multiprocessing
        # # pool = Pool(4)
        # pool = mp.Pool(4)
        # result_list = []
        # index_list = []
        # for item in score_top_container:
        #     count += 1
        #     index = [int(item[i]) for i in range(self.length)]
        #     if item[-1] == -1:
        #         continue
        #     if self.length == 1 and index == [self.pt]:
        #         continue
        #     # result, degree = self.evaluate_and_filter(index, isfullKG)
        #     res = pool.apply_async(func=self.evaluate_and_filter, args=(index, isfullKG))
        #     # result, degree = pool.apply_async(func=self.evaluate_and_filter, args=(index, isfullKG))
        #     result_list.append(res)
        #     index_list.append(index)
        # pool.close()
        # pool.join()
        #
        # for i in range(len(result_list)):
        #     result, degree = result_list[i].get()
        #     # result 0:None 1:rules 2:quality rules
        #     # result, degree = self.evaluate_and_filter(index, isfullKG)
        #     if result != 0:
        #         index = index_list[i]
        #         candidate.append([index, result, degree])

        # None-Multiprocessing
        for item in score_top_container:
            count += 1
            index = [int(item[i]) for i in range(self.length)]
            if item[-1] == -1:
                continue
            if self.length == 1 and index == [self.pt]:
                continue
            result, degree = self.evaluate_and_filter(index, isfullKG)
            if result != 0:
                result_type, degree_type = self.evaluate_and_filter(index, isfullKG, type_flag=True)
                candidate.append([index, result, degree, result_type, degree_type])

        print('Progress: %d - %d ' % (count, top_candidate_size))
        print("Time:%f." % (time.time() - tt))
        if not gc.isenabled():
            gc.enable()
        del score_top_container
        gc.collect()
        gc.disable()

        return candidate

    def search_and_evaluate_2(self, length, Pt, tempCandidate, sampledFacts, sampledEntitySize,
                              # _syn, _coocc, ent_emb, rel_emb, isfullKG):
                              _syn, _coocc, rel_emb, isfullKG):
        self.length = length
        self.pt = Pt
        self.tempCandidate = tempCandidate
        self.pre_fact_dic_sample = RuleSearchingAndEvaluating.get_pre_fact_dic_sample(sampledFacts)
        self.ent_size_sample = sampledEntitySize

        self._syn = _syn
        self._coocc = _coocc
        # self.ent_emb = ent_emb
        self.rel_emb = rel_emb

        candidate = []

        """Calculate the Score Functions."""
        '''Calculate the f4.'''
        if len(self.tempCandidate) < self._syn:
            top_candidate_size = len(self.tempCandidate)
        else:
            top_candidate_size = self._syn
        # print("The number of New-COOCC Top Candidates is %d" % top_candidate_size)
        score_top_container = np.ones(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        score_top_container = - score_top_container
        print("Begin to calculate the f4: New synonymy")
        score_top_container = self.scoreFunction4(score_top_container)

        '''Calculate the f3.'''
        '''
        if len(self.tempCandidate) < self._coocc:
            top_candidate_size = len(self.tempCandidate)
        else:
            top_candidate_size = self._coocc
        # print("The number of New-COOCC Top Candidates is %d" % top_candidate_size)
        score_top_container = np.ones(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        score_top_container = - score_top_container
        subdic, objdic = self.get_sub_and_obj_dic()
        print("Begin to calculate the f3: New Co-occurrence")
        score_top_container = self.scoreFunction3(score_top_container, subdic, objdic)
        
        if not gc.isenabled():
            gc.enable()
        del subdic, objdic
        gc.collect()
        gc.disable()
        '''

        '''Calculate the f2.'''
        '''
        # top_candidate_size = int(_coocc * self.index_tuple_size)
        if len(self.tempCandidate) < _coocc:
            top_candidate_size = len(self.tempCandidate)
        else:
            top_candidate_size = _coocc
        print("The number of COOCC Top Candidates is %d" % top_candidate_size)
        score_top_container = np.zeros(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        subdic, objdic = self.get_sub_and_obj_dic()
        print("\nBegin to calculate the f2: Co-occurrence")
        score_top_container = self.score_function2(score_top_container, subdic, objdic)
        '''

        '''Calculate the f1.'''
        '''
        # top_candidate_size = int(_syn * self.index_tuple_size)
        if self.index_tuple_size < _syn:
            top_candidate_size = self.index_tuple_size
        else:
            top_candidate_size = _syn
        top_candidate_size = _syn
        score_top_container = np.zeros(shape=(top_candidate_size, self.length + 1), dtype=np.float)
        print("The number of SYN Top Candidates is %d" % top_candidate_size)
        print("\nBegin to calculate the f1: synonymy")
        score_top_container = self.score_function1(f, score_top_container, rel_emb)
        # Method 1: Top ones until it reaches the 100th. OMIT!
        # Method 2: Use two matrices to catch rules.

        print("\n Begin to use syn to filter: ")
        for item in score_top_container:
            index = [int(item[i]) for i in range(self.length)]
            if f == 0:  # matrix
                result, degree = self.evaluate_and_filter(index, DEGREE)
                if result != 0 and index not in all_candidate_set:
                    all_candidate_set.append(index)
                    candidate.append([index, result, degree])
            elif f == 1:  # vector
                # It needs to evaluate for all arranges of index.
                for i in itertools.permutations(index, self.length):
                    # Deduplicate.
                    _index = list(np.array(i))
                    if _index in all_candidate_set:
                        continue
                    result, degree = self.evaluate_and_filter(_index, DEGREE, isfullKG)
                    if result != 0:
                        all_candidate_set.append(_index)
                        candidate.append([_index, result, degree])
        if not gc.isenabled():
            gc.enable()
        del rel_emb, score_top_container
        gc.collect()
        gc.disable()
        '''

        """Evaluate the rules in score_top_container."""
        # Get relation_type. pre_dom_type, pre_ran_type
        # Get type_entity.
        self.type_entities_dic = util.get_type_entities(filename="./benchmarks/{0}/".format(self.BENCHMARK))

        print("Get the Pt matrix before evaluating the rules:")
        # if not isfullKG:
        #     if len(self.E_0) > 50000:
        #         self.ptmatrix_sample = self.getmatrix(self.pt, 0)
        #         print(" Sample: len:%d  size:%d" % (len(self.ptmatrix_sample), sys.getsizeof(self.ptmatrix_sample)))
        #     else:
        #         self.ptmatrix_part = self.getmatrix(self.pt, 1)
        #         print(" Part: len:%d  size:%d" % (len(self.ptmatrix_part), sys.getsizeof(self.ptmatrix_part)))

        self.ptmatrix_full = self.getmatrix(self.pt, 2)
        # TODO!!!!!修改！！！！每个都是临时计算的！！！！
        # self.ptmatrix_full_type = self.getmatrixByType(self.pt, 2)
        print(" Full: len:%d  size:%d" % (len(self.ptmatrix_full), sys.getsizeof(self.ptmatrix_full)))

        print("Begin to evaluate the rules in score_top_container: ")
        count = 0
        tt = time.time()

        # # Multiprocessing
        # # pool = Pool(4)
        # pool = mp.Pool(4)
        # result_list = []
        # index_list = []
        # for item in score_top_container:
        #     count += 1
        #     index = [int(item[i]) for i in range(self.length)]
        #     if item[-1] == -1:
        #         continue
        #     if self.length == 1 and index == [self.pt]:
        #         continue
        #     # result, degree = self.evaluate_and_filter(index, isfullKG)
        #     res = pool.apply_async(func=self.evaluate_and_filter, args=(index, isfullKG))
        #     # result, degree = pool.apply_async(func=self.evaluate_and_filter, args=(index, isfullKG))
        #     result_list.append(res)
        #     index_list.append(index)
        # pool.close()
        # pool.join()
        #
        # for i in range(len(result_list)):
        #     result, degree = result_list[i].get()
        #     # result 0:None 1:rules 2:quality rules
        #     # result, degree = self.evaluate_and_filter(index, isfullKG)
        #     if result != 0:
        #         index = index_list[i]
        #         candidate.append([index, result, degree])

        # None-Multiprocessing
        for item in score_top_container:
            count += 1
            index = [int(item[i]) for i in range(self.length)]
            if item[-1] == -1:
                continue
            if self.length == 1 and index == [self.pt]:
                continue
            result, degree = self.evaluate_and_filter(index, isfullKG)
            if result != 0:
                result_type, degree_type = self.evaluate_and_filter(index, isfullKG, type_flag=True)
                candidate.append([index, result, degree, result_type, degree_type])

        print('Progress: %d - %d ' % (count, top_candidate_size))
        print("Time:%f." % (time.time() - tt))
        if not gc.isenabled():
            gc.enable()
        del score_top_container
        gc.collect()
        gc.disable()

        return candidate