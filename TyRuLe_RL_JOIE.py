# -*- coding: utf-8 -*
import numpy as np
import gc
import time
import send_process_report_email
import csv
import sys
import copy
import argparse
import os
import tensorflow as tf
import pickle
from KG import KG

# file imports 
import sampling as s
import ruleSearching as r
import util as util

BENCHMARK = "FB15K237"
# for FB15K237
R_minSC = 0.01
R_minHC = 0.001
QR_minSC = 0.5
QR_minHC = 0.001

# BENCHMARK = "WN18RR"
# # for WN18RR
# R_minSC = 0.01
# R_minHC = 0.001
# QR_minSC = 0.5
# QR_minHC = 0.001
DEGREE = [R_minSC, R_minHC, QR_minSC, QR_minHC]
Max_body_rule_length = 3  # not include head atom

# score function parameters
_syn = 1000
_coocc = 1000


if __name__ == '__main__':
    begin = time.time()

    parser = argparse.ArgumentParser(description="Setting some arguments.")
    parser.add_argument('--lpp', type=str, default="./linkprediction/{0}".format(BENCHMARK))
    args = parser.parse_args()
    print(args)
    print(args.lpp)
    lp_save_path = args.lpp
    if not os.path.exists(lp_save_path):
        os.makedirs(lp_save_path)
    lp_save_path += "/"

    print("\nLink Prediction of RLinker.\nThe benchmark is {0}.".format(BENCHMARK))

    facts_all, entity_name, predicate_name, type_name = \
        util.read_data(filename="./benchmarks/{0}/".format(BENCHMARK), file_type="train")
    print("Time for reading the benchmarks: %s." % (str(time.time() - begin)))

    """Data preprocess"""
    # Read the type info for relation.
    t = time.time()
    pre_dom_type = None
    pre_ran_type = None
    if type_name is not None:
        pre_dom_type, pre_ran_type = util.read_relation_type(filename="./benchmarks/{0}/".format(BENCHMARK))
        print("Time for reading the type infomation of relations: %s." % (str(time.time() - t)))

    allEntitySize = len(entity_name)
    allRelationSize = len(predicate_name)
    # Get the pre_fact_dic.
    t = time.time()
    pre_fact_dic_all = util.get_pre_fact_dic_all(facts_all)
    rsae = r.RuleSearchingAndEvaluating()
    # rsae.__int__(DEGREE, pre_fact_dic_all, allEntitySize, allRelationSize)
    rsae.__int__(DEGREE, pre_fact_dic_all, allEntitySize, allRelationSize, pre_dom_type, pre_ran_type, BENCHMARK)
    print("Time for getting the facts(entity pairs) for relations: %s." % (str(time.time() - t)))

    # Get the filter triples.
    t = time.time()
    h_r_dic_t, r_t_dic_h = util.get_filter_triple(facts_all)
    print("Time for getting the filter triples: %s.\n" % (str(time.time() - t)))

    # Train JOIE on our dataset, then get the trained embedding.
    t = time.time()
    # ent_emb, rel_emb = util.getEmbeddingJOIE()
    rel_emb = util.getEmbeddingJOIE()
    print("Time for getting the embedding: %s.\n" % (str(time.time() - t)))

    total_time = 0

    train_Pre_list = []
    if BENCHMARK == "FB15K237" or BENCHMARK == "WN18RR":
        with open("./benchmarks/{0}/train/target_pre.txt".format(BENCHMARK), 'r') as f:
            test_pre_num = f.readline()
            train_Pre_list = [int(line.strip('\n')) for line in f.readlines()]
            # Serve 11:CPU
            # For WN18RR
            # train_Pre_list = all pre,  len = 3 raw,    1
            # train_Pre_list = all pre,  len = 4 raw,    2
            # train_Pre_list = all pre,  len = 3 filter,    3
            # train_Pre_list = all pre,  len = 4 filter,    4
    else:
        pass
        train_pre_size = 10  # Todo:to set
        train_Pre_list = np.random.randint(0, allRelationSize, size=train_pre_size)
   
    predict_fact_num_total = 0
    # predict_Qfact_num_total = 0

    # MRR_total = []
    # Hit_1_total = []
    # Hit_3_total = []
    # Hit_10_total = []
    #
    # MRR_total_all = []
    # Hit_1_total_all = []
    # Hit_3_total_all = []
    # Hit_10_total_all = []

    with open("{0}{1}.csv".format(lp_save_path, BENCHMARK), "w") as csvfile:
        writer = csv.writer(csvfile)
        if Max_body_rule_length == 2:
            writer.writerow(
                ["Pt", "len=1", "len=2", "Total R num", "Total QR num", "Total TR num",
                 "time=1", "time=2", "Total time"])
        elif Max_body_rule_length == 3:
            writer.writerow(
                ["Pt", "len=1", "len=2", "len=3", "Total R num", "Total QR num", "Total TR num",
                 "time=1", "time=2", "time=3", "Total time"])
        elif Max_body_rule_length == 4:
            writer.writerow(
                ["Pt", "len=1", "len=2", "len=3", "len=4", "Total R num", "Total QR num", "Total TR num",
                 "time=1", "time=2", "time=3", "time=4", "Total time"])

    for Pt in train_Pre_list:
        Pt_start = time.time()

        # Add a column to identify usage flag.
        if facts_all.shape[1] == 4:
            facts_all = np.delete(facts_all, -1, axis=1)
        fl = np.zeros(facts_all.shape[0], dtype='int32')
        facts_all = np.c_[facts_all, fl]

        # Initialization all the variables. TODO
        new_index_Pt = None
        fact_dic_sample = None
        fact_dic_all = None
        facts_sample = None
        ent_size_sample = None
        pre_sample = None
        P_i_list_new = None
        P_count_new = None
        candidate_of_Pt = []
        pre_sample_of_Pt = []

        # Garbage collection.
        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()

        print("-------------------------------------------------pt: %d-------------------------------------------" % Pt)
        print("******************************Begin to sample********************************")
        # After sampling by Pt, return the E_0 and F_0.
        E_0, P_0, F_0, facts_all = s.first_sample_by_Pt(Pt, facts_all)
        # Initialization the sample all data E' F' and P' for different levels.
        E_level_1 = set()
        P_level_1 = set()
        F_level_1 = []
        E_level_2 = set()
        P_level_2 = set()
        F_level_2 = []

        E_i_minus_1 = E_0

        P_i_list = [list(P_0)]
        E_i_1_new = E_0  # todo????

        max_i = int((Max_body_rule_length + 1) / 2)
        if max_i > 2:
            print("Sorry, the body length is too long to learn the rules.")
            sys.exit(0)
        elif max_i < 1:
            print("Sorry, the body length is too short to learn the rules.")
            sys.exit(0)
        if max_i >= 1:
            print("Need to compute the P_1.")
            E_1, P_1, F_1_new, facts_all, P_count_old_1 = s.sample_by_i(1, E_i_minus_1, facts_all)
            E_i_minus_1 = E_1 - E_0  # Todo:whether to remove the duplicate entity.
            # E_i_minus_1 = E_1
            # print(len(E_i_minus_1))
            # print("The new entity size :%d   (RLvLR need to less than 800.)" % len(E_i_1_new))
            # print("let check the P_1 list:{0}".format(str(P_1)))
            # Merge the result.
            E_level_1 = E_0 | E_1  # set
            P_level_1 = P_0 | P_1  # set
            F_level_1.extend(F_0)
            F_level_1.extend(F_1_new)
            P_i_list.append(list(P_1))
        if max_i == 2:
            print("Need to compute the P_2.")
            E_2, P_2, F_2_new, facts_all, P_count_old_2 = s.sample_by_i(2, E_i_minus_1, facts_all)
            # since we remove the duplicate entity of last layer, the pt will be not included.
            P_2.add(Pt)
            # print("let check the P_2 list:{0}".format(str(P_2)))
            # Merge the result.
            E_level_2 = E_level_1 | E_2
            P_level_2 = P_level_1 | P_2
            F_level_2.extend(F_level_1)
            F_level_2.extend(F_2_new)
            P_i_list.append(list(P_2))

        print("\nStep %d: Get the sampled E', P' and F'" % (max_i + 2))
        print("E'(i=1) size: %d" % (len(E_level_1)))
        print("P'(i=1) size: %d" % (len(P_level_1)))
        print("F'(i=1) size: %d" % (len(F_level_1)))
        print("E'(i=2) size: %d" % (len(E_level_2)))
        print("P'(i=2) size: %d" % (len(P_level_2)))
        print("F'(i=2) size: %d" % (len(F_level_2)))

        # Reindex
        # 1.Reindex the relations, since we allow inverse relations.
        # ONLY RELATIONS HAVE THE INVERSE, NOT FACTS AND ENTITIES.
        # print(P_i_list)
        for p_i in P_i_list:
            p_i_inverse = []
            for p in p_i:
                p_i_inverse.append(p + allRelationSize)
            p_i.extend(p_i_inverse)
        # print(P_i_list)
        print("Reindex for allow inverse relations has been done.")
        # 2.Todo: Reindex the relations, entities and facts,
        # since we allow inverse relations and need to be used in embedding model.
        # new_index_Pt, P_i_list_new, facts_sample = \
        #     reindex(Pt, max_i, P_i_list, E, P, F, predicate_name, )
        # Todo: whetehr to save in file is decided by whether to use the embedding model.
        # save_path = './sampled/' + BENCHMARK
        # save_sample_data()

        print("*********************************End to sample**********************************\n")

        print("************************Begin to filter by relation type************************")
        pt_dom_type_list = None
        pt_ran_type_list = None
        if type_name is not None:
            pt_dom_type_list = pre_dom_type.get(Pt)
            # print(pt_dom_type_list)
            pt_ran_type_list = pre_ran_type.get(Pt)
            # print(pt_ran_type_list)
        candidate_all_length_list = []
        if Max_body_rule_length >= 1:
            candidate_1 = []
            Pos_1 = P_i_list[1]
            if type_name is not None:
                for p in Pos_1:
                    if p >= allRelationSize:
                        p_dom_type_list = pre_ran_type.get(p - allRelationSize)
                        p_ran_type_list = pre_dom_type.get(p - allRelationSize)
                    else:
                        p_dom_type_list = pre_dom_type.get(p)
                        p_ran_type_list = pre_ran_type.get(p)
                    if len(set(p_dom_type_list) & set(pt_dom_type_list)) != 0:
                        if len(set(p_ran_type_list) & set(pt_ran_type_list)) != 0:
                            candidate_1.append(p)
            else:
                candidate_1 = Pos_1
            candidate_all_length_list.append(candidate_1)
            print("Body length = 1 filtering done. Before: %d candidates. After: %d candidates."
                  % (len(Pos_1), len(candidate_1)))
        if Max_body_rule_length >= 2:
            candidate_2 = []
            Pos_1 = copy.deepcopy(P_i_list[1])
            Pos_2 = copy.deepcopy(P_i_list[1])
            if type_name is not None:
                for p in P_i_list[1]:
                    if p >= allRelationSize:
                        p_dom_type_list = pre_ran_type.get(p - allRelationSize)
                    else:
                        p_dom_type_list = pre_dom_type.get(p)
                    if len(set(p_dom_type_list) & set(pt_dom_type_list)) == 0:
                        Pos_1.remove(p)
                for p in P_i_list[1]:
                    if p >= allRelationSize:
                        p_ran_type_list = pre_dom_type.get(p - allRelationSize)
                    else:
                        p_ran_type_list = pre_ran_type.get(p)
                    if len(set(p_ran_type_list) & set(pt_ran_type_list)) == 0:
                        Pos_2.remove(p)
            # constrain of middle variable
            tempCandidate = [[p1, p2] for p1 in Pos_1 for p2 in Pos_2]
            if type_name is not None:
                for tempRule in tempCandidate:
                    if tempRule[0] >= allRelationSize:
                        Pos_1_z = pre_dom_type.get(tempRule[0] - allRelationSize)
                    else:
                        Pos_1_z = pre_ran_type.get(tempRule[0])
                    if tempRule[1] >= allRelationSize:
                        Pos_2_z = pre_ran_type.get(tempRule[1] - allRelationSize)
                    else:
                        Pos_2_z = pre_dom_type.get(tempRule[1])
                    if len(set(Pos_1_z) & set(Pos_2_z)) != 0:
                        candidate_2.append(tempRule)
            else:
                candidate_2 = tempCandidate
            candidate_all_length_list.append(candidate_2)
            print("Body length = 2 filtering done. Before: %d candidates. Middle: %d candidates. After: %d candidates."
                  % (len(P_i_list[1]) * len(P_i_list[1]), len(tempCandidate), len(candidate_2)))

        if Max_body_rule_length >= 3:
            candidate_3 = []
            Pos_1 = copy.deepcopy(P_i_list[1])
            Pos_2 = copy.deepcopy(P_i_list[2])
            Pos_3 = copy.deepcopy(P_i_list[1])
            if type_name is not None:
                for p in P_i_list[1]:
                    if p >= allRelationSize:
                        p_dom_type_list = pre_ran_type.get(p - allRelationSize)
                    else:
                        p_dom_type_list = pre_dom_type.get(p)
                    if len(set(p_dom_type_list) & set(pt_dom_type_list)) == 0:
                        Pos_1.remove(p)
                for p in P_i_list[1]:
                    if p >= allRelationSize:
                        p_ran_type_list = pre_dom_type.get(p - allRelationSize)
                    else:
                        p_ran_type_list = pre_ran_type.get(p)
                    if len(set(p_ran_type_list) & set(pt_ran_type_list)) == 0:
                        Pos_3.remove(p)
            # constrain of middle variable
            tempCandidate = [[p1, p2, p3] for p1 in Pos_1 for p2 in Pos_2 for p3 in Pos_3]
            print("the number of tempCandidate to be test is %d" % (len(tempCandidate)))
            if type_name is not None:
                for tempRule in tempCandidate:  # [p1(x,z1), p2(z1,z2), p3(z2,y)]
                    if tempRule[0] >= allRelationSize:
                        Pos_1_z1 = pre_dom_type.get(tempRule[0] - allRelationSize)
                    else:
                        Pos_1_z1 = pre_ran_type.get(tempRule[0])
                    if tempRule[1] >= allRelationSize:
                        Pos_2_z1 = pre_ran_type.get(tempRule[1] - allRelationSize)
                        Pos_2_z2 = pre_dom_type.get(tempRule[1] - allRelationSize)
                    else:
                        Pos_2_z1 = pre_dom_type.get(tempRule[1])
                        Pos_2_z2 = pre_ran_type.get(tempRule[1])
                    if tempRule[2] >= allRelationSize:
                        Pos_3_z2 = pre_ran_type.get(tempRule[2] - allRelationSize)
                    else:
                        Pos_3_z2 = pre_dom_type.get(tempRule[2])
                    if len(set(Pos_1_z1) & set(Pos_2_z1)) != 0:
                        if set(Pos_2_z2) & set(Pos_3_z2) != 0:
                            candidate_3.append(tempRule)
            else:
                candidate_3 = tempCandidate
            candidate_all_length_list.append(candidate_3)
            print("Body length = 3 filtering done. Before: %d candidates. Middle: %d candidates. After: %d candidates."
                  % (len(P_i_list[1]) * len(P_i_list[2]) * len(P_i_list[1]), len(tempCandidate), len(candidate_3)))
        if Max_body_rule_length == 4:
            candidate_4 = []
            Pos_1 = copy.deepcopy(P_i_list[1])
            Pos_2 = copy.deepcopy(P_i_list[2])
            Pos_3 = copy.deepcopy(P_i_list[2])
            Pos_4 = copy.deepcopy(P_i_list[1])
            if type_name is not None:
                for p in P_i_list[1]:
                    if p >= allRelationSize:
                        p_dom_type_list = pre_ran_type.get(p - allRelationSize)
                    else:
                        p_dom_type_list = pre_dom_type.get(p)
                    if len(set(p_dom_type_list) & set(pt_dom_type_list)) == 0:
                        Pos_1.remove(p)
                for p in P_i_list[1]:
                    if p >= allRelationSize:
                        p_ran_type_list = pre_dom_type.get(p - allRelationSize)
                    else:
                        p_ran_type_list = pre_ran_type.get(p)
                    if len(set(p_ran_type_list) & set(pt_ran_type_list)) == 0:
                        Pos_4.remove(p)
            # constrain of middle variable
            tempCandidate = [[p1, p2, p3, p4] for p1 in Pos_1 for p2 in Pos_2 for p3 in Pos_3 for p4 in Pos_4]
            if type_name is not None:
                for tempRule in tempCandidate:  # [p1(x,z1), p2(z1,z2), p3(z2,z3), p4(z3,y)]
                    if tempRule[0] >= allRelationSize:
                        Pos_1_z1 = pre_dom_type.get(tempRule[0] - allRelationSize)
                    else:
                        Pos_1_z1 = pre_ran_type.get(tempRule[0])
                    if tempRule[1] >= allRelationSize:
                        Pos_2_z1 = pre_ran_type.get(tempRule[1] - allRelationSize)
                        Pos_2_z2 = pre_dom_type.get(tempRule[1] - allRelationSize)
                    else:
                        Pos_2_z1 = pre_dom_type.get(tempRule[1])
                        Pos_2_z2 = pre_ran_type.get(tempRule[1])
                    if tempRule[2] >= allRelationSize:
                        Pos_3_z2 = pre_ran_type.get(tempRule[2] - allRelationSize)
                        Pos_3_z3 = pre_dom_type.get(tempRule[2] - allRelationSize)
                    else:
                        Pos_3_z2 = pre_dom_type.get(tempRule[2])
                        Pos_3_z3 = pre_ran_type.get(tempRule[2])
                    if tempRule[3] >= allRelationSize:
                        Pos_4_z3 = pre_ran_type.get(tempRule[3] - allRelationSize)
                    else:
                        Pos_4_z3 = pre_dom_type.get(tempRule[3])
                    if len(set(Pos_1_z1) & set(Pos_2_z1)) != 0:
                        if set(Pos_2_z2) & set(Pos_3_z2) != 0:
                            if set(Pos_3_z3) & set(Pos_4_z3) != 0:
                                candidate_4.append(tempRule)
            else:
                candidate_4 = tempCandidate
            candidate_all_length_list.append(candidate_4)
            print("Body length = 4 filtering done. Before: %d candidates. Middle: %d candidates. After: %d candidates."
                  % (len(P_i_list[1]) * len(P_i_list[2]) * len(P_i_list[2]) * len(P_i_list[1]), len(tempCandidate),
                     len(candidate_4)))
        print("**********************End to filter by relation type********************\n")

        if not gc.isenabled():
            gc.enable()
        gc.collect()
        gc.disable()

        print("***************Begin to search rules by scoring function*****************")
        candidate_of_Pt = []
        sampledFacts = None
        sampledEntitySize = 0
        num_li = []
        time_li = []
        Pt_i_1 = time.time()
        for length in range(1, Max_body_rule_length + 1):
            print("^^^^^^^^^^^^^^^^^^^^^Body length = {0}^^^^^^^^^^^^^^^^^^^^^".format(length))
            if int((length + 1) / 2) == 1:
                sampledFacts = F_level_1
                sampledEntitySize = len(E_level_1)
            elif int((length + 1) / 2) == 2:
                sampledFacts = F_level_2
                sampledEntitySize = len(E_level_2)
            tempCandidate = candidate_all_length_list[length - 1]
            isfullKG = 1
            candidate = rsae.search_and_evaluate_2(length, Pt, tempCandidate, sampledFacts, sampledEntitySize,
                                                   # _syn, _coocc, ent_emb, rel_emb, isfullKG)
                                                   _syn, _coocc, rel_emb, isfullKG)
            print("Body length = %d searching done. Before: %d candidates. After: %d rules.\n"
                  % (length, len(tempCandidate), len(candidate)))
            candidate_of_Pt.extend(candidate)
            num_li.append(len(candidate))
            Pt_i = time.time()
            time_li.append(Pt_i - Pt_i_1)
            print("Time = %f." % (Pt_i - Pt_i_1))
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            Pt_i_1 = Pt_i

        print("***************End to search rules by scoring function*****************\n")
        # Save rules and timing.

        num_rule, num_Qrule, typeR_num = util.save_rules(lp_save_path, Pt, candidate_of_Pt, predicate_name)
        # num_rule = len(candidate_of_Pt)
        # num_Qrule = 0

        Pt_end = time.time()
        Pt_time = Pt_end - Pt_start
        total_time += Pt_time
        print("This %d th predicate's total Rule num: %d" % (Pt, num_rule))
        print("This %d th predicate's total Quality Rule num: %d" % (Pt, num_Qrule))
        print("This %d th predicate's total Type Rule num: %d" % (Pt, typeR_num))
        print("This %d th predicate's total time: %f\n" % (Pt, Pt_time))

        line = [Pt]
        line.extend(num_li)
        line.append(num_rule)
        line.append(num_Qrule)
        line.append(typeR_num)
        line.extend(time_li)
        line.append(Pt_time / 3600)
        # line.extend(reslut_list)
        # line.append(float(lp_time / 3600))

        with open("{0}{1}.csv".format(lp_save_path, BENCHMARK), "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(line)

    # Save MRR, Hit_10 in file.
    # with open('./linkprediction/' + BENCHMARK + '/' + 'test' + '.txt', 'w') as f:
    #     # Save predict_fact_num, predict_Qfact_num.
    #     # predict_fact_num_total /= len(train_Pre_list)
    #     # predict_Qfact_num_total /= len(train_Pre_list)
    #     # f.write("predict_fact_avg: " + str(predict_fact_num_total) + '\n')
    #     # f.write("predict_Qfact_avg: " + str(predict_Qfact_num_total) + '\n')
    #     # MRR
    #     f.write("-----MRR_total: %d-----\n" % len(MRR_total))
    #     for mrr in MRR_total:
    #         f.write("%d, %f\n" % (mrr[0], mrr[1]))
    #     f.write("-----MRR_total AVG: %f-----\n" % np.mean([mrr[1] for mrr in MRR_total]))
    #     # Hit@10
    #     f.write("-----Hit_10_total: %d-----\n" % len(Hit_10_total))
    #     for hit10 in Hit_10_total:
    #         f.write("%d, %f\n" % (hit10[0], hit10[1]))
    #     f.write("-----Hit_10_total AVG: %f-----\n" % np.mean([hit10[1] for hit10 in Hit_10_total]))
    #     # Hit@3
    #     f.write("-----Hit_3_total: %d-----\n" % len(Hit_3_total))
    #     for hit3 in Hit_3_total:
    #         f.write("%d, %f\n" % (hit3[0], hit3[1]))
    #     f.write("-----Hit_3_total AVG: %f-----\n" % np.mean([hit3[1] for hit3 in Hit_3_total]))
    #     # Hit@1
    #     f.write("-----Hit_1_total: %d-----\n" % len(Hit_1_total))
    #     for hit1 in Hit_1_total:
    #         f.write("%d, %f\n" % (hit1[0], hit1[1]))
    #     f.write("-----Hit_1_totalAVG: %f-----\n" % np.mean([hit1[1] for hit1 in Hit_1_total]))
    #     f.write("\n")
    #
    #     # MRR_all
    #     f.write("-----MRR_total_all: %d-----\n" % len(MRR_total_all))
    #     for mrr in MRR_total_all:
    #         f.write("%d, %f\n" % (mrr[0], mrr[1]))
    #     f.write("-----MRR_total_all AVG: %f-----\n" % np.mean([mrr[1] for mrr in MRR_total_all]))
    #     # Hit@10_all
    #     f.write("-----Hit_10_total_all: %d-----\n" % len(Hit_10_total_all))
    #     for hit10 in Hit_10_total_all:
    #         f.write("%d, %f\n" % (hit10[0], hit10[1]))
    #     f.write("-----Hit_10_total_all AVG: %f-----\n" % np.mean([hit10[1] for hit10 in Hit_10_total_all]))
    #     # Hit@3_all
    #     f.write("-----Hit_3_total_all: %d-----\n" % len(Hit_3_total_all))
    #     for hit3 in Hit_3_total_all:
    #         f.write("%d, %f\n" % (hit3[0], hit3[1]))
    #     f.write("-----Hit_3_total_all AVG: %f-----\n" % np.mean([hit3[1] for hit3 in Hit_3_total_all]))
    #     # Hit@1_all
    #     f.write("-----Hit_1_total_all: %d-----\n" % len(Hit_1_total_all))
    #     for hit1 in Hit_1_total_all:
    #         f.write("%d, %f\n" % (hit1[0], hit1[1]))
    #     f.write("-----Hit_1_total_all AVG: %f-----\n" % np.mean([hit1[1] for hit1 in Hit_1_total_all]))

    '''Total time'''
    end = time.time() - begin
    hour = int(end / 3600)
    minute = int((end - hour * 3600) / 60)
    second = end - hour * 3600 - minute * 60
    print("Average time:%s" % str(total_time / len(train_Pre_list)))
    print("Algorithm total time: %f" % end)
    print("Algorithm total time: %d : %d : %f\n" % (hour, minute, second))

    subject = BENCHMARK + ", over!"
    text = "MAX LEN={0}\n train_Pre_list={1}\n Let's watch the result! Go Go Go!\n".\
        format(str(Max_body_rule_length), str(train_Pre_list))
    send_process_report_email.send_email_main_process(subject, text)

