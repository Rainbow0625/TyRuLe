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

# file imports
import linkPrediction as lp
import util as util

BENCHMARK = "AirGraph"
# BENCHMARK = "YAGO26K906"
# BENCHMARK = "FB15K237"
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
# _syn = 800
_coocc = 1000


if __name__ == '__main__':
    begin = time.time()

    parser = argparse.ArgumentParser(description="Setting some arguments.")
    parser.add_argument('--lpp', type=str, default="./linkprediction/{0}".format(BENCHMARK))
    args = parser.parse_args()
    print(args)
    print(args.lpp)
    rules_save_path = "./linkprediction/{0}/rules/".format(BENCHMARK)
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
    # if Max_body_rule_length > 2:
    #     pre_dom_type, pre_ran_type, pre_dom_type_one, pre_ran_type_one = \
    #         read_relation_type(filename="./benchmarks/{0}/".format(BENCHMARK))
    # else:
    #     pre_dom_type, pre_ran_type = read_relation_type(filename="./benchmarks/{0}/".format(BENCHMARK))
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
    print("Time for getting the facts(entity pairs) for relations: %s." % (str(time.time() - t)))

    # Get the filter triples.
    t = time.time()
    h_r_dic_t, r_t_dic_h = util.get_filter_triple(facts_all)
    print("Time for getting the filter triples: %s.\n" % (str(time.time() - t)))

    total_time = 0

    test_Pre_list = []
    if BENCHMARK == "FB15K237" or BENCHMARK == "WN18RR" or BENCHMARK == "AirGraph":
        with open("./benchmarks/{0}/train/target_pre.txt".format(BENCHMARK), 'r') as f:
            test_pre_num = f.readline()
            # test_Pre_list = [int(line.strip('\n')) for line in f.readlines()]  # 1 51688
            # AirGraph
            test_Pre_list = [2, 9, 18, 19, 21, 22, 23, 26, 27, 29, 30, 33, 38, 39]

    predict_fact_num_total = 0
    # predict_Qfact_num_total = 0

    with open("{0}{1}.csv".format(lp_save_path, BENCHMARK), "w") as csvfile:
        writer = csv.writer(csvfile)
        if Max_body_rule_length == 2:
            writer.writerow(
                ["Pt",

                 "right_MRR_part3", "right_Hits_1_part3", "right_Hits_3_part3", "right_Hits_10_part3",
                 "left_MRR_part3", "left_Hits_1_part3", "left_Hits_3_part3", "left_Hits_10_part3",
                 "MRR_part3", "Hit_1_part3", "Hit_3_part3", "Hit_10_part3",

                 "right_MRR_part23", "right_Hits_1_part23", "right_Hits_3_part23", "right_Hits_10_part23",
                 "left_MRR_part23", "left_Hits_1_part23", "left_Hits_3_part23", "left_Hits_10_part23",
                 "MRR_part23", "Hit_1_part23", "Hit_3_part23", "Hit_10_part23",

                 "right_MRR_all", "right_Hits_1_all", "right_Hits_3_all", "right_Hits_10_all",
                 "left_MRR_all", "left_Hits_1_all", "left_Hits_3_all", "left_Hits_10_all",
                 "MRR_all", "Hit_1_all", "Hit_3_all", "Hit_10_all",

                 "LinkPrediction time"])
        elif Max_body_rule_length == 3:
            writer.writerow(
                ["Pt",

                 "right_MRR_part3", "right_Hits_1_part3", "right_Hits_3_part3", "right_Hits_10_part3",
                 "left_MRR_part3", "left_Hits_1_part3", "left_Hits_3_part3", "left_Hits_10_part3",
                 "MRR_part3", "Hit_1_part3", "Hit_3_part3", "Hit_10_part3",

                 "right_MRR_part23", "right_Hits_1_part23", "right_Hits_3_part23", "right_Hits_10_part23",
                 "left_MRR_part23", "left_Hits_1_part23", "left_Hits_3_part23", "left_Hits_10_part23",
                 "MRR_part23", "Hit_1_part23", "Hit_3_part23", "Hit_10_part23",

                 "right_MRR_all", "right_Hits_1_all", "right_Hits_3_all", "right_Hits_10_all",
                 "left_MRR_all", "left_Hits_1_all", "left_Hits_3_all", "left_Hits_10_all",
                 "MRR_all", "Hit_1_all", "Hit_3_all", "Hit_10_all",

                 "LinkPrediction time"])
        elif Max_body_rule_length == 4:
            writer.writerow(
                ["Pt",

                  "right_MRR_part3", "right_Hits_1_part3", "right_Hits_3_part3", "right_Hits_10_part3",
                 "left_MRR_part3", "left_Hits_1_part3", "left_Hits_3_part3", "left_Hits_10_part3",
                  "MRR_part3", "Hit_1_part3", "Hit_3_part3", "Hit_10_part3",

                  "right_MRR_part23", "right_Hits_1_part23", "right_Hits_3_part23", "right_Hits_10_part23",
                  "left_MRR_part23", "left_Hits_1_part23", "left_Hits_3_part23", "left_Hits_10_part23",
                  "MRR_part23", "Hit_1_part23", "Hit_3_part23", "Hit_10_part23",

                  "right_MRR_all", "right_Hits_1_all", "right_Hits_3_all", "right_Hits_10_all",
                  "left_MRR_all", "left_Hits_1_all", "left_Hits_3_all", "left_Hits_10_all",
                  "MRR_all", "Hit_1_all", "Hit_3_all", "Hit_10_all",

                  "LinkPrediction time"])

    test_file_path = './benchmarks/' + BENCHMARK + '/'
    test_facts = util.read_data(filename=test_file_path, file_type="test")

    for Pt in test_Pre_list:
        Pt_start = time.time()

        candidate_of_Pt = util.read_rules(rules_save_path, Pt)
        # print(candidate_of_Pt)
        if candidate_of_Pt is None:
            print("The rule can not be read.")
            sys.exit()
        # num_rule = len(candidate_of_Pt)
        # num_Qrule = 0

        candidateWithType = None
        x_entity_set = None
        y_entity_set = None
        # if type_name is not None:
        #     # candidateWithType = util.mark_and_count_type_of_rules(Pt, candidate_of_Pt, pre_dom_type, pre_ran_type,
        #     #                                                  allRelationSize, type_name)
        #
        #     # for 5
        #     x_entity_set, y_entity_set = util.mark_and_count_type_of_rules_2(
        #         BENCHMARK, Pt, candidate_of_Pt, pre_dom_type, pre_ran_type, allRelationSize, type_name)

        '''Link prediction based on test facts'''
        print("**********************Begin to link prediction*********************")
        time_lp_start = time.time()
        test_facts_pt = lp.filter_fb15k237(test_facts, Pt)

        print("Step 1: Begin to predict the facts by rules.")
        # 1 and 5
        # predict_matrix, predict_fact_num, predict_facts_by_rule = lp.predict(lp_save_path, Pt,
        #                                                                  pre_fact_dic_all, candidate_of_Pt,
        #                                                                  allEntitySize, allRelationSize)
        # 1.2
        predict_matrix, predict_fact_num, predict_facts_by_rule = lp.predict_SC_type(lp_save_path, Pt,
                                                                                     pre_fact_dic_all, candidate_of_Pt,
                                                                                     allEntitySize, allRelationSize)
        # 1.3 only type rules
        # predict_matrix, predict_fact_num, predict_facts_by_rule = lp.predict_SC_type_only(lp_save_path, Pt,
        #                                                                              pre_fact_dic_all, candidate_of_Pt,
        #                                                                              allEntitySize, allRelationSize)
        # 2
        # predict_matrix_right, predict_matrix_left, \
        # predict_fact_num, \
        # predict_facts_by_rule_right, predict_facts_by_rule_left = lp.predictByType_Zn_Y(lp_save_path, Pt,
        #                                                                      pre_fact_dic_all, candidateWithType,
        #                                                                      allEntitySize, allRelationSize,
        #                                                                                 BENCHMARK)
        # 3
        # predict_matrix_right, predict_matrix_left, \
        # predict_fact_num, \
        # predict_facts_by_rule_right, predict_facts_by_rule_left = lp.predictByType_Y(lp_save_path, Pt,
        #                                                                      pre_fact_dic_all, candidateWithType,
        #                                                                      allEntitySize, allRelationSize,
        #                                                                              BENCHMARK)
        # 4
        # predict_matrix, predict_fact_num, predict_facts_by_rule = lp.predictByType(lp_save_path, Pt,
        #                                                                        pre_fact_dic_all, candidateWithType,
        #                                                                        allEntitySize, allRelationSize,
        #                                                                        BENCHMARK)

        predict_fact_num_total += predict_fact_num
        # predict_Qfact_num_total += predict_Qfact_num
        mid = time.time()
        print("Predict time: %f\n" % (mid - time_lp_start))

        print("Step 2: Begin to test data by predicted facts.")
        # For predict 1 and 4
        # 1. Noisy-OR-Path
        # MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all \
        #     = lp.testByNoisyORPath(lp_save_path, Pt, predict_matrix, test_facts)
        # 2. Noisy-OR
        # MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all \
        #     = lp.testByNoisyOR(lp_save_path, Pt, test_facts, predict_facts_by_rule)
        # 3. Max-Aggregation
        reslut_list = lp.testByMaxAggregation(lp_save_path, Pt, test_facts_pt, predict_facts_by_rule,
                                              h_r_dic_t, r_t_dic_h)
        # not change for 1. return list 2. filter facts  3. collect none predicted facts.

        # For predict 2 and 3
        # 1. Noisy-OR-Path
        # MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all \
        #     = lp.testByNoisyORPath_2(lp_save_path, Pt, predict_matrix_right, predict_matrix_left, test_facts)
        # 2. Noisy-OR
        # MRR, Hit_1, Hit_3, Hit_10, MRR_all, Hit_1_all, Hit_3_all, Hit_10_all \
        #     = lp.testByNoisyOR_2(lp_save_path, Pt, test_facts,
        #     predict_facts_by_rule_right, predict_facts_by_rule_left)
        # 3. Max-Aggregation
        # reslut_list = lp.testByMaxAggregation_2(lp_save_path, Pt, test_facts,
        #                                         predict_facts_by_rule_right, predict_facts_by_rule_left,
        #                                         h_r_dic_t, r_t_dic_h)

        # For predict 5
        # reslut_list = lp.testByMaxAggregation_filterType(lp_save_path, Pt, test_facts, predict_facts_by_rule,
        #                                       h_r_dic_t, r_t_dic_h, x_entity_set, y_entity_set)

        # MRR_total.append([Pt, MRR])
        # Hit_1_total.append([Pt, Hit_1])
        # Hit_3_total.append([Pt, Hit_3])
        # Hit_10_total.append([Pt, Hit_10])
        #
        # MRR_total_all.append([Pt, MRR_all])
        # Hit_1_total_all.append([Pt, Hit_1_all])
        # Hit_3_total_all.append([Pt, Hit_3_all])
        # Hit_10_total_all.append([Pt, Hit_10_all])

        mid2 = time.time()
        print("Test time: %f" % (mid2 - mid))

        lp_time = time.time() - time_lp_start
        print("\nLink prediction time: %f" % lp_time)
        hour = int(lp_time / 3600)
        minute = int((lp_time - hour * 3600) / 60)
        second = lp_time - hour * 3600 - minute * 60
        print(str(hour) + " : " + str(minute) + " : " + str(second))
        print("**********************End to link prediction**********************\n")

        # Save in .csv
        line = [Pt]
        # line.extend(num_li)
        # line.append(num_rule)
        # line.append(num_Qrule)
        # line.extend(time_li)
        # line.append(Pt_time / 3600)
        line.extend(reslut_list)
        line.append(float(lp_time / 3600))

        with open("{0}{1}.csv".format(lp_save_path, BENCHMARK), "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(line)

    '''Total time'''
    end = time.time() - begin
    hour = int(end / 3600)
    minute = int((end - hour * 3600) / 60)
    second = end - hour * 3600 - minute * 60
    print("Average time:%s" % str(total_time / len(test_Pre_list)))
    print("Algorithm total time: %f" % end)
    print("Algorithm total time: %d : %d : %f\n" % (hour, minute, second))

    subject = BENCHMARK + ", over!"
    text = "MAX LEN={0}\n train_Pre_list={1}\n Let's watch the result! Go Go Go!\n".\
        format(str(Max_body_rule_length), str(test_Pre_list))
    send_process_report_email.send_email_main_process(subject, text)


