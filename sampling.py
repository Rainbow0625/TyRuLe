import numpy as np
import time

'''
For the sampling process, RLvLR picked at most 50 neighbours of an entity 
and set the maximum size of each sample to 800 entities. 
'''


def first_sample_by_Pt(Pt, facts):
    print("Step 1: First sample by Pt to get E_0:")
    time_start = time.time()
    P_0 = set()
    P_0.add(Pt)
    E_0 = set()
    F_0 = []
    for f in facts:
        # Sample.
        if f[2] == Pt and f[3] == 0:
            fact = f.tolist()
            F_0.append(fact)
            f[3] = 1  # Mark it has been included.
            E_0.add(f[0])
            E_0.add(f[1])
    print("E_0 size: %d" % len(E_0))
    print("P_0 size: %d" % len(P_0))
    print("F_0 size: %d" % len(F_0))
    time_end = time.time()
    print('Step 1 cost time:', time_end-time_start)
    return E_0, P_0, F_0, facts


def sample_by_i(index, E_i_minis_1, facts):
    """
    Use the predicates' occurrences to filter some more frequent predicates.
    """

    print("\nStep %d: Sample by i=%d:" % (index+1, index))
    time_start = time.time()

    del_flag = 0
    E_i = set()  # Maybe it contains some repeated entities.
    F_i = []
    P_count = {}  # After filtering:   Key: p's index; Value: [count]

    # Statistical occurrences.
    P_dic = {}  # Key: p's index; Value: [count, [fact's index list]]
    for i in range(len(facts)):
        f = list(facts[i])
        if f[0] in E_i_minis_1 or f[1] in E_i_minis_1:
            if f[2] in P_dic.keys():
                value = P_dic.get(f[2])
                # Restrict the number of entity!
                value[1].append(i)  # else, only count the freq.
                P_dic[f[2]] = [value[0]+1, value[1]]
            else:
                P_dic[f[2]] = [1, [i]]

    keys = list(P_dic.keys())
    for key in keys:
        value = P_dic[key]
        # Pick out predicates with high frequency of occurrence.
        if value[0] < 0:   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            del_flag = del_flag + 1
        else:
            P_count[key] = value[0]
            # Get E_i and P_count.
            for j in value[1]:
                E_i.add(facts[j][0])
                E_i.add(facts[j][1])
                if facts[j][3] == 0:
                    F_i.append(list(facts[j]))
                    facts[j][3] = 1

    P_i = set(P_count.keys())
    # print("\n################################################")
    # print("Count and list the predicates' frequency:")
    # print(list(P_count.values()))
    count_mean = int(np.mean(np.array(list(P_count.values()), dtype=np.int32)))
    # print("The average frequency of all the predicates: %d" % count_mean)
    if del_flag > 0:
        print("Leave pre num:%d" % len(P_i))
        print("Delete pre num:%d \n" % del_flag)
    # print("################################################\n")

    print("E_%d size: %d (Maybe it contains some repeated entities.)" % (index, len(E_i)))
    print("P_%d size: %d" % (index, len(P_i)))
    print("F_%d_new size: %d" % (index, len(F_i)))
    time_end = time.time()
    print('Step %d cost time:%d' % (index+1, (time_end - time_start)))

    # todo: why return P_count????
    return E_i, P_i, F_i, facts, P_count


def filter_predicates_by_count(Pt, P_count_dic, P_new_index_list, fact_dic_sample, fact_dic_all):
    del_flag = 0
    keys = list(P_count_dic.keys())
    for key in keys:
        if key == Pt or key-1 == Pt:
            continue
        if P_count_dic.get(key) > 300 or P_count_dic.get(key) < 100:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Remove the elements filtered.
            P_new_index_list[-1].remove(key)
            if key in P_new_index_list[-2]:
                P_new_index_list[-2].remove(key)
            if key % 2 == 0:
                if key in fact_dic_all.keys():
                    del fact_dic_all[key]
                    del fact_dic_sample[key]
            else:
                if key-1 in fact_dic_all.keys():
                    del fact_dic_all[key-1]
                    del fact_dic_sample[key-1]
            del_flag = del_flag + 1
    print("Remove num: %d" % del_flag)
    return P_new_index_list, fact_dic_sample, fact_dic_all


