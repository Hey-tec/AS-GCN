# Author : Pey
# Time : 2021/3/31 18:35
# File_name : e_Choose_Features.py

# --------- Import Model ---------#

# --------- Sub Function ---------#

choose_file = open("./Source Data/file_2018_04_01/AS_Features.txt", "w")

with open("./Source Data/file_2018_04_01/AS_9_Features.txt") as f:
    for line in f:
        ASN, degree, transit_degree, distance_to_tier1, vp_out, hierarchy, \
        distance_to_vp_min, distance_to_vp_max, distance_to_vp_avg, as_type = line.strip().split()

        choose_features = [degree, transit_degree, distance_to_tier1, vp_out, hierarchy, distance_to_vp_min, distance_to_vp_max, distance_to_vp_avg, as_type]

        all_features = [ASN]
        for elem in choose_features:
            all_features.append(elem)

        for i in range(len(choose_features)):
            for j in range(i+1, len(choose_features)):
                all_features.append(str(float(choose_features[i]) * float(choose_features[j])))


        choose_file.write("\t".join(all_features) + "\n")


