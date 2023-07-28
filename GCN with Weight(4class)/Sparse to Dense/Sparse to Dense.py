# Author : Pey
# Time : 2021/4/5 21:54
# File_name : Sparse to Dense.py

# --------- Import Model ---------#
import numpy as np
# --------- Sub Function ---------#
File_parents_path = "./Source Data/file_2016_07_01/"
Output_File_parents_path = "../data/file_2016_07_01/"

with open(File_parents_path + "AS_9_Features.txt") as file1:
    AS_List = []
    for line in file1:
        elem = line.strip().split()
        AS_List.append(int(elem[0]))
    AS_List_sort = AS_List
    AS_List_sort.sort()

with open(File_parents_path + "AS_9_Features.txt") as file2:
    AS_Features_Dict = {}
    for line in file2:
        elem = line.strip().split()
        AS_Features_Dict[int(elem[0])] = "\t".join(elem[1:])


file3 = open(Output_File_parents_path + "AS_Features.txt", "w")
for AS in AS_List_sort:
    file3.write(str(AS_List_sort.index(AS)) + '\t' + AS_Features_Dict[AS] + '\n')

file5 = open(Output_File_parents_path + "line_graph_edge.txt", "w")
with open(File_parents_path + "line_graph_edge.txt") as file4:
    for line in file4:
        AS1, AS2 = line.strip().split()
        file5.write(str(AS_List_sort.index(int(AS1))) + '\t' + str(AS_List_sort.index(int(AS2))) + '\n')

file7 = open(Output_File_parents_path + "line_graph_edge_label.txt", "w")
with open(File_parents_path + "line_graph_edge_label.txt") as file6:
    for line in file6:
        AS1, AS2, rel = line.strip().split()
        file7.write(str(AS_List_sort.index(int(AS1))) + '\t' + str(AS_List_sort.index(int(AS2))) + '\t' + str(rel) + '\n')

file9 = open(Output_File_parents_path + "line_weight.txt", "w")
with open(File_parents_path + "line_weight.txt") as file8:
    weight_List = []
    for line in file8:
        AS1, AS2, weight = line.strip().split()
        file9.write(str(AS_List_sort.index(int(AS1))) + '\t' + str(AS_List_sort.index(int(AS2))) + '\t' + str(weight) + '\n')








