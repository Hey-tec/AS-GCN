# Author : Pey
# Time : 2021/4/5 21:54
# File_name : Sparse to Dense.py

# --------- Import Model ---------#
import numpy as np
# --------- Sub Function ---------#
File_parents_path = "./Source Data/file_2018_04_01/"
Output_File_parents_path = "../data/file_2018_04_01/"

with open(File_parents_path + "AS_Features.txt") as file1:
    AS_List = []
    for line in file1:
        elem = line.strip().split()
        AS_List.append(int(elem[0]))
    AS_List_sort = AS_List
    AS_List_sort.sort()

with open(File_parents_path + "AS_Features.txt") as file2:
    AS_Features_Dict = {}
    for line in file2:
        elem = line.strip().split()
        AS_Features_Dict[int(elem[0])] = "\t".join(elem[1:])

file3 = open(Output_File_parents_path + "AS_Features.txt", "w")
for AS in AS_List_sort:
    file3.write(str(AS_List_sort.index(AS)) + '\t' + AS_Features_Dict[AS] + '\n')

with open(File_parents_path + "line_graph_edge.txt") as file4:
    line_List = []
    for line in file4:
        AS1, AS2 = line.strip().split()
        line_List.append([int(AS1), int(AS2)])
    line_List.sort()

file5 = open(Output_File_parents_path + "line_graph_edge.txt", "w")
for line in line_List:
    file5.write(str(AS_List_sort.index(line[0])) + '\t' + str(AS_List_sort.index(line[1])) + '\n')

with open(File_parents_path + "line_graph_edge_label.txt") as file6:
    rel_List = []
    for line in file6:
        AS1, AS2, rel = line.strip().split()
        rel_List.append([int(AS1), int(AS2), int(rel)])
    rel_List.sort()

file7 = open(Output_File_parents_path + "line_graph_edge_label.txt", "w")
for rel in rel_List:
    file7.write(str(AS_List_sort.index(rel[0])) + '\t' + str(AS_List_sort.index(rel[1])) + '\t' + str(rel[2]) + '\n')

with open(File_parents_path + "line_weight.txt") as file8:
    weight_List = []
    for line in file8:
        AS1, AS2, weight = line.strip().split()
        weight_List.append([int(AS1), int(AS2), float(weight)])
    weight_List.sort()

file9 = open(Output_File_parents_path + "line_weight.txt", "w")
for weight in weight_List:
    file9.write(str(AS_List_sort.index(weight[0])) + '\t' + str(AS_List_sort.index(weight[1])) + '\t' + str(weight[2]) + '\n')








