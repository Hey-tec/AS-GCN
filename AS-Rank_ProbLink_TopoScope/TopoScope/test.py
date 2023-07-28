#  -*- coding: utf-8  -*-
# Author : Pey
# Time : 2021/2/9 11:39
# File_name : test.py

# --------- Import Model ---------#
from collections import defaultdict
# --------- Sub Function ---------#

file1 = open("./Infer_File/asrel0.txt", "r")
file2 = open("./Infer_File/asrel1.txt", "r")
file = open("./Infer_File/All_links.txt", "w")

# --------- Main Function --------#
if __name__ == "__main__":
    print("Start coding...")
    for line in file1:
        if "#" in line:
            continue
        path = line.split("|")
        file.write(path[0] + "\t" + path[1] + "\n")
    for line in file2:
        if "#" in line:
            continue
        path = line.split("|")
        file.write(path[0] + "\t" + path[1] + "\n")