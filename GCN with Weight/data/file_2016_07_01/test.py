# Author : Pey
# Time : 2021/9/3 20:18
# File_name : test.py

# --------- Import Model ---------#

# --------- Sub Function ---------#

save_file = open("./line_weight2.txt", "w")
with open("./line_weight.txt", "r") as f:
    for line in f:
        AS1, AS2, weight = line.strip().split()
        save_file.write(AS1 + "\t" + AS2 + "\t" + str(float(weight) + 1) + "\n")
