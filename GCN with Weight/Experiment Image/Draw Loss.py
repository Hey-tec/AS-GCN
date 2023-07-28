# Author : Pey
# Time : 2021/3/23 14:05
# File_name : Draw Loss.py

# --------- Import Model ---------#
from matplotlib import pyplot as plt
import re
# --------- Sub Function ---------#

Train_Accuracy_list = []
Valid_Accuracy_list = []
Test_Accuracy_list = []

def Get_Train_Loss_Accuracy(txt_file):
    try:
        with open(txt_file) as f:
            for line in f:
                if 'std' in line or 'loss' in line:
                    continue
                if 'Optimization' in line:
                    break
                data_list = line.strip().split()
                Train_Accuracy_list.append(100 * float(data_list[3][:-1]))
    except UnicodeDecodeError:
        with open(txt_file, encoding='utf-8') as f:
            for line in f:
                if 'std' in line or 'loss' in line:
                    continue
                if 'Optimization' in line:
                    break
                data_list = line.strip().split()
                Train_Accuracy_list.append(100*float(data_list[3][:-1]))
    print(Train_Accuracy_list)

def Get_Valid_Loss_Accuracy(txt_file):
    try:
        with open(txt_file) as f:
            for line in f:
                if 'std' in line or 'loss' in line:
                    continue
                if 'Optimization' in line:
                    break
                data_list = line.strip().split()
                Valid_Accuracy_list.append(100*float(data_list[5][:-1]))
    except UnicodeDecodeError:
        with open(txt_file, encoding='utf-8') as f:
            for line in f:
                if 'std' in line or 'loss' in line:
                    continue
                if 'Optimization' in line:
                    break
                data_list = line.strip().split()
                Valid_Accuracy_list.append(100*float(data_list[5][:-1]))
    print(Valid_Accuracy_list)

def Get_Test_Loss_Accuracy(txt_file):
    try:
        with open(txt_file) as f:
            for line in f:
                if 'std' in line or 'loss' in line:
                    continue
                if 'Optimization' in line:
                    break
                data_list = line.strip().split()
                Test_Accuracy_list.append(100*float(data_list[7]))
    except UnicodeDecodeError:
        with open(txt_file, encoding='utf-8') as f:
            for line in f:
                if 'std' in line or 'loss' in line:
                    continue
                if 'Optimization' in line:
                    break
                data_list = line.strip().split()
                Test_Accuracy_list.append(100*float(data_list[7]))
    print(Test_Accuracy_list)


def Draw_Acc_Line():
    x1 = range(0, 200)
    y1 = Train_Accuracy_list
    y2 = Valid_Accuracy_list
    y3 = Test_Accuracy_list

    fig, ax = plt.subplots()                        # 创建图实例
    ax.plot(x1, y1, label='Train_Accuracy_list')                  # 作y1 = x 图，并标记此线名为linear
    ax.plot(x1, y2, label='Valid_Accuracy_list')               # 作y2 = x^2 图，并标记此线名为quadratic
    ax.plot(x1, y3, label='Test_Accuracy_list')                   # 作y3 = x^3 图，并标记此线名为cubic
    plt.annotate(r'$Train-Last=%.2lf$' % Train_Accuracy_list[-1], xy=(80, 50), xytext=(150, 70))
    plt.annotate(r'$Valid-Last=%.2lf$' % Valid_Accuracy_list[-1], xy=(80, 50), xytext=(150, 65))
    plt.annotate(r'$Test-Last=%.2lf$' % Test_Accuracy_list[-1], xy=(80, 50), xytext=(150, 60))
    ax.set_xlabel('Epoch')                        # 设置x轴名称 x label
    ax.set_ylabel('Acc')                        # 设置y轴名称 y label
    ax.set_title('Binary Classification with Weight')                     # 设置图名为Simple Plot
    ax.legend()                                     # 自动检测要在图例中显示的元素，并且显示

    plt.show()



# --------- Main Function --------#
if __name__ == "__main__":
    print("Start coding...")
    txt_file = "./first.txt"
    Get_Train_Loss_Accuracy(txt_file)
    Get_Valid_Loss_Accuracy(txt_file)
    Get_Test_Loss_Accuracy(txt_file)
    Draw_Acc_Line()


