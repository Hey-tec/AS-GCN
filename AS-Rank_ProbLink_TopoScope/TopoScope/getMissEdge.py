from collections import defaultdict
import networkx as nx
for part in ['0', '1']:
    partial_VP = set(open('./Infer_File/chooseVP' + part + '.txt', 'r').read().split('\n'))
    fullVP = set(open('./Infer_File/fullVP.txt', 'r').read().split('\n'))
    fullVP1 = partial_VP & fullVP
    graph1 = nx.Graph()
    graph2 = nx.Graph()
    with open('./Infer_File/asrel' + part + '.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            [asn1, asn2, rel] = line.strip().split('|')
            graph1.add_edge(asn1, asn2)
    with open('./File/asrel.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            [asn1, asn2, rel] = line.strip().split('|')
            graph2.add_edge(asn1, asn2)
    miss = set()
    for edge in graph2.edges():
        (a, b) = edge
        if a in fullVP or b in fullVP:
            if a in graph1.nodes() and b in graph1.nodes():
                miss.add((a, b))                                # 若edge在graph2中存在，边的两端点在graph1中存在，则为miss

    data = []
    pos = defaultdict(set)
    cnt = 0
    with open('./Infer_File/aspaths' + part + '.txt') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            path = line.strip().split("|")
            data.append(path)
            cnt += 1
            for i in range(len(path)):
                pos[path[i]].add(cnt - 1)               # pos 代表该AS在第几条路径出现，并且数字个数代表出现次数

    path = defaultdict(set)
    for edge in miss:
        (a, b) = edge
        local = pos[a] & pos[b]

        for l in local:
            i = data[l].index(a)
            j = data[l].index(b)
            if (i == 0 or j == 0) and data[l][0] in fullVP1:
                continue
            if j - i != 2 and j - i != -2:
                continue
            if i < j:
                path[edge].add('|'.join(data[l][i:j+1]))
            else:
                path[edge].add('|'.join(data[l][j:i+1][::-1]))

    fout = open('./Infer_File/triplet_miss' + part + '.txt', 'w')
    for key in path:
        fout.write('|'.join(key) + '&' + '#'.join(path[key]) + '\n')
    fout.close()