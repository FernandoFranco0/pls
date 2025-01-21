import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt


with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
    G = pickle.load(f)


# invert the g.dregree dict, put the nodes in a list
# degree_dict = dict(G.degree())
# degree_dict_inv = {}
# for k, v in degree_dict.items():
#     if v not in degree_dict_inv:
#         degree_dict_inv[v] = []
#     degree_dict_inv[v].append(k)

# print(degree_dict_inv.keys())
# print(degree_dict_inv[14][0])

# a = []
# b = []
# for edge in G.edges():
#     if edge[0] == degree_dict_inv[14][0]:
#         a.append(edge[1])
#         if edge[1] in b:
#             print(f'foun in b: {edge[1]}')
#     elif edge[1] == degree_dict_inv[14][0]:
#         b.append(edge[0])
#         if edge[0] in a:
#             print(f'foun in a: {edge[0]}')

# print(a)
# print(b)
m = 0
count = 0
for n in G.nodes():
    for nei in G.neighbors(n):
        count += 1
    m = max(m, count)
    if count >= 9:
        print(n)
    count = 0
print(m)

G.neighbors("44042532")

a = []
b = []
for edge in G.edges():
    if edge[0] == "44042532":
        a.append(edge[1])
        if edge[1] in b:
            print(f'foun in b: {edge[1]}')
    elif edge[1] == "44042532":
        b.append(edge[0])
        if edge[0] in a:
            print(f'foun in a: {edge[0]}')

print(a)
print(b)

print(G.number_of_nodes())

# nx.draw_spectral(G, with_labels=False, node_size=10)
# plt.show()