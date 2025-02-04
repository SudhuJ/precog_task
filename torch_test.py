import torch
import networkx as nx
import matplotlib.pyplot as plt

x = torch.rand(3, 3)
print(x)

G = nx.complete_graph(5)
nx.draw(G)