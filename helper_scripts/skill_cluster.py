from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import __settings__
import os

os.chdir('../')
np.random.seed(__settings__.random_seed)

dir_name = 'graphs'
f_name = 'A1526860391.819854_2009_DMN_10_cor_wr'

wr = np.loadtxt(dir_name + '/' + f_name+'.txt')
X = TSNE(n_components=2, learning_rate=250, perplexity=20, early_exaggeration=5, method='exact', n_iter=500).fit_transform(wr)

plt.scatter(X[:, 0], X[:, 1], c=['C' + str(np.argmax(wr[i])) for i in range(len(wr))], edgecolors='Black', alpha=0.8)
for i in range(len(X)):
    txt = plt.text(X[i, 0], X[i, 1], str(i+1), size=12, color='black')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='C' + str(np.argmax(wr[i])))])
plt.axis('off')
plt.show()

cols = 3

with open('assist2009_updated_skill_mapping.txt', 'r') as f:
    tags = [x.split('\t') for x in f.readlines()]

id_to_tag = {int(tags[i][0])-1: tags[i][1] for i in range(len(tags))}
clustered_tags = [(i, id_to_tag[i], str(np.argmax(wr[i]))) for i in range(len(tags))]

sorted_tags = sorted(clustered_tags, key=lambda x: x[2])
tags = defaultdict(list)
for (exercise, tag, clust) in sorted_tags:
    tags[int(int(clust) / (wr.shape[1] / cols))].append((exercise, tag, clust))

print('\\hline')
ns = [len(x) for x in tags.values()]
for i in range(max(ns)):
    line = ''
    for j in range(cols):
        try:
            line += str(tags[j][i][0]+1) + ' \quad ' + tags[j][i][1].strip().replace('&', 'and') + ' &'
        except IndexError:
            line += ' &'
            continue
    print(line[:-1] + '\\\\')
    for j in range(cols):
        if i < len(tags[j])-1:
            if tags[j][i][2] != tags[j][i+1][2]:
                print('\\cline{' + str(j+1) + '-' + str(j+1) + '}')
print('\\hline')