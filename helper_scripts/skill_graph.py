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


wr_f_name = 'A1527044025.0787804_2009_IDMN_10_cor_wr'
ww_f_name2 = 'A1527044025.0787804_2009_IDMN_10_cor_ww'

wr = np.loadtxt(dir_name + '/' + wr_f_name + '.txt')[1:-1]
ww = np.loadtxt(dir_name + '/' + ww_f_name2 + '.txt')[1:-1]

dependencies = np.matmul(wr, ww.transpose())
arrows = np.argmax(dependencies, 1)
strengths = np.max(dependencies, 1)

X = TSNE(n_components=2, learning_rate=250, perplexity=14, early_exaggeration=1, method='exact', n_iter=500).fit_transform(wr) * 2

plt.scatter(X[:, 0], X[:, 1], c=['C' + str(np.argmax(wr[i])) for i in range(len(wr))], edgecolors='Black', alpha=0.8)#np.argmax(wr, 1))#c.labels_)

prox = 2.1

plotted_points = []
for i in range(len(X)):
    c = 'C' + str(np.argmax(wr[i]))
    plt.annotate('', xytext=(X[i, 0], X[i, 1]), xy=(X[arrows[i], 0], X[arrows[i], 1]), arrowprops=dict(color=c, shrink=0.05, width=0.005, headwidth=5, alpha=strengths[i]*2))
for i in range(len(X)):
    c = 'C' + str(np.argmax(wr[i]))
    new_point = X[i]
    if len(plotted_points) > 0:
        closest_point = plotted_points[np.argsort(np.sqrt(np.sum(np.square(np.array(plotted_points) - new_point), 1)))[0]]
        while np.sqrt(np.sum(np.square(closest_point - new_point))) < prox:
            new_point += ((new_point - closest_point) / np.linalg.norm(new_point - closest_point)) * prox
            closest_point = plotted_points[np.argsort(np.sqrt(np.sum(np.square(np.array(plotted_points) - new_point), 1)))[0]]

    plotted_points.append(new_point)
    txt = plt.text(new_point[0], new_point[1], str(i+1), size=12, color='black')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='C' + str(np.argmax(wr[i])))])
plt.draw()
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