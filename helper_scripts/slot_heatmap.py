import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('../')

dir_name = 'correlations'

wrs = dict()
for dataset_dir in os.listdir(dir_name):
    dir = os.path.join(dir_name, dataset_dir)
    wrs[dataset_dir] = dict()
    for f in os.listdir(dir):
        if os.path.splitext(f)[0][-2:] != 'wr':
            continue
        f_path = os.path.join(dir, f)
        wrs[dataset_dir][os.path.splitext(f)[0].split('_')[2]] = np.loadtxt(f_path)

for dataset in wrs:
    fig = plt.figure(figsize=(10, 10))
    i = 0
    max_value = max(np.max(x) for x in wrs[dataset].values())
    for model in ['DKVMN', 'DMN', 'ADMN']:#wrs[dataset]:
        plt.subplot(1, len(wrs[dataset]), i + 1)
        img = plt.imshow(wrs[dataset][model], cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=max_value)
        plt.title(model, fontsize=15)
        plt.xlabel('Slot ID', fontsize=14)
        if i == 0:
            plt.ylabel('Exercise ID', fontsize=14)
        i += 1
    fig.colorbar(img)
    plt.tight_layout()
    plt.show()
    #plt.savefig('Plots/outputs/' + str(time.time()) + '_' + dataset + '_heatmap.png', format='pdf')
