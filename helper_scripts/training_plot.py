import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.chdir('../')

base_dir_name = 'results'

color_list = {'LSTM': ((1, 0, 0, 1), (1, 0.5, 0.5, 0.8)),
              'DKVMN': ((0, 1, 0, 1), (0.5, 1, 0.5, 0.8)),
              'DMN': ((0, 0, 1, 1), (0.5, 0.5, 1, 0.8)),
              'ADMN': ((1, 0, 1, 1), (1, 0.5, 1, 0.8))}
line_style = ['s-', 'v--']

datasets = []
dataset_names = []
for dataset_dir in os.listdir(base_dir_name):
    dir_name = os.path.join(base_dir_name, dataset_dir)
    logs = dict()
    dataset_names.append(dataset_dir)
    for dir in os.listdir(dir_name):
        dir_name2 = os.path.join(dir_name, dir)
        for fn in os.listdir(dir_name2):
            f_name = os.path.join(dir_name2, fn)
            if os.path.splitext(fn)[0][-1] == 'y':
                continue
            with open(f_name, 'r') as f:
                train_runs = []
                valid_runs = []
                current_train = [[]]
                current_valid = [[]]
                f.readline()
                train = True
                test = False
                score = 0
                count = 0
                best_score = 0
                for l in f.readlines():
                    if test:
                        test = False
                        train = True
                        continue
                    if len(l.split(';')) > 1:
                        score /= count
                        if score > best_score:
                            best_train = current_train[0]
                            best_valid = current_valid[0]
                            best_score = score
                        current_train = [[]]
                        current_valid = [[]]
                        score = 0
                        count = 0
                        continue
                    sl = l.split('\t')
                    if len(sl) != 2:
                        try:
                            score += float(sl[0].split(' ')[1][1:-1])
                        except IndexError:
                            print('hi')
                        count += 1
                        current_train.append([])
                        current_valid.append([])
                        test = True
                        continue
                        #break
                    auc, r2 = sl
                    auc = float(auc.split(':')[-1][1:])
                    r2 = float(r2.split(':')[-1][1:-1])
                    if train:
                        current_train[-1].append((auc, r2))
                    else:
                        current_valid[-1].append((auc, r2))
                    train = not train
                logs[dir] = (best_train, best_valid)
    datasets.append(logs)
    logs = []

plt.figure(figsize=(14, 14))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
for i in range(len(datasets)):
    plt.subplot(len(datasets) // 2, len(datasets) // 2, i+1)
    for model in datasets[i]:
        if model not in color_list:
            continue
        train = [x[0] for x in datasets[i][model][0]]
        valid = [x[0] for x in datasets[i][model][1]]
        x_axis_values = [int(x) for x in range(len(train))]
        plt.plot(x_axis_values, train, line_style[0], markersize=6, label=model + ' train', color=color_list[model][0], linewidth=2)
        plt.plot(x_axis_values, valid, line_style[1], markersize=6, label=model + ' valid', color=color_list[model][1], linewidth=2)
    plt.title(dataset_names[i], fontsize=15)
    plt.legend(fontsize=15)
    if i % (len(datasets) // 2) == 0:
        plt.ylabel('AUC', fontsize=15)
    if i >= (len(datasets) // 2):
        plt.xlabel('epoch', fontsize=15)
    formatter = ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)

#plt.savefig('Plots/outputs/' + str(time.time()) + '.png', format='pdf')
plt.show()


