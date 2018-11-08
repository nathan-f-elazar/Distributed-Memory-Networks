from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import itertools
import operator


def print_metrics(rmse, auc, r2):
    print("rmse: {}\tauc: {}\tr2: {}".format(rmse, auc, r2))


def score(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_ = auc(fpr, tpr)
    r2 = r2_score(labels, preds)

    return auc_, r2


def score_skills(labels, preds, skills):
    data = sorted([x for x in zip(labels, preds, skills)], key=lambda x: x[2])
    data = [list(group) for key, group in itertools.groupby(data, operator.itemgetter(2))]
    num_skills = data[-1][0][-1] + 1
    aucs = [0] * num_skills
    r2s = [0] * num_skills
    for skill in data:
        if len(skill) <= 0:
            continue
        labels = [s[0] for s in skill]
        preds = [s[1] for s in skill]
        r2 = r2_score(labels, preds)
        r2s[skill[0][-1]] = r2
        if len(skill) <= 1:
            continue
        fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
        auc_ = auc(fpr, tpr)
        aucs[skill[0][-1]] = auc_

    return aucs, r2s


def plot_scores(labels, preds, feature):
    i = 0
    student_ers = []
    student_ids = []
    while i < len(feature):
        l = []
        p = []
        id = feature[i]
        while i < len(feature) and id == feature[i]:
            l.append(labels[i])
            p.append(preds[i])
            i += 1
        student_ers.append(sqrt(mean_squared_error(l, p)))
        student_ids.append(id)

    return student_ers, student_ids
