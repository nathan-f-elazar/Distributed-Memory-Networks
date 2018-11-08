import pandas as pd
import numpy as np
import csv
import __settings__
import random

min_len = 3
max_len = 1000
num_tr = 4000
num_te = 1000

output_file = 'data/junyi'
skill_col = 'exercise'
prob_col = 'problem_number'
corr_col = 'correct'
id_col = 'user_id'
transaction_col = 'time_done'

df = pd.read_csv(__settings__.base_dir + '../../data/Junyi/junyi_ProblemLog_original/junyi_ProblemLog_original.csv')

print('Dataset loaded, size: {},\t students: {}'.format(df.shape, len(df.groupby(id_col))))

skills = np.unique(df[skill_col])
skill_to_id = {skills[i]: i for i in range(len(skills))}

rows_train = []
rows_test = []
count = 0
for (st_id, st_df) in df.groupby(id_col):
    if st_df.shape[0] <= min_len:
        continue
    st_df = st_df.sort_values(transaction_col, ascending=True)
    if st_df.shape[0] > max_len:
        st_df = st_df[:max_len]
    rows = [[st_df.shape[0]], [skill_to_id[x] for x in st_df[skill_col]], [1 if x else 0 for x in st_df[corr_col]]]
    if count < num_tr:
        rows_train.append(rows)
    elif count < num_tr+num_te:
        rows_test.append(rows)
    else:
        break
    count += 1

random.seed(42)
random.shuffle(rows_train)
rows_train = [x for l in rows_train for x in l]
rows_test = [x for l in rows_test for x in l]

with open(__settings__.base_dir + output_file + '_train.csv', 'w', newline="") as ftr,\
        open(__settings__.base_dir + output_file + '_test.csv', 'w', newline="") as fte:
    writertr = csv.writer(ftr)
    writertr.writerows(rows_train)
    writerte = csv.writer(fte)
    writerte.writerows(rows_test)
