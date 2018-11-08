from KT_models import *
from data_loader import *
import gc
import time
np.set_printoptions(threshold=np.nan)
import __settings__
from math import floor
from evaluation import score

random.seed(__settings__.random_seed)

dataset = Dataset('data/assist2009_updated_train.csv', test_file='data/assist2009_updated_test.csv')
#dataset = Dataset('data/STATICS_train.csv', test_file='data/STATICS_test.csv', batch_size=8)
#dataset = Dataset('data/assist2015_train.csv', test_file='data/assist2015_test.csv')
#dataset = Dataset('data/junyi_train.csv', test_file='data/junyi_test.csv')

name = 'experiment'
tmp_file = './models/A' + str(time.time())+'/' + name + '_' + str(time.time())
output_name = './logs/A' + str(time.time()) + '_' + name
output_log_file = output_name + '.txt'
output_sum_file = output_name + '_summary.txt'
output_wr_file = output_name + '_wr' + '.txt'
output_ww_file = output_name + '_ww' + '.txt'

params = {'d': [60],
          'N': [30]}


def run_epoch(sess, model, pred_op, train_op, is_valid):
    dataset_tr = model.dataset.get_data(is_valid)
    dataset_tr.reset()
    model.reset()
    for i in range(floor(len(dataset_tr.students) / model.dataset.batch_size)):
        model.run_batch(sess, pred_op, train_op, is_valid)

    return score(dataset_tr.labels, model.predictions)


def train_model(session, d, N):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=5e-3, epsilon=1e-8)
    model = RNN_Model(dataset, d=d, N=N)
    model = KT_Model(model)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    grads_and_vars = [(tf.clip_by_norm(g, 50), v) for g, v in grads_and_vars if g is not None]
    train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
    saver = tf.train.Saver()

    with tf.variable_scope("training_scope"):
        best_scores = []
        test_scores = []
        for k in range(__settings__.num_folds):
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            best_score = (0, 0)
            dec_count = 0
            i = 0
            while dec_count <= 4:
                auc, r2 = run_epoch(session, model, model.pred, train_op, 'train')
                print("Epoch: %d Train Metrics:\n auc: %.5f \t r2: %.5f" % (i + 1, auc, r2))
                with open(output_log_file, 'a') as f:
                    f.write("Train %d: %.5f \t r2: %.5f\n" % (i+1, auc, r2))

                auc, r2 = run_epoch(session, model, model.pred, tf.no_op(), 'valid')
                print("Epoch: %d Valid Metrics:\n auc: %.5f \t r2: %.5f\n" % (i + 1, auc, r2))
                with open(output_log_file, 'a') as f:
                    f.write("Valid %d: %.5f \t r2: %.5f\n" % (i+1, auc, r2))

                if auc < best_score[0]:
                    dec_count += 1
                else:
                    best_score = (auc, r2)
                    dec_count = 0
                    saver.save(session, tmp_file)
                i += 1

            print('Finish: ' + str(best_score))
            with open(output_log_file, 'a') as f:
                f.write('Finish: ' + str(best_score) + '\n')
            best_scores.append(best_score)

            saver.restore(session, tmp_file)
            np.savetxt(output_wr_file, session.run(model.model.wrs))
            np.savetxt(output_ww_file, session.run(model.model.wws))

            auc, r2 = run_epoch(session, model, model.pred, tf.no_op(), 'test')
            print("Epoch: %d Test Metrics:\n auc: %.5f \t r2: %.5f\n" % (i + 1, auc, r2))
            with open(output_log_file, 'a') as f:
                f.write("Test %d: %.5f \t r2: %.5f\n" % (i + 1, auc, r2))
            test_scores.append((auc, r2))
    best_auc = [x[0] for x in best_scores]
    best_r2 = [x[1] for x in best_scores]
    b_auc = sum(best_auc) / len(best_scores)
    b_r2 = sum(best_r2) / len(best_scores)
    with open(output_sum_file, 'a') as f:
        f.write('\nValid\nd: %d, N: %d\nauc: %.5f, r2: %.5f\n' % (d, m, b_auc, b_r2) + ','.join(['%.5f' % s for s in best_auc]) + '\n' + ','.join(['%.5f' % s for s in best_r2]))

    best_auc = [x[0] for x in test_scores]
    best_r2 = [x[1] for x in test_scores]
    b_auc = sum(best_auc) / len(test_scores)
    b_r2 = sum(best_r2) / len(test_scores)
    with open(output_sum_file, 'a') as f:
        f.write('\nTest\nd: %d, N: %d\nauc: %.5f, r2: %.5f\n' % (d, m, b_auc, b_r2) + ','.join(
            ['%.5f' % s for s in best_auc]) + '\n' + ','.join(['%.5f' % s for s in best_r2]))
    return b_auc

glob_best_score = 0
global_d = 0
global_N = 0
global_i = 0
best_scores = []
for N in params['N']:
    c_best_score = 0
    for d in params['d']:
        print('N: ' + str(N))
        print('d: ' + str(d))
        with open(output_log_file, 'a') as f:
            f.write('N: %d; d: %d\n' % (N, d))
        with tf.Session() as session:
            b_score = train_model(session, d=d, N=N)
        tf.reset_default_graph()
        gc.collect()
        if b_score > glob_best_score:
            glob_best_score = b_score
            global_d = d
            global_N = N
        print('global score: ' + str(glob_best_score) + ', d: ' + str(global_d) + ', m: ' + str(global_N))
        if b_score > c_best_score:
            c_best_score = b_score
