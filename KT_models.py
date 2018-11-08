import tensorflow as tf
from util import _linear, _linearX, _embed, _gates


class KT_Model():
    """
    Abstract class for Knowledge Tracing Models.
    Contains all the data handling methods needed to train models.
    """
    def __init__(self, model):
        self.model = model
        self.dataset = model.dataset
        self.labels = tf.placeholder(tf.float32, [None])
        self.pred = tf.nn.sigmoid(model.logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model.logits, labels=self.labels))
        self.predictions = []

    def get_batch(self, is_valid):
        feed_dict, labels = self.model.get_batch(is_valid)
        feed_dict.update({
            self.labels: labels,
        })
        return feed_dict

    def run_batch(self, sess, pred_op, train_op, is_valid):
        pred, _ = sess.run([pred_op, train_op], feed_dict=self.get_batch(is_valid))
        self.predictions.extend([p for p in pred])

    def reset(self):
        self.predictions = []
        self.model.dataset.reset()


class RNN_Model():
    def LSTM_cell(self, h, c, x, name):
        with tf.variable_scope(name):
            i, o, f = _gates([h, x], 'gates', 3)
            new_h = tf.nn.tanh(_linearX([h, x], 'new_h', h.shape[-1]))
        new_c = f * c + i * new_h
        new_h = tf.nn.tanh(new_c) * o
        return new_h, new_c

    def LSTM(self, prev, curr):
        input_skill = curr
        h, c = prev[1]
        q = tf.mod(input_skill, self.dataset.num_skills)
        v = tf.gather(self.vs, [curr])
        k = tf.gather(self.ks, [q])
        kb = _embed(q, 'kb', self.dataset.num_skills, [1, 1])
        h_out = tf.matmul(h, k, transpose_b=True) + kb
        return h_out, self.LSTM_cell(h, c, v, 'new_h')

    def GRU_cell(self, h, x, name):
        with tf.variable_scope(name):
            r, z = _gates([h, x], 'gates', 2)
            new_h = tf.nn.tanh(_linearX([r*h, x], 'new_h', h.shape[-1]))
        return (1-z) * h + z * new_h

    def GRU(self, prev, curr):
        input_skill = curr
        h = prev[1]
        q = tf.mod(input_skill, self.dataset.num_skills)
        v = tf.gather(self.vs, [curr])
        k = tf.gather(self.ks, [q])
        kb = _embed(q, 'kb', self.dataset.num_skills, [1, 1])
        h_out = tf.matmul(h, k, transpose_b=True) + kb
        return h_out, self.GRU_cell(h, v, 'new_h')

    def DKVMN(self, prev, curr):
        M = prev[1]
        q = tf.mod(curr, self.dataset.num_skills)
        k = tf.gather(self.ks, [q])
        w = tf.gather(self.wrs, [q]).reshape([-1, 1])
        #w = tf.reshape(w, [-1, 1])
        r = tf.matmul(w, M, transpose_a=True)
        h_out = tf.nn.relu(_linearX([r, k], 'h_out', self.d))

        v = tf.gather(self.vs, [curr])
        e = tf.nn.sigmoid(_linear(v, 'e', self.d_v))
        a = tf.nn.tanh(_linear(v, 'a', self.d_v))
        new_h = M * (1-e) + a
        return h_out, new_h

    def DMN(self, prev, curr):
        M = prev[1]
        q = tf.mod(curr, self.dataset.num_skills)
        k = tf.gather(self.ks, [q])
        w = tf.gather(self.wrs, [q])
        r = tf.matmul(w, M)
        h_out = tf.nn.relu(_linearX([r, k], 'h_out', self.d))

        v = tf.gather(self.vs, [curr])
        new_h = self.GRU_cell(M, v * tf.reshape(w, [-1, 1]), 'new_h')
        return h_out, new_h

    def ADMN(self, prev, curr):
        M, h, pre_q = prev[1:]
        q = tf.mod(curr, self.dataset.num_skills)+1
        k = tf.gather(self.ks, [q])
        delta = tf.tile(tf.reshape(tf.not_equal(q, pre_q), [1, 1]), [1, self.d])
        d0 = tf.logical_and(delta, tf.greater(pre_q, tf.zeros([self.N, self.d], dtype=tf.int32)))
        w = tf.reshape(tf.gather(self.wrs, [q]), [-1, 1])
        new_Mv = tf.where(d0, self.GRU_cell(M, tf.concat([w * h], 1), 'M1'), M)
        h = tf.where(delta, tf.matmul(w, new_Mv, transpose_a=True), h)
        h_out = tf.nn.relu(_linearX([h, k], 'h_out', self.d))

        v = _embed(curr, 'v', self.dataset.num_skills * 2, [1, self.d_v])
        new_h = self.GRU_cell(h, v, 'new_h')
        return h_out, new_Mv, new_h, q

    def IADMN(self, prev, curr):
        M, h, pre_q = prev[1:]
        q = tf.mod(curr, self.dataset.num_skills)+1
        k = tf.gather(self.ks, [q]) + tf.gather(self.ks2, [q])
        delta = tf.tile(tf.reshape(tf.not_equal(q, pre_q), [1, 1]), [1, self.d])
        d0 = tf.logical_and(delta, tf.greater(pre_q, tf.zeros([self.N, self.d], dtype=tf.int32)))
        wr = tf.reshape(tf.gather(self.wrs, [q]), [-1, 1])
        ww = tf.reshape(tf.gather(self.wws, [pre_q]), [-1, 1])
        new_Mv = tf.where(d0, self.GRU_cell(M, tf.concat([ww * h], 1), 'M1'), M)
        h = tf.where(delta, tf.matmul(wr, new_Mv, transpose_a=True), h)
        h_out = tf.nn.relu(_linearX([h, k], 'h_out', self.d))

        v = _embed(curr, 'v', self.dataset.num_skills * 2, [1, self.d_v])
        new_h = self.GRU_cell(h, v, 'new_h')
        return h_out, new_Mv, new_h, q

    def __init__(self, dataset, N, d, d_k=None, d_v=None):
        self.dataset = dataset
        self.d_k = d if d_k is None else d_k
        self.d_v = d if d_v is None else d_v
        self.d = d
        self.N = N
        self.input_q = tf.placeholder(tf.int32, [dataset.batch_size, dataset.max_num_problems, dataset.num_features])
        self.input_size = tf.placeholder(tf.int32, [dataset.batch_size])
        self.initial_h = tf.get_variable('init_h', [1, self.d])
        self.initial_M = tf.get_variable('init_M', [self.N, self.d])
        self.ks = tf.get_variable('ks', [self.dataset.num_skills+1, self.d_k])
        self.ks2 = tf.get_variable('ks2', initializer=self.ks.initialized_value())
        MK = tf.get_variable('MK', [self.d_k, self.N])
        self.wrs = tf.nn.softmax(tf.matmul(self.ks, MK))
        self.wws = tf.nn.softmax(tf.matmul(self.ks2, MK))
        self.vs = tf.get_variable('vs', [self.dataset.num_skills*2, self.d])
        x = tf.unstack(tf.squeeze(self.input_q), dataset.batch_size, axis=0)
        x = [x[i][:self.input_size[i]] for i in range(self.input_size.shape[0])]

        #model, init = self.LSTM, (tf.zeros([1, self.d]), (tf.zeros([1, self.d]), tf.zeros([1, self.d])))
        #model, init = self.DKVMN, (tf.zeros([1, self.d_v]), self.initial_M)
        model, init = self.DMN, (tf.zeros([1, self.d_v]), self.initial_M)
        #model, init = self.ADMN, (tf.zeros([1, self.d_v]), self.initial_M, self.initial_h, 0)
        #model, init = self.IADMN, (tf.zeros([1, self.d_v]), self.initial_M, self.initial_h, 0)

        h_outs = [tf.scan(model, x_, initializer=init)[0][1:] for x_ in x]
        h_outs = tf.squeeze(tf.concat(h_outs, 0), -2)
        _linear(tf.zeros([1, self.d]), 'W', 1)
        self.logits = tf.reshape(_linear(h_outs, 'W', 1), [-1])
        #self.logits = tf.reshape(embeds, [-1])

    def get_batch(self, is_valid):
        x, labels, n = self.dataset.get_data(is_valid).get_batch()
        return {
            self.input_q: x,
            self.input_size: n
        }, labels
