import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self, chinese):
        path = tf.keras.utils.get_file('nietzsche.txt',
                                       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        if chinese:
            path = 'mds.txt'
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()\
                .replace('.', '. ')\
                .replace(',', ', ')\
                .replace('(', '')\
                .replace(')', '')\
                .replace('=', '')\
                .replace('--', '')\
                .replace('.', '')\
                .replace(',', '')\
                .replace(':', '')\
                .replace(';', '')\
                .replace('"', '')\
                .replace('_', '')\
                .replace('\n', ' ')\
                .replace('，', ' ')\
                .replace('。', ' ')\
                .replace('?', '')\
                .split(' ')
        self.raw_text = [w for w in self.raw_text if w != '']
        self.words = sorted(list(set([
            w
            for w in self.raw_text
            if w != ''])))
        print('number of unique words %d' % (len(self.words)))
        self.word_indices = dict((w, i) for i, w in enumerate(self.words))
        self.indices_word = dict((i, w) for i, w in enumerate(self.words))
        self.text = [self.word_indices[w] for w in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_word = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index: index+seq_length])
            next_word.append(self.text[index+seq_length])
        return np.array(seq), np.array(next_word)


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cell = tf.keras.layers.LSTMCell(256)
        self.dense = tf.keras.layers.Dense(self.num_chars)

    @tf.function
    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)
        state = self.cell.get_initial_state(
            batch_size=self.batch_size, dtype=tf.float32)

        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
            logits = self.dense(output)
            if from_logits:
                return logits
            else:
                return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits/temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :]) for i in range(batch_size.numpy())])

    def load(self, filename):
        self = tf.saved_model.load(filename)


def run_train(chinese=False):
    # training
    num_batchs = 1000
    seq_length = 40
    batch_size = 50
    learning_rate = 1e-3

    data_loader = DataLoader(chinese)

    model = RNN(num_chars=len(data_loader.words),
                batch_size=batch_size,
                seq_length=seq_length)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_batchs):
        x, y = data_loader.get_batch(seq_length, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    model.save('pretrain_rnn.SavedModel', save_format='tf')


def generating(chinese=False):
    # generating by temperature

    num_batchs = 1000
    seq_length = 40
    batch_size = 50
    learning_rate = 1e-3

    data_loader = DataLoader(chinese)

    model = RNN(num_chars=len(data_loader.words),
                batch_size=batch_size,
                seq_length=seq_length)
    model.load('pretrain_rnn.SavedModel')

    x_, _ = data_loader.get_batch(seq_length, 1)
    for diversity in [1.0, 1.2]:
        x = x_
        print('diversity %f' % diversity)

        for t in range(400):
            y_pred = model.predict(x, diversity)
            print(data_loader.indices_word[y_pred[0]], end=' ', flush=True)
            x = np.concatenate(
                [x[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
        print()
