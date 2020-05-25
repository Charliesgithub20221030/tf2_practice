
import tensorflow as tf
import numpy as np

# MLP to mnist


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data,
         self.train_label), \
            (self.test_data,
             self.test_label) = mnist.load_data()

        self.train_data = np.expand_dims(
            self.train_data.astype(np.float32) / 255, axis=-1)
        self.test_data = np.expand_dims(
            self.test_data.astype(np.float32) / 255, axis=-1)
        self.num_train, self.num_test = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output


model = MLP()

data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(0.001)

num_epochs = 30
batch_size = 1000

num_batches = int(data_loader.num_train // batch_size * num_epochs)
for batch_index in range(num_batches):
    x, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# evaluation
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * \
        batch_size, (batch_index+1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index:end_index])

sparse_categorical_accuracy.update_state(
    y_true=data_loader.test_label[start_index:end_index], y_pred=y_pred)
print("test accuracy : %f" % sparse_categorical_accuracy.result())
tf.saved_model.save(model, 'mnist_savedmodel')
