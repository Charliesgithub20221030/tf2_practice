import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Linear model
# class Linear(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense = tf.keras.layers.Dense(
#             units=1,
#             activation=None,
#             kernel_initializer=tf.zeros_initializer(),
#             bias_initializer=tf.zeros_initializer()
#         )

#     # call 函數表示，針對實例化的物件進行呼叫，就會執行 i.e. model(x) -> show output
#     def call(self, input):
#         output = self.dense(input)
#         return output


# model = Linear()
# optimizer = tf.keras.optimizers.SGD()

# for i in range(100):
#     with tf.GradientTape() as tape:
#         y_pred = model(x)
#         loss = tf.reduce_sum(tf.square(y_pred - y))
#     # model.variable 是原本 keras.Model 中所定義的，表示「所有模型中的變數」
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# print("model.variables", model.variables)

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


# CNN to mnist
class CNN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7*7*64,))
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output


# MNIST training

# model = MLP()
# model = CNN()

# data_loader = MNISTLoader()
# optimizer = tf.keras.optimizers.Adam(0.001)

# num_epochs = 30
# batch_size = 1000

# num_batches = int(data_loader.num_train // batch_size * num_epochs)
# for batch_index in range(num_batches):
#     x, y = data_loader.get_batch(batch_size)
#     with tf.GradientTape() as tape:
#         y_pred = model(x)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(
#             y_true=y, y_pred=y_pred)
#         loss = tf.reduce_mean(loss)
#         print("batch %d: loss %f" % (batch_index, loss.numpy()))
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# evaluation
# sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# num_batches = int(data_loader.num_test // batch_size)
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * \
#         batch_size, (batch_index+1) * batch_size
#     y_pred = model.predict(data_loader.test_data[start_index:end_index])

# sparse_categorical_accuracy.update_state(
#     y_true=data_loader.test_label[start_index:end_index], y_pred=y_pred)
# print("test accuracy : %f" % sparse_categorical_accuracy.result())

# MobileNetV2
num_batches = 1000
batch_size = 50
learning_rate = 0.001

# dataset = tfds.load('tf_flowers', split=tfds.Split.TRAIN, as_supervised=True)
# dataset = dataset.map(lambda img, label: (tf.image.resize(
#     img, [224, 224])/255.0, label)).shuffle(1024).batch(32)

# model = tf.keras.applications.MobileNetV2(weights=None, classes=5)

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# for images, labels in dataset:
#     with tf.GradientTape() as tape:
#         labels_pred = model(images)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(
#             y_true=labels, y_pred=labels_pred)
#         loss = tf.reduce_mean(loss)
#         print("loss %f" % loss.numpy())
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(grads_and_vars=zip(
#         grads, model.trainable_variables))


# evaluation

# sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# dataset_test = tfds.load(
#     'tf_flowers', split=tfds.Split.TEST, as_supervised=True)
# dataset_test = dataset_test.map(lambda img, label: (tf.images.resize(
#     img, [224, 224])/255.0, label)).shuffle(1024).batch(32)
# num_batches = int(len(dataset_test) // batch_size)
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * \
#         batch_size, (batch_index+1) * batch_size
#     y_pred = model.predict(dataset_test[start_index:end_index])

# sparse_categorical_accuracy.update_state(
#     y_true=dataset_test[start_index:end_index], y_pred=y_pred)
# print("test accuracy : %f" % sparse_categorical_accuracy.result())


# convolution principle

# image = np.array([[
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 2, 1, 0],
#     [0, 0, 2, 2, 0, 1, 0],
#     [0, 1, 1, 0, 2, 1, 0],
#     [0, 0, 2, 1, 1, 0, 0],
#     [0, 2, 1, 1, 2, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0]
# ]], dtype=np.float32)
# image = np.expand_dims(image, axis=-1)

# W = np.array([[
#     [0, 0, -1],
#     [0, 1, 0],
#     [-2, 0, 2]]], dtype=np.float32)
# b = np.array([1], dtype=np.float32)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(
#         1,
#         kernel_size=[3, 3],
#         kernel_initializer=tf.constant_initializer(W),
#         bias_initializer=tf.constant_initializer(b)
#     )
# ])

# output = model(image)
# print(tf.squeeze(output))  # remove redundent dimensions


# document generator
# 尼采文章

# Keras functional API (Sequential API 已經很常用，不用試，都要compile)


def functionalTraining():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizer.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label,
              epochs=num_epochs, batch_size=batch_size)


def v1training():
    model = CNN()
    data_loader = MNISTLoader()
    optimizer = tf.compat.v1.train.AdamOptimizer(.001)
    num_batches = int(data_loader.num_batch // batch_size*num_epochs)
    x_placeholder = tf.compat.v1.placeholder(name='x', shape=[None, 28, 28, 1])
    y_placdholder = tf.compat.v1.placeholder(
        name='y', shape=[None], dtype=tf.int32)
    y_pred = model(x_placeholder)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=y_placdholder, y_pred=y_pred)
    loss = tf.reduce_mean(loss)
    train_op = optimizer.minimize(loss)
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for batch_index in range(num_batches):
            x, y = data_loader.get_batch(batch_size)
            _, loss_value = sess.run([train_op, loss], fees_dict={
                                     x_placeholder: x, y_placdholder: y})
            print('batch %d loss %f' % (batch_index, loss_value))

        num_batches = int(data_loader.num_test//batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * \
                batch_size, (batch_index+1)*batch_size
            y_pred = model.predict(
                data_loader.test_data[start_index:end_index])
            sess.run(sparse_categorical_accuracy.update(
                y_true=data_loader.test_label[start_index:end_index], y_pred=y_pred))
            print("test accuracy: %f" % sess.run(
                sparse_categorical_accuracy.result()))


def array_write_read():
    arr = tf.TensorArray(dtype=tf.float32, size=3)
    arr = arr.write(0, tf.constant(0.0))
    arr = arr.write(1, tf.constant(1.0))
    arr = arr.write(2, tf.constant(2.0))
    arr_0 = arr.read(0)
    arr_1 = arr.read(1)
    arr_2 = arr.read(2)
    return arr_0, arr_1, arr_2


print(array_write_read())

# run_train()
# 待確認 tf.wiki 網址
