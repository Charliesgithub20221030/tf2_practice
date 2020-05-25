import requests
import json
import numpy as np
import tensorflow as tf
import os


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


data_loader = MNISTLoader()

data = json.dumps({
    "instances": data_loader.test_data[:3].tolist()
})
headers = {'content-type': 'application/json'}
url = 'http://localhost:8080/v1/models/MLP:predict'
resp = requests.post(
    url,
    data=data,
    headers=headers
)
pred = np.array(json.loads(resp.text))
print(pred)
# print(resp)
