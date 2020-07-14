# coding: utf-8
import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class InitLayer(tf.keras.Model):
    def __init__(self, filters):
        super(InitLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='SAME')
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
    def __call__(self, inputs, training=False):
        return self.pool(tf.nn.relu(self.bn(self.conv(inputs), training=training)))

class ResidualLayer(tf.keras.Model):
    def __init__(self, filters):
        super(ResidualLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='SAME')
        self.bn = tf.keras.layers.BatchNormalization()
    def __call__(self, inputs, training=False):
        return inputs + tf.nn.relu(self.bn(self.conv(inputs), training=training))

class ResidualDownLayer(tf.keras.Model):
    def __init__(self, filters):
        super(ResidualDownLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='SAME')
        self.bn = tf.keras.layers.BatchNormalization()
        self.mix = tf.keras.layers.Conv2D(filters, (1, 1), strides=(2, 2))
    def __call__(self, inputs, training=False):
        return self.mix(inputs) + tf.nn.relu(self.bn(self.conv(inputs), training=training))

class PredictionLayer(tf.keras.Model):
    def __init__(self, num_classes):
        super(PredictionLayer, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes)
        self.softmax = tf.keras.layers.Softmax()
    def __call__(self, inputs, training=False):
        return self.softmax(self.dense(self.pool(inputs)))

def construct_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=[28, 28]))
    model.add(tf.keras.layers.Reshape(target_shape=[28, 28, 1]))
    model.add(tf.keras.layers.Lambda(lambda x: (x - train_images.mean()) / train_images.std()))

    model.add(InitLayer(16))
    model.add(ResidualDownLayer(32))
    model.add(ResidualLayer(32))
    model.add(ResidualDownLayer(64))
    model.add(ResidualLayer(64))
    model.add(PredictionLayer(10))

    opt = tf.keras.optimizers.SGD(0.01, momentum=0.9)
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

# Initial train
model = construct_model()
model.fit(train_images, train_labels, batch_size=128, epochs=64)

metrics = model.evaluate(test_images, test_labels)
for e in dict(zip(model.metrics_names, metrics)).items():
    print('%s: %0.3f' % e)
