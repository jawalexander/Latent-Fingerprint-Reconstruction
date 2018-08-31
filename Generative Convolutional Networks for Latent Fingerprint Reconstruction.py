import tensorflow as tf
import numpy as np
import pickle
import os
import skimage
from skimage import data, transform
import matplotlib.pyplot as plt
import fnmatch

def load_data(data_directory):
    working_directories = []
    testing_directories = []
    for d in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, d)):
            temp = os.path.join(data_directory, d)
            if fnmatch.fnmatch(d, 'Impression*'):
                for f in os.listdir(temp):
                    working_directories.append(os.path.join(temp, f))
            elif fnmatch.fnmatch(d, 'Fingerprints'):
                for f in os.listdir(temp):
                    testing_directories.append(os.path.join(temp, f))
                    
    images = []
    labels = []
    groundTruths = []
    for d in working_directories:
        for file_name in os.listdir(d):
            images.append(data.imread(os.path.join(d, file_name)))
            labels.append(int(file_name.split('.')[0]))
    for d in testing_directories:
        for file_name in os.listdir(d):
            groundTruths.append(data.imread(os.path.join(d, file_name)))
    return images, labels, groundTruths

root_path = os.getcwd()
data_directory = os.path.join(root_path, 'Temp_Fingerprints')
fingerprints_pickle = os.path.join(root_path, 'fingerprints.pickle')
labels_pickle = os.path.join(root_path, 'labels.pickle')
groundTruths_pickle = os.path.join(root_path, 'groundTruths.pickle')

flag = input('Load from Scratch?Y or N')
if flag == 'N':
    print('loading data from pickle....')
    with open(fingerprints_pickle, 'rb') as read:
        images = pickle.load(read)
    with open(labels_pickle, 'rb') as read:
        labels = pickle.load(read)
    with open(groundTruths_pickle, 'rb') as read:
        groundTruths = pickle.load(read)
else:
    print('loading data from scratch....')
    images, labels, groundTruths = load_data(data_directory)
    with open(fingerprints_pickle, 'wb') as write:
        pickle.dump(images, write)
    with open(labels_pickle, 'wb') as write:
        pickle.dump(labels, write)
    with open(groundTruths_pickle, 'wb') as write:
        pickle.dump(groundTruths, write)

images = np.array([transform.resize(image, (320, 256, 1)) for image in images], dtype=np.float32)
labels = np.array(labels)
groundTruths = np.array([transform.resize(image, (320, 256, 1)) for image in groundTruths], dtype=np.float32)

''' AUTO-ENCODER '''

input_layer = tf.placeholder(dtype=tf.float32, shape=(None, 320, 256, 1))
targets = tf.placeholder(dtype=tf.float32, shape=(None, 320, 256, 1))
learning_rate = 0.005

''' ENCODER '''

# conv1.size = [#batch_size, 320, 256, 128]
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=128,
    kernel_size=13,
    padding='same',
    activation=tf.nn.relu)

# pool1.size = [#batch_size, 160, 128, 128]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

# conv2.size = [#batch_size, 160, 128, 256]
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=256,
    kernel_size=11,
    padding='same',
    activation=tf.nn.relu)

# pool2.size = [#batch_size, 80, 64, 256]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

# conv3.size = [#batch_size, 80, 64, 256]
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=256,
    kernel_size=9,
    padding='same',
    activation=tf.nn.relu)

# pool3.size = [#batch_size, 40, 32, 256]
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2)

# conv4.size = [#batch_size, 40, 32, 512]
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=512,
    kernel_size=7,
    padding='same',
    activation=tf.nn.relu)

# pool4.size = [#batch_size, 20, 16, 512]
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2)

# conv5.size = [#batch_size, 20, 16, 1024]
conv5 = tf.layers.conv2d(
    inputs=pool4,
    filters=1024,
    kernel_size=5,
    padding='same',
    activation=tf.nn.relu)

# encoded.size = [#batch_size, 10, 8, 1024]
encoded = tf.layers.max_pooling2d(inputs=conv5, pool_size=2, strides=2)

''' DECODER '''

upsample1 = tf.image.resize_images(encoded, size=(20, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# deconv1.size = [#batch_size, 20, 16, 1024]
deconv1 = tf.layers.conv2d(
    inputs=upsample1,
    filters=1024,
    kernel_size=5,
    padding='same',
    activation=tf.nn.leaky_relu)

upsample2 = tf.image.resize_images(deconv1, size=(40, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# deconv2.size = [#batch_size, 40, 32, 512]
deconv2 = tf.layers.conv2d(
    inputs=upsample2,
    filters=512,
    kernel_size=7,
    padding='same',
    activation=tf.nn.leaky_relu)

upsample3 = tf.image.resize_images(deconv2, size=(80, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# deconv3.size = [#batch_size, 80, 64, 256]
deconv3 = tf.layers.conv2d(
    inputs=upsample3,
    filters=256,
    kernel_size=9,
    padding='same',
    activation=tf.nn.leaky_relu)

upsample4 = tf.image.resize_images(deconv3, size=(160, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# deconv4.size = [#batch_size, 160, 128, 256]
deconv4 = tf.layers.conv2d(
    inputs=upsample4,
    filters=256,
    kernel_size=11,
    padding='same',
    activation=tf.nn.leaky_relu)

upsample5 = tf.image.resize_images(deconv4, size=(320, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# deconv5.size = [#batch_size, 320, 256, 128]
deconv5 = tf.layers.conv2d(
    inputs=upsample5,
    filters=128,
    kernel_size=13,
    padding='same',
    activation=tf.nn.leaky_relu)

# logits.size = [#batch_size, 320, 256, 1]
logits = tf.layers.conv2d(
    inputs=deconv5,
    filters=1,
    kernel_size=5,
    padding='same',
    activation=None)

# decoded.size = [#batch_size, 320, 256, 1]
decoded = tf.nn.sigmoid(logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
global_step = tf.Variable(0, name='global_step', trainable=False)

cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
images_length = images.shape[0]
reconstructed_images = np.zeros(images.shape)
epochs = 10
batch_size = 10
flag = input('Train the model?Y or N')
if flag == 'Y':
    saver = tf.train.Saver()
    for epoch in range(epochs):
        epoch_cost = 0
        for i in range(images_length//batch_size - 1):
            batch_cost, _ = sess.run([cost, opt],feed_dict={input_layer:images[i*batch_size:(i+1)*batch_size],targets:groundTruths[i*batch_size:(i+1)*batch_size]})
            epoch_cost += batch_cost
        print(epoch_cost)
    saver.save(sess, './accurate_fingerprint_model100', global_step=global_step)
    print('Saved the model.')
    for i in range(images_length//batch_size - 1):
        reconstructed_images[i*batch_size:(i+1)*batch_size] = sess.run(decoded, feed_dict={input_layer:images[i*batch_size:(i+1)*batch_size]})
else:
    saver = tf.train.import_meta_graph('./accurate_fingerprint_model100-290.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    for i in range(images_length//batch_size - 1):
        reconstructed_images[i*batch_size:(i+1)*batch_size] = sess.run(decoded, feed_dict={input_layer:images[i*batch_size:(i+1)*batch_size]})
sess.close()

for i in range(1, 6):
    plt.subplot(2, 5, i)
    plt.imshow(images[i].reshape([320, 256]), cmap='gray')
    plt.axis('off')
for i in range(1, 6):
    plt.subplot(2, 5, 5+i)
    plt.imshow(reconstructed_images[i].reshape([320, 256]), cmap='gray')
    plt.axis('off')
plt.show()
