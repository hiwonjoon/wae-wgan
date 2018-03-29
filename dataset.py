import sys
import os
import numpy as np
import tensorflow as tf

class MNIST :
    TARGET_DIR = '_dataset/mnist/'
    @staticmethod
    def _load_dataset(labels=True):
        if not os.path.exists(MNIST.TARGET_DIR):
            os.makedirs(MNIST.TARGET_DIR)

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, MNIST.TARGET_DIR+filename)
        import gzip
        def load_mnist_images(filename):
            if not os.path.exists(MNIST.TARGET_DIR+filename):
                download(filename)
            with gzip.open(MNIST.TARGET_DIR+filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28).transpose(0,2,3,1)
            return data / np.float32(255)
        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(MNIST.TARGET_DIR+filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            label = data.reshape(-1)
            return label
        if( labels ) :
            return load_mnist_images('train-images-idx3-ubyte.gz'), load_mnist_labels('train-labels-idx1-ubyte.gz')
        else :
            return load_mnist_images('train-images-idx3-ubyte.gz')

    def __init__(self,batch_size,train_epoch=None) :
        self.ims, self.labels = MNIST._load_dataset()
        self.train_ims, self.train_labels = self.ims[:50000], self.labels[:50000]
        self.valid_ims, self.valid_labels = self.ims[50000:], self.labels[50000:]

        train_ds = tf.data.Dataset.from_tensor_slices((self.train_ims,self.train_labels))\
                                  .shuffle(1000)\
                                  .repeat(train_epoch)\
                                  .batch(batch_size)

        valid_ds = tf.data.Dataset.from_tensor_slices((self.valid_ims,self.valid_labels))\
                                  .batch(batch_size)

        train_iterator = train_ds.make_initializable_iterator()
        valid_iterator = valid_ds.make_initializable_iterator()

        self.train_data_init_op,self.train_data_op = train_iterator.initializer, train_iterator.get_next()
        self.valid_data_init_op,self.valid_data_op = valid_iterator.initializer, valid_iterator.get_next()


if __name__=="__main__":
    mnist = MNIST(16)

    sess = tf.InteractiveSession()

    sess.run([mnist.train_data_init_op,mnist.valid_data_init_op])
    for _ in range(10):
        ims,labels = sess.run(mnist.train_data_op)
        print( ims.shape, labels )

    cnt = 0
    try:
        while(True):
            ims,labels = sess.run(mnist.valid_data_op)
            cnt += len(labels)
    except tf.errors.OutOfRangeError:
        print('# of validation images: %d'%cnt)

