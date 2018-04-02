import sys
import os
import numpy as np
import tensorflow as tf
import pathlib
import glob

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

class CelebA:
    TARGET_DIR = '_datasets/CelebA'

    @staticmethod
    def maybe_download_and_extract():
        from scripts import celeba_download as d
        def check_avail():
            p = pathlib.Path(CelebA.TARGET_DIR)
            if( not p.exists() ):
                return 0
            i = p/'img_align_celeba'
            if( not i.exists() ):
                return 1
            i = p/'splits'
            if( not i.exists() ):
                return 2
            return 3

        fns = [d.prepare_data_dir,d.download_celeb_a,d.add_splits]
        start = check_avail()
        for fn in fns[start:]:
            fn(CelebA.TARGET_DIR)

    @staticmethod
    def load_img_and_preprocess(filename):
        image_string = tf.read_file(filename)
        im = tf.image.decode_image(image_string,channels=3)
        im = tf.image.crop_to_bounding_box(im, 50, 25, 128, 128)
        im = tf.image.resize_images(im, [64, 64])

        return tf.cast(im,tf.float32)/255.0, tf.constant(-1,tf.int32)

    def __init__(self,batch_size,train_epoch=None):
        CelebA.maybe_download_and_extract()

        def _make_ds(set_name):
            files = glob.glob( str(pathlib.Path(CelebA.TARGET_DIR)/'splits'/set_name/'*.*') )

            ds = \
                tf.data.Dataset.from_tensor_slices(files)\
                               .map(CelebA.load_img_and_preprocess,num_parallel_calls=4)

            if set_name == 'train':
                ds = ds.shuffle(1000)\
                       .repeat(train_epoch)\
                       .batch(batch_size)
            else:
                ds = ds.batch(batch_size)

            return ds

        train_iterator = _make_ds('train').make_initializable_iterator()
        valid_iterator = _make_ds('valid').make_initializable_iterator()
        test_iterator = _make_ds('test').make_initializable_iterator()

        self.train_data_init_op,self.train_data_op = train_iterator.initializer, train_iterator.get_next()
        self.valid_data_init_op,self.valid_data_op = valid_iterator.initializer, valid_iterator.get_next()
        self.test_data_init_op,self.test_data_op = test_iterator.initializer, test_iterator.get_next()


if __name__=="__main__":
    def test_ds(ds):
        sess = tf.InteractiveSession()

        sess.run([ds.train_data_init_op,ds.valid_data_init_op])
        for _ in range(10):
            ims,labels = sess.run(ds.train_data_op)
            print( ims.shape, labels )

        valid_labels = []
        try:
            while(True):
                ims,labels = sess.run(ds.valid_data_op)
                valid_labels.append(labels)
        except tf.errors.OutOfRangeError:
            valid_labels = np.concatenate(valid_labels,axis=0)
            labels, count = np.unique(valid_labels,return_counts=True)
            print('# of validation images: %d'%len(valid_labels))
            print('labels/count')
            for l,c in zip(labels,count):
                print('%d: %d'%(l,c))
        sess.close()

    mnist = MNIST(16)
    test_ds(mnist)

    celeba = CelebA(16)
    test_ds(celeba)
