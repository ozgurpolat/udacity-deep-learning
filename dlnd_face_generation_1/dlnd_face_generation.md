# Face Generation
In this project, you'll use generative adversarial networks to generate new images of faces.
### Get the Data
You'll be using two datasets in this project:
- MNIST
- CelebA

Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.

If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".


```python
data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)
```

    Found mnist Data
    Found celeba Data
    

## Explore the Data
### MNIST
As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 


```python
show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f5bf845aa90>




![png](output_3_1.png)


### CelebA
The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.


```python
show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))
```




    <matplotlib.image.AxesImage at 0x7f5bf84076a0>




![png](output_5_1.png)


## Preprocess the Data
Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.

The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
## Build the Neural Network
You'll build the components necessary to build a GANs by implementing the following functions below:
- `model_inputs`
- `discriminator`
- `generator`
- `model_loss`
- `model_opt`
- `train`

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0
    

### Input
Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
- Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
- Z input placeholder with rank 2 using `z_dim`.
- Learning rate placeholder with rank 0.

Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)


```python
import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # TODO: Implement Function
    
    input_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32)
    
    return input_real, input_z, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed
    

### Discriminator
Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the generator, tensor logits of the generator).


```python
def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function

    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 28x28x3
        x1 = tf.layers.conv2d(images, 56, 5, strides=2, padding='same')
        relu1 = tf.maximum(0.2 * x1, x1)
        # 14x14x56

        x2 = tf.layers.conv2d(relu1, 112, 5, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)
        relu2 = tf.maximum(0.2 * bn2, bn2)
        # 7x7x112

        # Flatten it
        flat = tf.reshape(relu2, (-1, 7*7*112))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)
```

    Tests Passed
    

### Generator
Implement `generator` to generate an image using `z`. This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.


```python
def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function
    
    with tf.variable_scope("generator", reuse=not is_train) as scope:
        # Fully connected
        fc1 = tf.layers.dense(z, 7*7*256)

        # Reshape
        x1 = tf.reshape(fc1, (-1, 7, 7, 256))
        x1 = tf.layers.batch_normalization(x1, training=True)
        x1 = tf.nn.relu(x1)

        # Second layer 
        x2 = tf.layers.conv2d_transpose(x1, 128, 5, strides=2, padding='SAME')
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        x2 = tf.nn.relu(x2)

        # Ouput layer
        logits = tf.layers.conv2d_transpose(x2, out_channel_dim, 5, strides=2, padding='SAME')
        out = tf.tanh(logits)

    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)
```

    Tests Passed
    

### Loss
Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
- `discriminator(images, reuse=False)`
- `generator(z, out_channel_dim, is_train=True)`


```python
def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    
    g_model = generator(input_z, out_channel_dim, is_train=True)
    d_model_real, d_logits_real = discriminator(input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_loss(model_loss)
```

    Tests Passed
    

### Optimization
Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).


```python
def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # TODO: Implement Function
    
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)
```

    Tests Passed
    

## Neural Network Training
### Show Output
Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()
```

### Train
Implement `train` to build and train the GANs.  Use the following functions you implemented:
- `model_inputs(image_width, image_height, image_channels, z_dim)`
- `model_loss(input_real, input_z, out_channel_dim)`
- `model_opt(d_loss, g_loss, learning_rate, beta1)`

Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.


```python
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # TODO: Build Model
    lr_placeholder = tf.placeholder(tf.float32)
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    steps = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # TODO: Train Model
                steps += 1
            
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                #_ = sess.run(g_opt, feed_dict={input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_z: batch_z, lr_placeholder: learning_rate, input_real: batch_images})
                if steps % 10 == 0:
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    
                    _ = show_generator_output(sess, 25, input_z, data_shape[3], data_image_mode)
                    
                if steps % 100 == 0:
                    show_generator_output(sess, 25, input_z, data_shape[3], data_image_mode)
```

### MNIST
Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.


```python
batch_size = 56
z_dim = 56
learning_rate = 0.002
beta1 = 0.001


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)
```

    Epoch 1/2... Discriminator Loss: 1.2261... Generator Loss: 14.6240
    


![png](output_23_1.png)


    Epoch 1/2... Discriminator Loss: 0.0112... Generator Loss: 7.9779
    


![png](output_23_3.png)


    Epoch 1/2... Discriminator Loss: 2.1599... Generator Loss: 0.2790
    


![png](output_23_5.png)


    Epoch 1/2... Discriminator Loss: 1.9341... Generator Loss: 0.2911
    


![png](output_23_7.png)


    Epoch 1/2... Discriminator Loss: 1.0604... Generator Loss: 0.5696
    


![png](output_23_9.png)


    Epoch 1/2... Discriminator Loss: 0.8654... Generator Loss: 3.1912
    


![png](output_23_11.png)


    Epoch 1/2... Discriminator Loss: 2.6598... Generator Loss: 0.0880
    


![png](output_23_13.png)


    Epoch 1/2... Discriminator Loss: 0.4305... Generator Loss: 3.0117
    


![png](output_23_15.png)


    Epoch 1/2... Discriminator Loss: 1.6471... Generator Loss: 5.0652
    


![png](output_23_17.png)


    Epoch 1/2... Discriminator Loss: 0.1650... Generator Loss: 3.2047
    


![png](output_23_19.png)



![png](output_23_20.png)


    Epoch 1/2... Discriminator Loss: 2.8900... Generator Loss: 0.0801
    


![png](output_23_22.png)


    Epoch 1/2... Discriminator Loss: 0.1838... Generator Loss: 2.7260
    


![png](output_23_24.png)


    Epoch 1/2... Discriminator Loss: 0.6277... Generator Loss: 0.9868
    


![png](output_23_26.png)


    Epoch 1/2... Discriminator Loss: 0.3015... Generator Loss: 1.8584
    


![png](output_23_28.png)


    Epoch 1/2... Discriminator Loss: 0.1050... Generator Loss: 3.3189
    


![png](output_23_30.png)


    Epoch 1/2... Discriminator Loss: 3.8892... Generator Loss: 0.0386
    


![png](output_23_32.png)


    Epoch 1/2... Discriminator Loss: 2.1428... Generator Loss: 0.1774
    


![png](output_23_34.png)


    Epoch 1/2... Discriminator Loss: 1.1135... Generator Loss: 0.5404
    


![png](output_23_36.png)


    Epoch 1/2... Discriminator Loss: 2.9776... Generator Loss: 5.5497
    


![png](output_23_38.png)


    Epoch 1/2... Discriminator Loss: 0.0329... Generator Loss: 6.0530
    


![png](output_23_40.png)



![png](output_23_41.png)


    Epoch 1/2... Discriminator Loss: 0.5494... Generator Loss: 1.2682
    


![png](output_23_43.png)


    Epoch 1/2... Discriminator Loss: 0.2906... Generator Loss: 3.9644
    


![png](output_23_45.png)


    Epoch 1/2... Discriminator Loss: 0.1338... Generator Loss: 4.5578
    


![png](output_23_47.png)


    Epoch 1/2... Discriminator Loss: 0.4091... Generator Loss: 1.8749
    


![png](output_23_49.png)


    Epoch 1/2... Discriminator Loss: 0.4011... Generator Loss: 2.1767
    


![png](output_23_51.png)


    Epoch 1/2... Discriminator Loss: 0.0606... Generator Loss: 5.6216
    


![png](output_23_53.png)


    Epoch 1/2... Discriminator Loss: 2.8060... Generator Loss: 0.1433
    


![png](output_23_55.png)


    Epoch 1/2... Discriminator Loss: 2.5915... Generator Loss: 0.1005
    


![png](output_23_57.png)


    Epoch 1/2... Discriminator Loss: 2.2427... Generator Loss: 0.1373
    


![png](output_23_59.png)


    Epoch 1/2... Discriminator Loss: 0.8525... Generator Loss: 1.9061
    


![png](output_23_61.png)



![png](output_23_62.png)


    Epoch 1/2... Discriminator Loss: 0.2018... Generator Loss: 3.4076
    


![png](output_23_64.png)


    Epoch 1/2... Discriminator Loss: 0.3527... Generator Loss: 3.7156
    


![png](output_23_66.png)


    Epoch 1/2... Discriminator Loss: 0.1926... Generator Loss: 4.3408
    


![png](output_23_68.png)


    Epoch 1/2... Discriminator Loss: 0.3192... Generator Loss: 5.1486
    


![png](output_23_70.png)


    Epoch 1/2... Discriminator Loss: 7.3557... Generator Loss: 5.8863
    


![png](output_23_72.png)


    Epoch 1/2... Discriminator Loss: 3.7650... Generator Loss: 4.8933
    


![png](output_23_74.png)


    Epoch 1/2... Discriminator Loss: 1.3202... Generator Loss: 1.7784
    


![png](output_23_76.png)


    Epoch 1/2... Discriminator Loss: 0.5123... Generator Loss: 2.8005
    


![png](output_23_78.png)


    Epoch 1/2... Discriminator Loss: 2.2692... Generator Loss: 3.5610
    


![png](output_23_80.png)


    Epoch 1/2... Discriminator Loss: 0.3506... Generator Loss: 2.2483
    


![png](output_23_82.png)



![png](output_23_83.png)


    Epoch 1/2... Discriminator Loss: 1.4111... Generator Loss: 2.9186
    


![png](output_23_85.png)


    Epoch 1/2... Discriminator Loss: 0.7546... Generator Loss: 3.0202
    


![png](output_23_87.png)


    Epoch 1/2... Discriminator Loss: 2.4791... Generator Loss: 2.6285
    


![png](output_23_89.png)


    Epoch 1/2... Discriminator Loss: 0.4697... Generator Loss: 3.1709
    


![png](output_23_91.png)


    Epoch 1/2... Discriminator Loss: 0.3695... Generator Loss: 1.7159
    


![png](output_23_93.png)


    Epoch 1/2... Discriminator Loss: 0.2266... Generator Loss: 3.5390
    


![png](output_23_95.png)


    Epoch 1/2... Discriminator Loss: 1.5838... Generator Loss: 0.3395
    


![png](output_23_97.png)


    Epoch 1/2... Discriminator Loss: 0.6441... Generator Loss: 0.9174
    


![png](output_23_99.png)


    Epoch 1/2... Discriminator Loss: 1.1410... Generator Loss: 0.5188
    


![png](output_23_101.png)


    Epoch 1/2... Discriminator Loss: 0.9330... Generator Loss: 0.6984
    


![png](output_23_103.png)



![png](output_23_104.png)


    Epoch 1/2... Discriminator Loss: 2.1377... Generator Loss: 0.1521
    


![png](output_23_106.png)


    Epoch 1/2... Discriminator Loss: 1.3232... Generator Loss: 2.0784
    


![png](output_23_108.png)


    Epoch 1/2... Discriminator Loss: 0.7933... Generator Loss: 3.0372
    


![png](output_23_110.png)


    Epoch 1/2... Discriminator Loss: 0.4522... Generator Loss: 4.1338
    


![png](output_23_112.png)


    Epoch 1/2... Discriminator Loss: 1.9179... Generator Loss: 1.9984
    


![png](output_23_114.png)


    Epoch 1/2... Discriminator Loss: 0.5655... Generator Loss: 3.4448
    


![png](output_23_116.png)


    Epoch 1/2... Discriminator Loss: 0.3893... Generator Loss: 3.2052
    


![png](output_23_118.png)


    Epoch 1/2... Discriminator Loss: 0.1809... Generator Loss: 3.5390
    


![png](output_23_120.png)


    Epoch 1/2... Discriminator Loss: 0.7167... Generator Loss: 0.8763
    


![png](output_23_122.png)


    Epoch 1/2... Discriminator Loss: 0.2429... Generator Loss: 2.2202
    


![png](output_23_124.png)



![png](output_23_125.png)


    Epoch 1/2... Discriminator Loss: 0.2280... Generator Loss: 2.1145
    


![png](output_23_127.png)


    Epoch 1/2... Discriminator Loss: 1.5149... Generator Loss: 0.9747
    


![png](output_23_129.png)


    Epoch 1/2... Discriminator Loss: 1.5202... Generator Loss: 3.5012
    


![png](output_23_131.png)


    Epoch 1/2... Discriminator Loss: 0.9370... Generator Loss: 2.7142
    


![png](output_23_133.png)


    Epoch 1/2... Discriminator Loss: 1.4916... Generator Loss: 0.3355
    


![png](output_23_135.png)


    Epoch 1/2... Discriminator Loss: 1.3563... Generator Loss: 0.3883
    


![png](output_23_137.png)


    Epoch 1/2... Discriminator Loss: 0.6237... Generator Loss: 3.0932
    


![png](output_23_139.png)


    Epoch 1/2... Discriminator Loss: 0.5379... Generator Loss: 1.1635
    


![png](output_23_141.png)


    Epoch 1/2... Discriminator Loss: 0.1729... Generator Loss: 3.0632
    


![png](output_23_143.png)


    Epoch 1/2... Discriminator Loss: 0.3919... Generator Loss: 5.8485
    


![png](output_23_145.png)



![png](output_23_146.png)


    Epoch 1/2... Discriminator Loss: 0.1606... Generator Loss: 3.2614
    


![png](output_23_148.png)


    Epoch 1/2... Discriminator Loss: 0.7284... Generator Loss: 0.8491
    


![png](output_23_150.png)


    Epoch 1/2... Discriminator Loss: 3.6135... Generator Loss: 0.0376
    


![png](output_23_152.png)


    Epoch 1/2... Discriminator Loss: 0.5273... Generator Loss: 1.1443
    


![png](output_23_154.png)


    Epoch 1/2... Discriminator Loss: 0.1906... Generator Loss: 2.5082
    


![png](output_23_156.png)


    Epoch 1/2... Discriminator Loss: 0.4068... Generator Loss: 4.8177
    


![png](output_23_158.png)


    Epoch 1/2... Discriminator Loss: 0.0571... Generator Loss: 4.1781
    


![png](output_23_160.png)


    Epoch 1/2... Discriminator Loss: 1.2073... Generator Loss: 0.7646
    


![png](output_23_162.png)


    Epoch 1/2... Discriminator Loss: 2.0916... Generator Loss: 3.2694
    


![png](output_23_164.png)


    Epoch 1/2... Discriminator Loss: 0.6891... Generator Loss: 2.3470
    


![png](output_23_166.png)



![png](output_23_167.png)


    Epoch 1/2... Discriminator Loss: 0.6513... Generator Loss: 0.9727
    


![png](output_23_169.png)


    Epoch 1/2... Discriminator Loss: 0.0903... Generator Loss: 3.4215
    


![png](output_23_171.png)


    Epoch 1/2... Discriminator Loss: 0.4779... Generator Loss: 1.4508
    


![png](output_23_173.png)


    Epoch 1/2... Discriminator Loss: 0.1090... Generator Loss: 2.8844
    


![png](output_23_175.png)


    Epoch 1/2... Discriminator Loss: 0.1554... Generator Loss: 3.7850
    


![png](output_23_177.png)


    Epoch 1/2... Discriminator Loss: 1.3210... Generator Loss: 0.4144
    


![png](output_23_179.png)


    Epoch 1/2... Discriminator Loss: 1.9325... Generator Loss: 0.2296
    


![png](output_23_181.png)


    Epoch 1/2... Discriminator Loss: 2.1304... Generator Loss: 0.1672
    


![png](output_23_183.png)


    Epoch 1/2... Discriminator Loss: 1.1927... Generator Loss: 0.4287
    


![png](output_23_185.png)


    Epoch 1/2... Discriminator Loss: 0.4643... Generator Loss: 2.2139
    


![png](output_23_187.png)



![png](output_23_188.png)


    Epoch 1/2... Discriminator Loss: 0.2082... Generator Loss: 4.3159
    


![png](output_23_190.png)


    Epoch 1/2... Discriminator Loss: 0.1291... Generator Loss: 2.9374
    


![png](output_23_192.png)


    Epoch 1/2... Discriminator Loss: 3.0922... Generator Loss: 0.0555
    


![png](output_23_194.png)


    Epoch 1/2... Discriminator Loss: 1.1995... Generator Loss: 0.4049
    


![png](output_23_196.png)


    Epoch 1/2... Discriminator Loss: 0.2082... Generator Loss: 4.4679
    


![png](output_23_198.png)


    Epoch 1/2... Discriminator Loss: 0.8123... Generator Loss: 0.8215
    


![png](output_23_200.png)


    Epoch 1/2... Discriminator Loss: 0.1722... Generator Loss: 2.4955
    


![png](output_23_202.png)


    Epoch 1/2... Discriminator Loss: 0.3916... Generator Loss: 1.4538
    


![png](output_23_204.png)


    Epoch 1/2... Discriminator Loss: 0.0620... Generator Loss: 5.9466
    


![png](output_23_206.png)


    Epoch 1/2... Discriminator Loss: 0.0668... Generator Loss: 4.0754
    


![png](output_23_208.png)



![png](output_23_209.png)


    Epoch 1/2... Discriminator Loss: 0.1631... Generator Loss: 3.2057
    


![png](output_23_211.png)


    Epoch 1/2... Discriminator Loss: 0.0767... Generator Loss: 6.3399
    


![png](output_23_213.png)


    Epoch 1/2... Discriminator Loss: 0.0392... Generator Loss: 4.9570
    


![png](output_23_215.png)


    Epoch 1/2... Discriminator Loss: 0.0542... Generator Loss: 4.1162
    


![png](output_23_217.png)


    Epoch 1/2... Discriminator Loss: 0.6757... Generator Loss: 0.9015
    


![png](output_23_219.png)


    Epoch 1/2... Discriminator Loss: 0.0440... Generator Loss: 4.7718
    


![png](output_23_221.png)


    Epoch 1/2... Discriminator Loss: 2.3119... Generator Loss: 0.1238
    


![png](output_23_223.png)


    Epoch 2/2... Discriminator Loss: 2.4561... Generator Loss: 0.1186
    


![png](output_23_225.png)


    Epoch 2/2... Discriminator Loss: 2.1055... Generator Loss: 0.1550
    


![png](output_23_227.png)


    Epoch 2/2... Discriminator Loss: 0.7172... Generator Loss: 0.8548
    


![png](output_23_229.png)



![png](output_23_230.png)


    Epoch 2/2... Discriminator Loss: 0.1442... Generator Loss: 3.7476
    


![png](output_23_232.png)


    Epoch 2/2... Discriminator Loss: 1.2344... Generator Loss: 0.9512
    


![png](output_23_234.png)


    Epoch 2/2... Discriminator Loss: 1.2582... Generator Loss: 2.8309
    


![png](output_23_236.png)


    Epoch 2/2... Discriminator Loss: 0.5331... Generator Loss: 1.3741
    


![png](output_23_238.png)


    Epoch 2/2... Discriminator Loss: 0.1627... Generator Loss: 2.5947
    


![png](output_23_240.png)


    Epoch 2/2... Discriminator Loss: 0.1854... Generator Loss: 2.3831
    


![png](output_23_242.png)


    Epoch 2/2... Discriminator Loss: 3.5496... Generator Loss: 4.3924
    


![png](output_23_244.png)


    Epoch 2/2... Discriminator Loss: 0.6645... Generator Loss: 2.4294
    


![png](output_23_246.png)


    Epoch 2/2... Discriminator Loss: 2.5362... Generator Loss: 0.1127
    


![png](output_23_248.png)


    Epoch 2/2... Discriminator Loss: 1.7130... Generator Loss: 3.5991
    


![png](output_23_250.png)



![png](output_23_251.png)


    Epoch 2/2... Discriminator Loss: 0.5255... Generator Loss: 1.9622
    


![png](output_23_253.png)


    Epoch 2/2... Discriminator Loss: 0.4688... Generator Loss: 1.5192
    


![png](output_23_255.png)


    Epoch 2/2... Discriminator Loss: 0.1277... Generator Loss: 2.9132
    


![png](output_23_257.png)


    Epoch 2/2... Discriminator Loss: 0.2558... Generator Loss: 2.9062
    


![png](output_23_259.png)


    Epoch 2/2... Discriminator Loss: 1.6762... Generator Loss: 0.2651
    


![png](output_23_261.png)


    Epoch 2/2... Discriminator Loss: 0.4974... Generator Loss: 1.1355
    


![png](output_23_263.png)


    Epoch 2/2... Discriminator Loss: 0.2174... Generator Loss: 2.0616
    


![png](output_23_265.png)


    Epoch 2/2... Discriminator Loss: 0.1032... Generator Loss: 3.0918
    


![png](output_23_267.png)


    Epoch 2/2... Discriminator Loss: 0.0646... Generator Loss: 4.2043
    


![png](output_23_269.png)


    Epoch 2/2... Discriminator Loss: 0.1876... Generator Loss: 3.2429
    


![png](output_23_271.png)



![png](output_23_272.png)


    Epoch 2/2... Discriminator Loss: 0.0405... Generator Loss: 6.1928
    


![png](output_23_274.png)


    Epoch 2/2... Discriminator Loss: 0.0513... Generator Loss: 5.1277
    


![png](output_23_276.png)


    Epoch 2/2... Discriminator Loss: 0.2411... Generator Loss: 1.9368
    


![png](output_23_278.png)


    Epoch 2/2... Discriminator Loss: 0.0315... Generator Loss: 4.7680
    


![png](output_23_280.png)


    Epoch 2/2... Discriminator Loss: 0.0284... Generator Loss: 5.6939
    


![png](output_23_282.png)


    Epoch 2/2... Discriminator Loss: 3.1200... Generator Loss: 0.0775
    


![png](output_23_284.png)


    Epoch 2/2... Discriminator Loss: 1.6197... Generator Loss: 0.2821
    


![png](output_23_286.png)


    Epoch 2/2... Discriminator Loss: 1.5342... Generator Loss: 0.3375
    


![png](output_23_288.png)


    Epoch 2/2... Discriminator Loss: 2.3648... Generator Loss: 0.1353
    


![png](output_23_290.png)


    Epoch 2/2... Discriminator Loss: 1.1282... Generator Loss: 0.5075
    


![png](output_23_292.png)



![png](output_23_293.png)


    Epoch 2/2... Discriminator Loss: 0.2790... Generator Loss: 1.8894
    


![png](output_23_295.png)


    Epoch 2/2... Discriminator Loss: 0.8800... Generator Loss: 3.5650
    


![png](output_23_297.png)


    Epoch 2/2... Discriminator Loss: 1.2440... Generator Loss: 3.0623
    


![png](output_23_299.png)


    Epoch 2/2... Discriminator Loss: 0.3928... Generator Loss: 1.4332
    


![png](output_23_301.png)


    Epoch 2/2... Discriminator Loss: 0.0908... Generator Loss: 3.3215
    


![png](output_23_303.png)


    Epoch 2/2... Discriminator Loss: 0.3973... Generator Loss: 1.4660
    


![png](output_23_305.png)


    Epoch 2/2... Discriminator Loss: 0.1252... Generator Loss: 2.7286
    


![png](output_23_307.png)


    Epoch 2/2... Discriminator Loss: 0.3632... Generator Loss: 1.4507
    


![png](output_23_309.png)


    Epoch 2/2... Discriminator Loss: 0.0691... Generator Loss: 3.5829
    


![png](output_23_311.png)


    Epoch 2/2... Discriminator Loss: 0.1031... Generator Loss: 4.2847
    


![png](output_23_313.png)



![png](output_23_314.png)


    Epoch 2/2... Discriminator Loss: 0.1073... Generator Loss: 3.0223
    


![png](output_23_316.png)


    Epoch 2/2... Discriminator Loss: 0.0548... Generator Loss: 4.0912
    


![png](output_23_318.png)


    Epoch 2/2... Discriminator Loss: 0.0365... Generator Loss: 6.5471
    


![png](output_23_320.png)


    Epoch 2/2... Discriminator Loss: 0.0409... Generator Loss: 5.3347
    


![png](output_23_322.png)


    Epoch 2/2... Discriminator Loss: 4.0909... Generator Loss: 5.0975
    


![png](output_23_324.png)


    Epoch 2/2... Discriminator Loss: 1.2978... Generator Loss: 2.6845
    


![png](output_23_326.png)


    Epoch 2/2... Discriminator Loss: 0.9822... Generator Loss: 2.9978
    


![png](output_23_328.png)


    Epoch 2/2... Discriminator Loss: 0.7066... Generator Loss: 3.7135
    


![png](output_23_330.png)


    Epoch 2/2... Discriminator Loss: 0.5901... Generator Loss: 4.1160
    


![png](output_23_332.png)


    Epoch 2/2... Discriminator Loss: 0.5798... Generator Loss: 4.0398
    


![png](output_23_334.png)



![png](output_23_335.png)


    Epoch 2/2... Discriminator Loss: 1.6264... Generator Loss: 5.4876
    


![png](output_23_337.png)


    Epoch 2/2... Discriminator Loss: 0.4664... Generator Loss: 3.7270
    


![png](output_23_339.png)


    Epoch 2/2... Discriminator Loss: 0.2106... Generator Loss: 3.0926
    


![png](output_23_341.png)


    Epoch 2/2... Discriminator Loss: 0.6523... Generator Loss: 0.9424
    


![png](output_23_343.png)


    Epoch 2/2... Discriminator Loss: 1.5874... Generator Loss: 3.2035
    


![png](output_23_345.png)


    Epoch 2/2... Discriminator Loss: 0.5286... Generator Loss: 4.4705
    


![png](output_23_347.png)


    Epoch 2/2... Discriminator Loss: 0.5577... Generator Loss: 1.4467
    


![png](output_23_349.png)


    Epoch 2/2... Discriminator Loss: 0.7975... Generator Loss: 0.7620
    


![png](output_23_351.png)


    Epoch 2/2... Discriminator Loss: 0.1278... Generator Loss: 2.8273
    


![png](output_23_353.png)


    Epoch 2/2... Discriminator Loss: 0.0823... Generator Loss: 3.7050
    


![png](output_23_355.png)



![png](output_23_356.png)


    Epoch 2/2... Discriminator Loss: 0.1968... Generator Loss: 3.6295
    


![png](output_23_358.png)


    Epoch 2/2... Discriminator Loss: 0.1790... Generator Loss: 4.8742
    


![png](output_23_360.png)


    Epoch 2/2... Discriminator Loss: 1.0800... Generator Loss: 3.6032
    


![png](output_23_362.png)


    Epoch 2/2... Discriminator Loss: 0.6580... Generator Loss: 4.2079
    


![png](output_23_364.png)


    Epoch 2/2... Discriminator Loss: 0.6977... Generator Loss: 2.2375
    


![png](output_23_366.png)


    Epoch 2/2... Discriminator Loss: 0.1166... Generator Loss: 2.9945
    


![png](output_23_368.png)


    Epoch 2/2... Discriminator Loss: 0.5067... Generator Loss: 1.1740
    


![png](output_23_370.png)


    Epoch 2/2... Discriminator Loss: 0.0604... Generator Loss: 4.4699
    


![png](output_23_372.png)


    Epoch 2/2... Discriminator Loss: 0.0402... Generator Loss: 5.1654
    


![png](output_23_374.png)


    Epoch 2/2... Discriminator Loss: 0.0397... Generator Loss: 4.7602
    


![png](output_23_376.png)



![png](output_23_377.png)


    Epoch 2/2... Discriminator Loss: 0.0642... Generator Loss: 6.1633
    


![png](output_23_379.png)


    Epoch 2/2... Discriminator Loss: 0.0312... Generator Loss: 5.2793
    


![png](output_23_381.png)


    Epoch 2/2... Discriminator Loss: 0.0312... Generator Loss: 5.5244
    


![png](output_23_383.png)


    Epoch 2/2... Discriminator Loss: 0.0590... Generator Loss: 3.9297
    


![png](output_23_385.png)


    Epoch 2/2... Discriminator Loss: 2.4096... Generator Loss: 0.1321
    


![png](output_23_387.png)


    Epoch 2/2... Discriminator Loss: 1.4176... Generator Loss: 0.3297
    


![png](output_23_389.png)


    Epoch 2/2... Discriminator Loss: 1.9080... Generator Loss: 0.2012
    


![png](output_23_391.png)


    Epoch 2/2... Discriminator Loss: 0.1960... Generator Loss: 3.6243
    


![png](output_23_393.png)


    Epoch 2/2... Discriminator Loss: 0.1920... Generator Loss: 3.8572
    


![png](output_23_395.png)


    Epoch 2/2... Discriminator Loss: 0.1322... Generator Loss: 3.6270
    


![png](output_23_397.png)



![png](output_23_398.png)


    Epoch 2/2... Discriminator Loss: 0.1661... Generator Loss: 2.4033
    


![png](output_23_400.png)


    Epoch 2/2... Discriminator Loss: 0.1705... Generator Loss: 2.2206
    


![png](output_23_402.png)


    Epoch 2/2... Discriminator Loss: 0.0628... Generator Loss: 6.5685
    


![png](output_23_404.png)


    Epoch 2/2... Discriminator Loss: 0.0443... Generator Loss: 6.1479
    


![png](output_23_406.png)


    Epoch 2/2... Discriminator Loss: 0.0378... Generator Loss: 4.6473
    


![png](output_23_408.png)


    Epoch 2/2... Discriminator Loss: 0.0668... Generator Loss: 3.4329
    


![png](output_23_410.png)


    Epoch 2/2... Discriminator Loss: 0.0394... Generator Loss: 6.9475
    


![png](output_23_412.png)


    Epoch 2/2... Discriminator Loss: 0.1323... Generator Loss: 2.6323
    


![png](output_23_414.png)


    Epoch 2/2... Discriminator Loss: 0.0900... Generator Loss: 6.7955
    


![png](output_23_416.png)


    Epoch 2/2... Discriminator Loss: 0.0991... Generator Loss: 3.0365
    


![png](output_23_418.png)



![png](output_23_419.png)


    Epoch 2/2... Discriminator Loss: 0.0571... Generator Loss: 4.0384
    


![png](output_23_421.png)


    Epoch 2/2... Discriminator Loss: 0.0350... Generator Loss: 4.4516
    


![png](output_23_423.png)


    Epoch 2/2... Discriminator Loss: 0.0548... Generator Loss: 6.8472
    


![png](output_23_425.png)


    Epoch 2/2... Discriminator Loss: 0.0919... Generator Loss: 4.9369
    


![png](output_23_427.png)


    Epoch 2/2... Discriminator Loss: 2.2444... Generator Loss: 0.1674
    


![png](output_23_429.png)


    Epoch 2/2... Discriminator Loss: 1.6615... Generator Loss: 0.3063
    


![png](output_23_431.png)


    Epoch 2/2... Discriminator Loss: 0.3106... Generator Loss: 1.7031
    


![png](output_23_433.png)


    Epoch 2/2... Discriminator Loss: 1.9206... Generator Loss: 0.2391
    


![png](output_23_435.png)


    Epoch 2/2... Discriminator Loss: 1.1986... Generator Loss: 0.4694
    


![png](output_23_437.png)


    Epoch 2/2... Discriminator Loss: 0.2522... Generator Loss: 3.2587
    


![png](output_23_439.png)



![png](output_23_440.png)


    Epoch 2/2... Discriminator Loss: 0.8674... Generator Loss: 1.4216
    


![png](output_23_442.png)


    Epoch 2/2... Discriminator Loss: 1.7080... Generator Loss: 3.9627
    


![png](output_23_444.png)


    Epoch 2/2... Discriminator Loss: 1.6252... Generator Loss: 0.3130
    


![png](output_23_446.png)


    Epoch 2/2... Discriminator Loss: 1.7275... Generator Loss: 0.2618
    


![png](output_23_448.png)


### CelebA
Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.


```python
batch_size = 56
z_dim = 56
learning_rate = 0.002
beta1 = 0.001


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)
```

    Epoch 1/1... Discriminator Loss: 7.1379... Generator Loss: 0.0108
    


![png](output_25_1.png)


    Epoch 1/1... Discriminator Loss: 2.4169... Generator Loss: 10.2369
    


![png](output_25_3.png)


    Epoch 1/1... Discriminator Loss: 1.2249... Generator Loss: 1.1537
    


![png](output_25_5.png)


    Epoch 1/1... Discriminator Loss: 4.7351... Generator Loss: 9.3861
    


![png](output_25_7.png)


    Epoch 1/1... Discriminator Loss: 0.1076... Generator Loss: 4.2833
    


![png](output_25_9.png)


    Epoch 1/1... Discriminator Loss: 0.5722... Generator Loss: 0.9761
    


![png](output_25_11.png)


    Epoch 1/1... Discriminator Loss: 0.0700... Generator Loss: 4.0924
    


![png](output_25_13.png)


    Epoch 1/1... Discriminator Loss: 0.9599... Generator Loss: 0.8709
    


![png](output_25_15.png)


    Epoch 1/1... Discriminator Loss: 0.8792... Generator Loss: 2.5862
    


![png](output_25_17.png)


    Epoch 1/1... Discriminator Loss: 0.3742... Generator Loss: 1.6799
    


![png](output_25_19.png)



![png](output_25_20.png)


    Epoch 1/1... Discriminator Loss: 0.3028... Generator Loss: 1.8224
    


![png](output_25_22.png)


    Epoch 1/1... Discriminator Loss: 1.3584... Generator Loss: 0.5601
    


![png](output_25_24.png)


    Epoch 1/1... Discriminator Loss: 1.4002... Generator Loss: 0.7501
    


![png](output_25_26.png)


    Epoch 1/1... Discriminator Loss: 0.1278... Generator Loss: 3.2307
    


![png](output_25_28.png)


    Epoch 1/1... Discriminator Loss: 0.3087... Generator Loss: 3.8003
    


![png](output_25_30.png)


    Epoch 1/1... Discriminator Loss: 0.4823... Generator Loss: 3.4336
    


![png](output_25_32.png)


    Epoch 1/1... Discriminator Loss: 0.2752... Generator Loss: 3.3997
    


![png](output_25_34.png)


    Epoch 1/1... Discriminator Loss: 1.2219... Generator Loss: 5.0850
    


![png](output_25_36.png)


    Epoch 1/1... Discriminator Loss: 0.5407... Generator Loss: 1.7755
    


![png](output_25_38.png)


    Epoch 1/1... Discriminator Loss: 0.4210... Generator Loss: 2.1057
    


![png](output_25_40.png)



![png](output_25_41.png)


    Epoch 1/1... Discriminator Loss: 0.2535... Generator Loss: 2.2243
    


![png](output_25_43.png)


    Epoch 1/1... Discriminator Loss: 0.5470... Generator Loss: 3.7580
    


![png](output_25_45.png)


    Epoch 1/1... Discriminator Loss: 0.2581... Generator Loss: 4.0780
    


![png](output_25_47.png)


    Epoch 1/1... Discriminator Loss: 3.0883... Generator Loss: 3.9748
    


![png](output_25_49.png)


    Epoch 1/1... Discriminator Loss: 2.0868... Generator Loss: 4.1233
    


![png](output_25_51.png)


    Epoch 1/1... Discriminator Loss: 0.5273... Generator Loss: 1.3650
    


![png](output_25_53.png)


    Epoch 1/1... Discriminator Loss: 0.7212... Generator Loss: 1.2536
    


![png](output_25_55.png)


    Epoch 1/1... Discriminator Loss: 0.5212... Generator Loss: 1.6492
    


![png](output_25_57.png)


    Epoch 1/1... Discriminator Loss: 0.7839... Generator Loss: 1.0615
    


![png](output_25_59.png)


    Epoch 1/1... Discriminator Loss: 2.3296... Generator Loss: 1.4844
    


![png](output_25_61.png)



![png](output_25_62.png)


    Epoch 1/1... Discriminator Loss: 2.2583... Generator Loss: 2.1736
    


![png](output_25_64.png)


    Epoch 1/1... Discriminator Loss: 2.5814... Generator Loss: 2.1216
    


![png](output_25_66.png)


    Epoch 1/1... Discriminator Loss: 1.5433... Generator Loss: 1.4624
    


![png](output_25_68.png)


    Epoch 1/1... Discriminator Loss: 1.4868... Generator Loss: 0.3356
    


![png](output_25_70.png)


    Epoch 1/1... Discriminator Loss: 1.3319... Generator Loss: 1.2925
    


![png](output_25_72.png)


    Epoch 1/1... Discriminator Loss: 2.3025... Generator Loss: 0.1559
    


![png](output_25_74.png)


    Epoch 1/1... Discriminator Loss: 1.6387... Generator Loss: 1.4267
    


![png](output_25_76.png)


    Epoch 1/1... Discriminator Loss: 0.7765... Generator Loss: 1.8763
    


![png](output_25_78.png)


    Epoch 1/1... Discriminator Loss: 1.6221... Generator Loss: 1.0078
    


![png](output_25_80.png)


    Epoch 1/1... Discriminator Loss: 1.2773... Generator Loss: 1.0317
    


![png](output_25_82.png)



![png](output_25_83.png)


    Epoch 1/1... Discriminator Loss: 0.9696... Generator Loss: 1.3942
    


![png](output_25_85.png)


    Epoch 1/1... Discriminator Loss: 1.2051... Generator Loss: 0.5788
    


![png](output_25_87.png)


    Epoch 1/1... Discriminator Loss: 1.5162... Generator Loss: 0.3492
    


![png](output_25_89.png)


    Epoch 1/1... Discriminator Loss: 1.0557... Generator Loss: 1.2959
    


![png](output_25_91.png)


    Epoch 1/1... Discriminator Loss: 1.0046... Generator Loss: 0.6598
    


![png](output_25_93.png)


    Epoch 1/1... Discriminator Loss: 1.0761... Generator Loss: 0.7165
    


![png](output_25_95.png)


    Epoch 1/1... Discriminator Loss: 2.9019... Generator Loss: 1.6352
    


![png](output_25_97.png)


    Epoch 1/1... Discriminator Loss: 1.0642... Generator Loss: 1.4679
    


![png](output_25_99.png)


    Epoch 1/1... Discriminator Loss: 2.0849... Generator Loss: 0.1711
    


![png](output_25_101.png)


    Epoch 1/1... Discriminator Loss: 1.8053... Generator Loss: 0.2642
    


![png](output_25_103.png)



![png](output_25_104.png)


    Epoch 1/1... Discriminator Loss: 2.3625... Generator Loss: 2.7324
    


![png](output_25_106.png)


    Epoch 1/1... Discriminator Loss: 1.9223... Generator Loss: 2.1565
    


![png](output_25_108.png)


    Epoch 1/1... Discriminator Loss: 1.0086... Generator Loss: 1.0449
    


![png](output_25_110.png)


    Epoch 1/1... Discriminator Loss: 1.0786... Generator Loss: 0.6500
    


![png](output_25_112.png)


    Epoch 1/1... Discriminator Loss: 2.1309... Generator Loss: 0.1628
    


![png](output_25_114.png)


    Epoch 1/1... Discriminator Loss: 1.0772... Generator Loss: 0.5990
    


![png](output_25_116.png)


    Epoch 1/1... Discriminator Loss: 1.2657... Generator Loss: 0.4475
    


![png](output_25_118.png)


    Epoch 1/1... Discriminator Loss: 1.3027... Generator Loss: 0.4583
    


![png](output_25_120.png)


    Epoch 1/1... Discriminator Loss: 1.4231... Generator Loss: 0.3986
    


![png](output_25_122.png)


    Epoch 1/1... Discriminator Loss: 1.2910... Generator Loss: 0.7982
    


![png](output_25_124.png)



![png](output_25_125.png)


    Epoch 1/1... Discriminator Loss: 1.2961... Generator Loss: 0.4132
    


![png](output_25_127.png)


    Epoch 1/1... Discriminator Loss: 1.0774... Generator Loss: 0.8293
    


![png](output_25_129.png)


    Epoch 1/1... Discriminator Loss: 0.8072... Generator Loss: 0.8854
    


![png](output_25_131.png)


    Epoch 1/1... Discriminator Loss: 1.5242... Generator Loss: 0.8131
    


![png](output_25_133.png)


    Epoch 1/1... Discriminator Loss: 1.1376... Generator Loss: 0.5727
    


![png](output_25_135.png)


    Epoch 1/1... Discriminator Loss: 0.9629... Generator Loss: 1.1052
    


![png](output_25_137.png)


    Epoch 1/1... Discriminator Loss: 0.9667... Generator Loss: 0.6991
    


![png](output_25_139.png)


    Epoch 1/1... Discriminator Loss: 0.9599... Generator Loss: 0.6061
    


![png](output_25_141.png)


    Epoch 1/1... Discriminator Loss: 0.3247... Generator Loss: 2.0537
    


![png](output_25_143.png)


    Epoch 1/1... Discriminator Loss: 2.7935... Generator Loss: 1.9979
    


![png](output_25_145.png)



![png](output_25_146.png)


    Epoch 1/1... Discriminator Loss: 1.7304... Generator Loss: 0.4520
    


![png](output_25_148.png)


    Epoch 1/1... Discriminator Loss: 1.0943... Generator Loss: 0.6033
    


![png](output_25_150.png)


    Epoch 1/1... Discriminator Loss: 1.0870... Generator Loss: 0.5443
    


![png](output_25_152.png)


    Epoch 1/1... Discriminator Loss: 0.7949... Generator Loss: 0.8497
    


![png](output_25_154.png)


    Epoch 1/1... Discriminator Loss: 2.9865... Generator Loss: 0.0658
    


![png](output_25_156.png)


    Epoch 1/1... Discriminator Loss: 0.5348... Generator Loss: 1.4236
    


![png](output_25_158.png)


    Epoch 1/1... Discriminator Loss: 1.9375... Generator Loss: 1.8243
    


![png](output_25_160.png)


    Epoch 1/1... Discriminator Loss: 1.2442... Generator Loss: 0.9110
    


![png](output_25_162.png)


    Epoch 1/1... Discriminator Loss: 0.3887... Generator Loss: 1.8090
    


![png](output_25_164.png)


    Epoch 1/1... Discriminator Loss: 0.7233... Generator Loss: 3.4422
    


![png](output_25_166.png)



![png](output_25_167.png)


    Epoch 1/1... Discriminator Loss: 1.1795... Generator Loss: 1.0261
    


![png](output_25_169.png)


    Epoch 1/1... Discriminator Loss: 1.5697... Generator Loss: 1.2678
    


![png](output_25_171.png)


    Epoch 1/1... Discriminator Loss: 0.9212... Generator Loss: 0.6364
    


![png](output_25_173.png)


    Epoch 1/1... Discriminator Loss: 1.9358... Generator Loss: 2.9997
    


![png](output_25_175.png)


    Epoch 1/1... Discriminator Loss: 1.2382... Generator Loss: 2.9212
    


![png](output_25_177.png)


    Epoch 1/1... Discriminator Loss: 2.0104... Generator Loss: 0.1801
    


![png](output_25_179.png)


    Epoch 1/1... Discriminator Loss: 1.3144... Generator Loss: 2.4022
    


![png](output_25_181.png)


    Epoch 1/1... Discriminator Loss: 0.9077... Generator Loss: 2.9655
    


![png](output_25_183.png)


    Epoch 1/1... Discriminator Loss: 0.9507... Generator Loss: 2.2347
    


![png](output_25_185.png)


    Epoch 1/1... Discriminator Loss: 1.2073... Generator Loss: 1.7896
    


![png](output_25_187.png)



![png](output_25_188.png)


    Epoch 1/1... Discriminator Loss: 1.5047... Generator Loss: 2.1650
    


![png](output_25_190.png)


    Epoch 1/1... Discriminator Loss: 1.3328... Generator Loss: 0.4540
    


![png](output_25_192.png)


    Epoch 1/1... Discriminator Loss: 0.4751... Generator Loss: 1.4487
    


![png](output_25_194.png)


    Epoch 1/1... Discriminator Loss: 0.5614... Generator Loss: 1.9522
    


![png](output_25_196.png)


    Epoch 1/1... Discriminator Loss: 2.3207... Generator Loss: 0.1984
    


![png](output_25_198.png)


    Epoch 1/1... Discriminator Loss: 0.1968... Generator Loss: 2.1403
    


![png](output_25_200.png)


    Epoch 1/1... Discriminator Loss: 2.1859... Generator Loss: 0.1449
    


![png](output_25_202.png)


    Epoch 1/1... Discriminator Loss: 1.3097... Generator Loss: 3.4742
    


![png](output_25_204.png)


    Epoch 1/1... Discriminator Loss: 0.9123... Generator Loss: 1.6745
    


![png](output_25_206.png)


    Epoch 1/1... Discriminator Loss: 0.3535... Generator Loss: 1.4745
    


![png](output_25_208.png)



![png](output_25_209.png)


    Epoch 1/1... Discriminator Loss: 0.4951... Generator Loss: 1.3906
    


![png](output_25_211.png)


    Epoch 1/1... Discriminator Loss: 1.8253... Generator Loss: 0.2916
    


![png](output_25_213.png)


    Epoch 1/1... Discriminator Loss: 0.4203... Generator Loss: 2.2328
    


![png](output_25_215.png)


    Epoch 1/1... Discriminator Loss: 0.3139... Generator Loss: 4.6071
    


![png](output_25_217.png)


    Epoch 1/1... Discriminator Loss: 0.2502... Generator Loss: 3.4406
    


![png](output_25_219.png)


    Epoch 1/1... Discriminator Loss: 0.3013... Generator Loss: 3.2959
    


![png](output_25_221.png)


    Epoch 1/1... Discriminator Loss: 0.2884... Generator Loss: 1.8764
    


![png](output_25_223.png)


    Epoch 1/1... Discriminator Loss: 0.6182... Generator Loss: 1.0520
    


![png](output_25_225.png)


    Epoch 1/1... Discriminator Loss: 1.8914... Generator Loss: 0.2844
    


![png](output_25_227.png)


    Epoch 1/1... Discriminator Loss: 1.2010... Generator Loss: 0.7993
    


![png](output_25_229.png)



![png](output_25_230.png)


    Epoch 1/1... Discriminator Loss: 0.4296... Generator Loss: 2.1121
    


![png](output_25_232.png)


    Epoch 1/1... Discriminator Loss: 0.7682... Generator Loss: 0.9779
    


![png](output_25_234.png)


    Epoch 1/1... Discriminator Loss: 1.5784... Generator Loss: 0.2965
    


![png](output_25_236.png)


    Epoch 1/1... Discriminator Loss: 2.7297... Generator Loss: 3.0809
    


![png](output_25_238.png)


    Epoch 1/1... Discriminator Loss: 1.0842... Generator Loss: 0.8515
    


![png](output_25_240.png)


    Epoch 1/1... Discriminator Loss: 2.2870... Generator Loss: 0.1284
    


![png](output_25_242.png)


    Epoch 1/1... Discriminator Loss: 0.3620... Generator Loss: 2.1605
    


![png](output_25_244.png)


    Epoch 1/1... Discriminator Loss: 0.2247... Generator Loss: 1.9841
    


![png](output_25_246.png)


    Epoch 1/1... Discriminator Loss: 0.5320... Generator Loss: 3.1864
    


![png](output_25_248.png)


    Epoch 1/1... Discriminator Loss: 1.0197... Generator Loss: 0.5539
    


![png](output_25_250.png)



![png](output_25_251.png)


    Epoch 1/1... Discriminator Loss: 1.0879... Generator Loss: 2.0606
    


![png](output_25_253.png)


    Epoch 1/1... Discriminator Loss: 0.1445... Generator Loss: 6.4704
    


![png](output_25_255.png)


    Epoch 1/1... Discriminator Loss: 0.2373... Generator Loss: 3.1770
    


![png](output_25_257.png)


    Epoch 1/1... Discriminator Loss: 0.2213... Generator Loss: 2.0291
    


![png](output_25_259.png)


    Epoch 1/1... Discriminator Loss: 1.3277... Generator Loss: 1.2328
    


![png](output_25_261.png)


    Epoch 1/1... Discriminator Loss: 0.5691... Generator Loss: 1.9335
    


![png](output_25_263.png)


    Epoch 1/1... Discriminator Loss: 1.2930... Generator Loss: 0.3935
    


![png](output_25_265.png)


    Epoch 1/1... Discriminator Loss: 1.3529... Generator Loss: 2.9937
    


![png](output_25_267.png)


    Epoch 1/1... Discriminator Loss: 0.9759... Generator Loss: 1.6091
    


![png](output_25_269.png)


    Epoch 1/1... Discriminator Loss: 0.3126... Generator Loss: 2.7530
    


![png](output_25_271.png)



![png](output_25_272.png)


    Epoch 1/1... Discriminator Loss: 1.0233... Generator Loss: 1.6549
    


![png](output_25_274.png)


    Epoch 1/1... Discriminator Loss: 0.8166... Generator Loss: 4.6092
    


![png](output_25_276.png)


    Epoch 1/1... Discriminator Loss: 1.0103... Generator Loss: 0.5637
    


![png](output_25_278.png)


    Epoch 1/1... Discriminator Loss: 0.4424... Generator Loss: 1.1608
    


![png](output_25_280.png)


    Epoch 1/1... Discriminator Loss: 0.0616... Generator Loss: 3.7977
    


![png](output_25_282.png)


    Epoch 1/1... Discriminator Loss: 1.8251... Generator Loss: 1.7990
    


![png](output_25_284.png)


    Epoch 1/1... Discriminator Loss: 1.5308... Generator Loss: 0.7302
    


![png](output_25_286.png)


    Epoch 1/1... Discriminator Loss: 2.1396... Generator Loss: 0.1685
    


![png](output_25_288.png)


    Epoch 1/1... Discriminator Loss: 2.3644... Generator Loss: 3.9645
    


![png](output_25_290.png)


    Epoch 1/1... Discriminator Loss: 0.7411... Generator Loss: 2.0112
    


![png](output_25_292.png)



![png](output_25_293.png)


    Epoch 1/1... Discriminator Loss: 1.0928... Generator Loss: 0.5227
    


![png](output_25_295.png)


    Epoch 1/1... Discriminator Loss: 0.8214... Generator Loss: 3.8063
    


![png](output_25_297.png)


    Epoch 1/1... Discriminator Loss: 0.3317... Generator Loss: 3.8892
    


![png](output_25_299.png)


    Epoch 1/1... Discriminator Loss: 2.0364... Generator Loss: 2.4488
    


![png](output_25_301.png)


    Epoch 1/1... Discriminator Loss: 0.7650... Generator Loss: 0.7305
    


![png](output_25_303.png)


    Epoch 1/1... Discriminator Loss: 0.4764... Generator Loss: 1.4482
    


![png](output_25_305.png)


    Epoch 1/1... Discriminator Loss: 0.2613... Generator Loss: 1.9441
    


![png](output_25_307.png)


    Epoch 1/1... Discriminator Loss: 0.2015... Generator Loss: 2.1826
    


![png](output_25_309.png)


    Epoch 1/1... Discriminator Loss: 0.2022... Generator Loss: 2.9977
    


![png](output_25_311.png)


    Epoch 1/1... Discriminator Loss: 0.6337... Generator Loss: 3.3210
    


![png](output_25_313.png)



![png](output_25_314.png)


    Epoch 1/1... Discriminator Loss: 1.5506... Generator Loss: 0.5117
    


![png](output_25_316.png)


    Epoch 1/1... Discriminator Loss: 2.0083... Generator Loss: 0.3445
    


![png](output_25_318.png)


    Epoch 1/1... Discriminator Loss: 1.4085... Generator Loss: 0.4110
    


![png](output_25_320.png)


    Epoch 1/1... Discriminator Loss: 1.8715... Generator Loss: 0.8164
    


![png](output_25_322.png)


    Epoch 1/1... Discriminator Loss: 0.8664... Generator Loss: 1.3870
    


![png](output_25_324.png)


    Epoch 1/1... Discriminator Loss: 0.7588... Generator Loss: 0.8393
    


![png](output_25_326.png)


    Epoch 1/1... Discriminator Loss: 1.1877... Generator Loss: 0.6148
    


![png](output_25_328.png)


    Epoch 1/1... Discriminator Loss: 1.8084... Generator Loss: 1.0370
    


![png](output_25_330.png)


    Epoch 1/1... Discriminator Loss: 1.0399... Generator Loss: 3.1324
    


![png](output_25_332.png)


    Epoch 1/1... Discriminator Loss: 0.3424... Generator Loss: 1.5801
    


![png](output_25_334.png)



![png](output_25_335.png)


    Epoch 1/1... Discriminator Loss: 0.3365... Generator Loss: 4.0526
    


![png](output_25_337.png)


    Epoch 1/1... Discriminator Loss: 0.1147... Generator Loss: 3.5987
    


![png](output_25_339.png)


    Epoch 1/1... Discriminator Loss: 0.1594... Generator Loss: 2.5169
    


![png](output_25_341.png)


    Epoch 1/1... Discriminator Loss: 0.5187... Generator Loss: 1.0484
    


![png](output_25_343.png)


    Epoch 1/1... Discriminator Loss: 0.4126... Generator Loss: 2.4555
    


![png](output_25_345.png)


    Epoch 1/1... Discriminator Loss: 2.1064... Generator Loss: 2.3689
    


![png](output_25_347.png)


    Epoch 1/1... Discriminator Loss: 0.9434... Generator Loss: 3.3191
    


![png](output_25_349.png)


    Epoch 1/1... Discriminator Loss: 0.0854... Generator Loss: 4.2301
    


![png](output_25_351.png)


    Epoch 1/1... Discriminator Loss: 0.2962... Generator Loss: 1.5566
    


![png](output_25_353.png)


    Epoch 1/1... Discriminator Loss: 1.2833... Generator Loss: 1.8426
    


![png](output_25_355.png)



![png](output_25_356.png)


    Epoch 1/1... Discriminator Loss: 0.3840... Generator Loss: 1.3606
    


![png](output_25_358.png)


    Epoch 1/1... Discriminator Loss: 2.5387... Generator Loss: 3.3730
    


![png](output_25_360.png)


    Epoch 1/1... Discriminator Loss: 0.2345... Generator Loss: 4.0744
    


![png](output_25_362.png)


    Epoch 1/1... Discriminator Loss: 0.2637... Generator Loss: 2.9353
    


![png](output_25_364.png)


    Epoch 1/1... Discriminator Loss: 0.3035... Generator Loss: 1.7016
    


![png](output_25_366.png)


    Epoch 1/1... Discriminator Loss: 2.6198... Generator Loss: 2.4019
    


![png](output_25_368.png)


    Epoch 1/1... Discriminator Loss: 1.5693... Generator Loss: 2.8267
    


![png](output_25_370.png)


    Epoch 1/1... Discriminator Loss: 0.0987... Generator Loss: 3.5690
    


![png](output_25_372.png)


    Epoch 1/1... Discriminator Loss: 1.7236... Generator Loss: 1.2767
    


![png](output_25_374.png)


    Epoch 1/1... Discriminator Loss: 1.6411... Generator Loss: 0.5340
    


![png](output_25_376.png)



![png](output_25_377.png)


    Epoch 1/1... Discriminator Loss: 1.3696... Generator Loss: 0.4660
    


![png](output_25_379.png)


    Epoch 1/1... Discriminator Loss: 1.2387... Generator Loss: 0.8844
    


![png](output_25_381.png)


    Epoch 1/1... Discriminator Loss: 1.3374... Generator Loss: 0.4277
    


![png](output_25_383.png)


    Epoch 1/1... Discriminator Loss: 1.3725... Generator Loss: 0.4297
    


![png](output_25_385.png)


    Epoch 1/1... Discriminator Loss: 0.5582... Generator Loss: 1.9280
    


![png](output_25_387.png)


    Epoch 1/1... Discriminator Loss: 1.8623... Generator Loss: 0.2164
    


![png](output_25_389.png)


    Epoch 1/1... Discriminator Loss: 0.2961... Generator Loss: 1.9986
    


![png](output_25_391.png)


    Epoch 1/1... Discriminator Loss: 0.4121... Generator Loss: 1.8829
    


![png](output_25_393.png)


    Epoch 1/1... Discriminator Loss: 0.1236... Generator Loss: 3.6106
    


![png](output_25_395.png)


    Epoch 1/1... Discriminator Loss: 3.0496... Generator Loss: 0.0663
    


![png](output_25_397.png)



![png](output_25_398.png)


    Epoch 1/1... Discriminator Loss: 0.8198... Generator Loss: 0.6783
    


![png](output_25_400.png)


    Epoch 1/1... Discriminator Loss: 2.1408... Generator Loss: 1.7254
    


![png](output_25_402.png)


    Epoch 1/1... Discriminator Loss: 0.3017... Generator Loss: 2.0647
    


![png](output_25_404.png)


    Epoch 1/1... Discriminator Loss: 0.8832... Generator Loss: 2.2248
    


![png](output_25_406.png)


    Epoch 1/1... Discriminator Loss: 0.0826... Generator Loss: 3.8126
    


![png](output_25_408.png)


    Epoch 1/1... Discriminator Loss: 0.0667... Generator Loss: 5.1363
    


![png](output_25_410.png)


    Epoch 1/1... Discriminator Loss: 0.5582... Generator Loss: 1.3784
    


![png](output_25_412.png)


    Epoch 1/1... Discriminator Loss: 0.1924... Generator Loss: 2.0945
    


![png](output_25_414.png)


    Epoch 1/1... Discriminator Loss: 1.2574... Generator Loss: 1.1185
    


![png](output_25_416.png)


    Epoch 1/1... Discriminator Loss: 0.7226... Generator Loss: 1.2023
    


![png](output_25_418.png)



![png](output_25_419.png)


    Epoch 1/1... Discriminator Loss: 1.5064... Generator Loss: 0.3185
    


![png](output_25_421.png)


    Epoch 1/1... Discriminator Loss: 0.9186... Generator Loss: 4.2014
    


![png](output_25_423.png)


    Epoch 1/1... Discriminator Loss: 0.0636... Generator Loss: 3.9982
    


![png](output_25_425.png)


    Epoch 1/1... Discriminator Loss: 0.3305... Generator Loss: 1.9118
    


![png](output_25_427.png)


    Epoch 1/1... Discriminator Loss: 0.4590... Generator Loss: 3.8796
    


![png](output_25_429.png)


    Epoch 1/1... Discriminator Loss: 1.2845... Generator Loss: 0.5034
    


![png](output_25_431.png)


    Epoch 1/1... Discriminator Loss: 1.1494... Generator Loss: 2.4219
    


![png](output_25_433.png)


    Epoch 1/1... Discriminator Loss: 1.0432... Generator Loss: 2.4228
    


![png](output_25_435.png)


    Epoch 1/1... Discriminator Loss: 0.1039... Generator Loss: 2.8636
    


![png](output_25_437.png)


    Epoch 1/1... Discriminator Loss: 0.2302... Generator Loss: 2.2075
    


![png](output_25_439.png)



![png](output_25_440.png)


    Epoch 1/1... Discriminator Loss: 0.4740... Generator Loss: 1.4436
    


![png](output_25_442.png)


    Epoch 1/1... Discriminator Loss: 0.8207... Generator Loss: 0.7063
    


![png](output_25_444.png)


    Epoch 1/1... Discriminator Loss: 0.9972... Generator Loss: 2.1799
    


![png](output_25_446.png)


    Epoch 1/1... Discriminator Loss: 0.3849... Generator Loss: 1.6467
    


![png](output_25_448.png)


    Epoch 1/1... Discriminator Loss: 0.1348... Generator Loss: 3.1726
    


![png](output_25_450.png)


    Epoch 1/1... Discriminator Loss: 0.6666... Generator Loss: 0.8250
    


![png](output_25_452.png)


    Epoch 1/1... Discriminator Loss: 1.9017... Generator Loss: 0.2811
    


![png](output_25_454.png)


    Epoch 1/1... Discriminator Loss: 1.4846... Generator Loss: 0.3125
    


![png](output_25_456.png)


    Epoch 1/1... Discriminator Loss: 0.8613... Generator Loss: 0.7753
    


![png](output_25_458.png)


    Epoch 1/1... Discriminator Loss: 1.8619... Generator Loss: 0.2143
    


![png](output_25_460.png)



![png](output_25_461.png)


    Epoch 1/1... Discriminator Loss: 0.6581... Generator Loss: 4.6361
    


![png](output_25_463.png)


    Epoch 1/1... Discriminator Loss: 0.3339... Generator Loss: 4.5264
    


![png](output_25_465.png)


    Epoch 1/1... Discriminator Loss: 1.0137... Generator Loss: 1.2228
    


![png](output_25_467.png)


    Epoch 1/1... Discriminator Loss: 1.1831... Generator Loss: 2.2831
    


![png](output_25_469.png)


    Epoch 1/1... Discriminator Loss: 2.7218... Generator Loss: 2.9119
    


![png](output_25_471.png)


    Epoch 1/1... Discriminator Loss: 1.8731... Generator Loss: 0.3584
    


![png](output_25_473.png)


    Epoch 1/1... Discriminator Loss: 1.3080... Generator Loss: 0.6605
    


![png](output_25_475.png)


    Epoch 1/1... Discriminator Loss: 1.2068... Generator Loss: 0.6301
    


![png](output_25_477.png)


    Epoch 1/1... Discriminator Loss: 1.0634... Generator Loss: 0.8868
    


![png](output_25_479.png)


    Epoch 1/1... Discriminator Loss: 0.6784... Generator Loss: 0.8678
    


![png](output_25_481.png)



![png](output_25_482.png)


    Epoch 1/1... Discriminator Loss: 0.6973... Generator Loss: 1.0188
    


![png](output_25_484.png)


    Epoch 1/1... Discriminator Loss: 0.6125... Generator Loss: 1.0303
    


![png](output_25_486.png)


    Epoch 1/1... Discriminator Loss: 0.1466... Generator Loss: 3.6779
    


![png](output_25_488.png)


    Epoch 1/1... Discriminator Loss: 0.1053... Generator Loss: 5.6308
    


![png](output_25_490.png)


    Epoch 1/1... Discriminator Loss: 0.2136... Generator Loss: 3.6596
    


![png](output_25_492.png)


    Epoch 1/1... Discriminator Loss: 0.1957... Generator Loss: 3.4238
    


![png](output_25_494.png)


    Epoch 1/1... Discriminator Loss: 0.3128... Generator Loss: 1.5787
    


![png](output_25_496.png)


    Epoch 1/1... Discriminator Loss: 1.3339... Generator Loss: 0.8129
    


![png](output_25_498.png)


    Epoch 1/1... Discriminator Loss: 1.1869... Generator Loss: 1.2406
    


![png](output_25_500.png)


    Epoch 1/1... Discriminator Loss: 1.1215... Generator Loss: 0.8618
    


![png](output_25_502.png)



![png](output_25_503.png)


    Epoch 1/1... Discriminator Loss: 0.4761... Generator Loss: 1.2665
    


![png](output_25_505.png)


    Epoch 1/1... Discriminator Loss: 1.3790... Generator Loss: 1.3245
    


![png](output_25_507.png)


    Epoch 1/1... Discriminator Loss: 1.2642... Generator Loss: 0.6917
    


![png](output_25_509.png)


    Epoch 1/1... Discriminator Loss: 0.5361... Generator Loss: 1.0161
    


![png](output_25_511.png)


    Epoch 1/1... Discriminator Loss: 2.3336... Generator Loss: 2.4896
    


![png](output_25_513.png)


    Epoch 1/1... Discriminator Loss: 0.9111... Generator Loss: 0.8567
    


![png](output_25_515.png)


    Epoch 1/1... Discriminator Loss: 0.4217... Generator Loss: 1.2507
    


![png](output_25_517.png)


    Epoch 1/1... Discriminator Loss: 0.3826... Generator Loss: 1.8536
    


![png](output_25_519.png)


    Epoch 1/1... Discriminator Loss: 0.7363... Generator Loss: 0.8081
    


![png](output_25_521.png)


    Epoch 1/1... Discriminator Loss: 0.1809... Generator Loss: 2.3217
    


![png](output_25_523.png)



![png](output_25_524.png)


    Epoch 1/1... Discriminator Loss: 0.2734... Generator Loss: 1.6521
    


![png](output_25_526.png)


    Epoch 1/1... Discriminator Loss: 1.2549... Generator Loss: 0.6400
    


![png](output_25_528.png)


    Epoch 1/1... Discriminator Loss: 0.2446... Generator Loss: 2.4774
    


![png](output_25_530.png)


    Epoch 1/1... Discriminator Loss: 0.1915... Generator Loss: 1.9666
    


![png](output_25_532.png)


    Epoch 1/1... Discriminator Loss: 0.1033... Generator Loss: 3.6110
    


![png](output_25_534.png)


    Epoch 1/1... Discriminator Loss: 0.7864... Generator Loss: 2.3353
    


![png](output_25_536.png)


    Epoch 1/1... Discriminator Loss: 4.4435... Generator Loss: 3.5275
    


![png](output_25_538.png)


    Epoch 1/1... Discriminator Loss: 1.8466... Generator Loss: 2.6200
    


![png](output_25_540.png)


    Epoch 1/1... Discriminator Loss: 0.6368... Generator Loss: 1.3827
    


![png](output_25_542.png)


    Epoch 1/1... Discriminator Loss: 1.8507... Generator Loss: 1.3334
    


![png](output_25_544.png)



![png](output_25_545.png)


    Epoch 1/1... Discriminator Loss: 0.2465... Generator Loss: 1.8912
    


![png](output_25_547.png)


    Epoch 1/1... Discriminator Loss: 0.2316... Generator Loss: 2.4102
    


![png](output_25_549.png)


    Epoch 1/1... Discriminator Loss: 0.0826... Generator Loss: 3.9098
    


![png](output_25_551.png)


    Epoch 1/1... Discriminator Loss: 0.1166... Generator Loss: 2.5776
    


![png](output_25_553.png)


    Epoch 1/1... Discriminator Loss: 0.2011... Generator Loss: 2.0017
    


![png](output_25_555.png)


    Epoch 1/1... Discriminator Loss: 0.5307... Generator Loss: 1.7063
    


![png](output_25_557.png)


    Epoch 1/1... Discriminator Loss: 2.4755... Generator Loss: 2.8101
    


![png](output_25_559.png)


    Epoch 1/1... Discriminator Loss: 1.1274... Generator Loss: 1.0498
    


![png](output_25_561.png)


    Epoch 1/1... Discriminator Loss: 1.4503... Generator Loss: 1.2533
    


![png](output_25_563.png)


    Epoch 1/1... Discriminator Loss: 0.7430... Generator Loss: 1.0063
    


![png](output_25_565.png)



![png](output_25_566.png)


    Epoch 1/1... Discriminator Loss: 0.1480... Generator Loss: 2.8074
    


![png](output_25_568.png)


    Epoch 1/1... Discriminator Loss: 0.5854... Generator Loss: 3.0970
    


![png](output_25_570.png)


    Epoch 1/1... Discriminator Loss: 1.1118... Generator Loss: 0.5275
    


![png](output_25_572.png)


    Epoch 1/1... Discriminator Loss: 0.2612... Generator Loss: 2.5360
    


![png](output_25_574.png)


    Epoch 1/1... Discriminator Loss: 1.2954... Generator Loss: 0.3634
    


![png](output_25_576.png)


    Epoch 1/1... Discriminator Loss: 0.4099... Generator Loss: 3.3986
    


![png](output_25_578.png)


    Epoch 1/1... Discriminator Loss: 0.0535... Generator Loss: 4.3577
    


![png](output_25_580.png)


    Epoch 1/1... Discriminator Loss: 2.2107... Generator Loss: 3.0941
    


![png](output_25_582.png)


    Epoch 1/1... Discriminator Loss: 0.1695... Generator Loss: 3.8233
    


![png](output_25_584.png)


    Epoch 1/1... Discriminator Loss: 2.0176... Generator Loss: 1.2548
    


![png](output_25_586.png)



![png](output_25_587.png)


    Epoch 1/1... Discriminator Loss: 0.7995... Generator Loss: 1.4008
    


![png](output_25_589.png)


    Epoch 1/1... Discriminator Loss: 0.5801... Generator Loss: 0.9861
    


![png](output_25_591.png)


    Epoch 1/1... Discriminator Loss: 0.1423... Generator Loss: 2.9249
    


![png](output_25_593.png)


    Epoch 1/1... Discriminator Loss: 1.3413... Generator Loss: 0.8722
    


![png](output_25_595.png)


    Epoch 1/1... Discriminator Loss: 0.4772... Generator Loss: 2.3239
    


![png](output_25_597.png)


    Epoch 1/1... Discriminator Loss: 0.0862... Generator Loss: 6.0137
    


![png](output_25_599.png)


    Epoch 1/1... Discriminator Loss: 0.0921... Generator Loss: 2.9847
    


![png](output_25_601.png)


    Epoch 1/1... Discriminator Loss: 1.4861... Generator Loss: 0.2898
    


![png](output_25_603.png)


    Epoch 1/1... Discriminator Loss: 0.8478... Generator Loss: 1.8307
    


![png](output_25_605.png)


    Epoch 1/1... Discriminator Loss: 0.1232... Generator Loss: 5.2018
    


![png](output_25_607.png)



![png](output_25_608.png)


    Epoch 1/1... Discriminator Loss: 0.1531... Generator Loss: 2.2358
    


![png](output_25_610.png)


    Epoch 1/1... Discriminator Loss: 0.0217... Generator Loss: 6.2485
    


![png](output_25_612.png)


    Epoch 1/1... Discriminator Loss: 1.5895... Generator Loss: 0.3089
    


![png](output_25_614.png)


    Epoch 1/1... Discriminator Loss: 1.1096... Generator Loss: 0.6113
    


![png](output_25_616.png)


    Epoch 1/1... Discriminator Loss: 0.1514... Generator Loss: 2.7978
    


![png](output_25_618.png)


    Epoch 1/1... Discriminator Loss: 0.0990... Generator Loss: 4.4769
    


![png](output_25_620.png)


    Epoch 1/1... Discriminator Loss: 0.6406... Generator Loss: 1.0181
    


![png](output_25_622.png)


    Epoch 1/1... Discriminator Loss: 1.2597... Generator Loss: 0.6598
    


![png](output_25_624.png)


    Epoch 1/1... Discriminator Loss: 0.0614... Generator Loss: 4.9766
    


![png](output_25_626.png)


    Epoch 1/1... Discriminator Loss: 0.2084... Generator Loss: 4.8504
    


![png](output_25_628.png)



![png](output_25_629.png)


    Epoch 1/1... Discriminator Loss: 1.2113... Generator Loss: 1.7877
    


![png](output_25_631.png)


    Epoch 1/1... Discriminator Loss: 0.9653... Generator Loss: 1.5737
    


![png](output_25_633.png)


    Epoch 1/1... Discriminator Loss: 0.1368... Generator Loss: 2.9633
    


![png](output_25_635.png)


    Epoch 1/1... Discriminator Loss: 2.3101... Generator Loss: 2.5429
    


![png](output_25_637.png)


    Epoch 1/1... Discriminator Loss: 1.8297... Generator Loss: 0.2119
    


![png](output_25_639.png)


    Epoch 1/1... Discriminator Loss: 0.4219... Generator Loss: 1.5407
    


![png](output_25_641.png)


    Epoch 1/1... Discriminator Loss: 0.0777... Generator Loss: 3.4062
    


![png](output_25_643.png)


    Epoch 1/1... Discriminator Loss: 0.6068... Generator Loss: 1.2621
    


![png](output_25_645.png)


    Epoch 1/1... Discriminator Loss: 1.1982... Generator Loss: 0.6885
    


![png](output_25_647.png)


    Epoch 1/1... Discriminator Loss: 0.6823... Generator Loss: 0.8117
    


![png](output_25_649.png)



![png](output_25_650.png)


    Epoch 1/1... Discriminator Loss: 1.0142... Generator Loss: 1.7787
    


![png](output_25_652.png)


    Epoch 1/1... Discriminator Loss: 1.4415... Generator Loss: 3.2022
    


![png](output_25_654.png)


    Epoch 1/1... Discriminator Loss: 2.3050... Generator Loss: 3.8712
    


![png](output_25_656.png)


    Epoch 1/1... Discriminator Loss: 0.1150... Generator Loss: 5.1105
    


![png](output_25_658.png)


    Epoch 1/1... Discriminator Loss: 0.1435... Generator Loss: 3.7605
    


![png](output_25_660.png)


    Epoch 1/1... Discriminator Loss: 0.5218... Generator Loss: 1.0098
    


![png](output_25_662.png)


    Epoch 1/1... Discriminator Loss: 0.7001... Generator Loss: 3.9457
    


![png](output_25_664.png)


    Epoch 1/1... Discriminator Loss: 1.6973... Generator Loss: 2.1224
    


![png](output_25_666.png)


    Epoch 1/1... Discriminator Loss: 1.3013... Generator Loss: 0.6476
    


![png](output_25_668.png)


    Epoch 1/1... Discriminator Loss: 0.5196... Generator Loss: 2.9741
    


![png](output_25_670.png)



![png](output_25_671.png)


    Epoch 1/1... Discriminator Loss: 0.3338... Generator Loss: 1.5880
    


![png](output_25_673.png)


    Epoch 1/1... Discriminator Loss: 0.4349... Generator Loss: 2.8476
    


![png](output_25_675.png)


    Epoch 1/1... Discriminator Loss: 1.0661... Generator Loss: 2.1367
    


![png](output_25_677.png)


    Epoch 1/1... Discriminator Loss: 0.1588... Generator Loss: 4.0736
    


![png](output_25_679.png)


    Epoch 1/1... Discriminator Loss: 0.3501... Generator Loss: 1.5946
    


![png](output_25_681.png)


    Epoch 1/1... Discriminator Loss: 0.9242... Generator Loss: 2.9098
    


![png](output_25_683.png)


    Epoch 1/1... Discriminator Loss: 1.5035... Generator Loss: 1.0274
    


![png](output_25_685.png)


    Epoch 1/1... Discriminator Loss: 1.3270... Generator Loss: 0.5291
    


![png](output_25_687.png)


    Epoch 1/1... Discriminator Loss: 1.2923... Generator Loss: 0.4890
    


![png](output_25_689.png)


    Epoch 1/1... Discriminator Loss: 0.6284... Generator Loss: 1.0808
    


![png](output_25_691.png)



![png](output_25_692.png)


    Epoch 1/1... Discriminator Loss: 1.4599... Generator Loss: 0.3389
    


![png](output_25_694.png)


    Epoch 1/1... Discriminator Loss: 0.1841... Generator Loss: 2.2469
    


![png](output_25_696.png)


    Epoch 1/1... Discriminator Loss: 0.1252... Generator Loss: 2.5657
    


![png](output_25_698.png)


    Epoch 1/1... Discriminator Loss: 0.1430... Generator Loss: 2.5109
    


![png](output_25_700.png)


    Epoch 1/1... Discriminator Loss: 0.0394... Generator Loss: 4.4241
    


![png](output_25_702.png)


    Epoch 1/1... Discriminator Loss: 0.8477... Generator Loss: 1.6568
    


![png](output_25_704.png)


    Epoch 1/1... Discriminator Loss: 1.9748... Generator Loss: 0.2184
    


![png](output_25_706.png)


    Epoch 1/1... Discriminator Loss: 1.2224... Generator Loss: 0.5928
    


![png](output_25_708.png)


    Epoch 1/1... Discriminator Loss: 1.4681... Generator Loss: 2.4623
    


![png](output_25_710.png)


    Epoch 1/1... Discriminator Loss: 1.3213... Generator Loss: 0.6633
    


![png](output_25_712.png)



![png](output_25_713.png)


    Epoch 1/1... Discriminator Loss: 0.9784... Generator Loss: 1.1493
    


![png](output_25_715.png)


    Epoch 1/1... Discriminator Loss: 0.1326... Generator Loss: 2.7837
    


![png](output_25_717.png)


    Epoch 1/1... Discriminator Loss: 0.1324... Generator Loss: 2.5960
    


![png](output_25_719.png)


    Epoch 1/1... Discriminator Loss: 0.0675... Generator Loss: 3.2235
    


![png](output_25_721.png)


    Epoch 1/1... Discriminator Loss: 0.0461... Generator Loss: 4.2253
    


![png](output_25_723.png)


    Epoch 1/1... Discriminator Loss: 0.1453... Generator Loss: 2.2081
    


![png](output_25_725.png)


    Epoch 1/1... Discriminator Loss: 0.5532... Generator Loss: 5.3244
    


![png](output_25_727.png)


    Epoch 1/1... Discriminator Loss: 1.6842... Generator Loss: 0.5445
    


![png](output_25_729.png)


    Epoch 1/1... Discriminator Loss: 1.5736... Generator Loss: 0.6038
    


![png](output_25_731.png)


    Epoch 1/1... Discriminator Loss: 0.7954... Generator Loss: 1.6383
    


![png](output_25_733.png)



![png](output_25_734.png)


    Epoch 1/1... Discriminator Loss: 0.0815... Generator Loss: 3.2835
    


![png](output_25_736.png)


    Epoch 1/1... Discriminator Loss: 1.1101... Generator Loss: 1.0234
    


![png](output_25_738.png)


    Epoch 1/1... Discriminator Loss: 1.2962... Generator Loss: 0.4734
    


![png](output_25_740.png)


    Epoch 1/1... Discriminator Loss: 0.0598... Generator Loss: 3.6971
    


![png](output_25_742.png)


    Epoch 1/1... Discriminator Loss: 0.0772... Generator Loss: 6.4004
    


![png](output_25_744.png)


    Epoch 1/1... Discriminator Loss: 1.0409... Generator Loss: 0.6085
    


![png](output_25_746.png)


    Epoch 1/1... Discriminator Loss: 0.5111... Generator Loss: 3.4475
    


![png](output_25_748.png)


    Epoch 1/1... Discriminator Loss: 0.4655... Generator Loss: 2.3241
    


![png](output_25_750.png)


    Epoch 1/1... Discriminator Loss: 0.3409... Generator Loss: 1.4974
    


![png](output_25_752.png)


    Epoch 1/1... Discriminator Loss: 1.2692... Generator Loss: 0.5713
    


![png](output_25_754.png)



![png](output_25_755.png)


    Epoch 1/1... Discriminator Loss: 1.7252... Generator Loss: 0.4239
    


![png](output_25_757.png)


### Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
