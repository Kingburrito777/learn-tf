# this script is highly commented to help begginers understand what's happening in tensorflow
# during a basic machine learning application.

# loading a "MNIST_data" set and starting a tensorflow session
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# starting tensorflow interactive session
# first create graph, then launch in a session

#importing tensorflo
import tensorflow as tf
#starting a graoh session ising the "InteractiveSession" class
sess = tf.InteractiveSession()

# building a softmax regression model
# first we start with placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# Here x and y_ aren't specific values.
# Rather, they are each a placeholder - a value that we'll input when we ask TensorFlow to run a computation.
# x is a 2d tensor, its a shape of [None, 784]. 784 is the
# dimensionality of a single flattened 28 by 28 pixel MNIST image.
# 'None' indicates that the first dimension, corresponding to the batch size, can be of any size.
# 'y_' will also consist of a 2d tensor, where each row is a "one-hot 10-dimensional vector"
# indicating which digit class (0-9) the MNIST image belongs to.
# note: the shape is optional, but it catches bugs from inconsistent data types

# Variables
# using weights - W - and biases - b -
# these values reside in tensorflow's graph
# it can then be used by the computation
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# we call - W - and - b - to tf.Variable.
# We then make them tensors full of zeroes (tf.zeroes)
# In this case: - W - is a 784 by 10 matrix
# and b is a 10-dimensional vector. (we have 10 classes)

# We need to initialize the variables in the session before we use them
# we have already specified the variables (tensors full of zeroes)
# but we have to initialize them and we can do it for all variables at once:
sess.run(tf.global_variables_initializer())

# Now we should impliment the regression
# using - x - as the vectorized input we created before
# we multiply that by the weight matrix - W -
# then we add it to the bias - b -
y = tf.matmul(x,W) + b

# now using cross entropy we predict how bad the models prediction was on a specific example.
# every time we train our model we try to minimize this as much as possible
# the loss function is the cross-entropy between the target and the softmax activation function
# applied to the model's prediction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# a note from tensorflow: "tf.nn.softmax_cross_entropy_with_logits internally applies
# the softmax on the model's unnormalized model prediction and sums across all classes,
# and tf.reduce_mean takes the average over these sums."

# Training
# tensorflow now knows what our data looks like and we can now use tensorflow classes
# to automatically differentiate and find gradients of the loss with respect to each of the variables.
# Using gradient descent we descend the cross-entropy (step of 0.5, this can be anything but note that if
# too large, you could overstep. If too small, you'll have to wait longer)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# train_step, when run, will apply the gradient descent updates to the parameters.
# Training the model can then be accomplished by repeating train_step.
for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
# this runs the iteration 1000 times,
# with 100 training examples in each batch
# THEN: train_step is run, and the temporary placeholders we made earlier are replaced with the MNIST data!
# We use feed_dict to replace the data

# now we want to evaluate the model. How well did it do?
# We use tf.argmax to find the index of the highest entry in a tensor along some axis.
# ex): tf.argmax(y,1) is the label our model thinks is most likely for each input,
# while tf.argmax(y_,1) is the true label.
# tf.equal compares them
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# what this does is give us a list of booleans, but we need to make them values we can average
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# now we can print the accuracy on the testing data
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# go to the next file to see a more accurate representation
