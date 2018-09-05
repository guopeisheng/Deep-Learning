import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)#call mnist function
learningRate = 0.001
trainingIters = 128000
batchSize = 128
displayStep = 10
nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 400 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
 'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}
biases = {
 'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels
    # cell = tf.contrib.rnn.BasicRNNCell(nHidden)#find which lstm to use in the documentation
    # cell = tf.contrib.rnn.BasicLSTMCell(nHidden)
    cell = tf.contrib.rnn.GRUCell(nHidden)
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)#for the rnn where to get the output and hidden state
    return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)
prediction = tf.nn.softmax(pred)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

#dispaly on tensorboard
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', cost)
summary_op = tf.summary.merge_all()
result_dir = './RNNresult'

# Instantiate a SummaryWriter to output summaries and the Graph.
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
init = tf.initialize_all_variables()
# Create a saver for writing training checkpoints
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batchSize <= trainingIters:
     batchX, batchY = mnist.train.next_batch(batchSize)#mnist has a way to get the next batch
     batchX = batchX.reshape((batchSize, nSteps, nInput))
     sess.run(optimizer, feed_dict={x:batchX, y:batchY})
     if step % displayStep == 0:
      acc = sess.run(accuracy,feed_dict={x: batchX, y: batchY})
      loss = sess.run(cost,feed_dict={x: batchX, y: batchY})
      print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
       "{:.6f}".format(loss) + ", Training Accuracy= " + \
       "{:.5f}".format(acc))

      summary_str = sess.run(summary_op, feed_dict={x: batchX, y: batchY})
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

     if step % 1280 == 0 or step == trainingIters:
      checkpoint_file = os.path.join(result_dir, 'checkpoint')
      saver.save(sess, checkpoint_file, global_step=step)
     step +=1
    print('Optimization finished')

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))