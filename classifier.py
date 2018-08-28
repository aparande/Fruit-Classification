import tensorflow as tf
import dataset
import random
import sys
from tensorflow.python import debug as tf_debug

NUM_CLASSES = len(dataset.CLASSES)
IMAGE_SIZE = 100

def placeholder_inputs(batch_size):
    #Generates placeholders for the inputs
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), name="images")
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size,), name="labels")
    return images_placeholder, labels_placeholder

def get_weights(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name="Weights")

def get_biases(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name="Biases")

def conv_layer(input, filter_size, name, strides = [1, 1, 1, 1], pool_size=2, padding="SAME"):
    with tf.name_scope(name):
        weights = get_weights(filter_size)
        biases = get_biases([filter_size[3]])

        #Convolve the image
        conv = tf.nn.sigmoid(tf.nn.conv2d(input, weights, strides, padding))
        #Pools the convoluted layer. Arguments are input, pool_size, strides, padding
        pool = tf.nn.max_pool(conv, [1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], padding)

        return pool

def dense_layer(input, input_size, output_size, name):
    with tf.name_scope(name):
        weights = get_weights([input_size, output_size])
        biases = get_biases([output_size])

        h = tf.matmul(input, weights) + biases
        return tf.nn.sigmoid(h)

def define_model(images):
    #Output shape = (batch_size, 50, 50, 10)
    conv1 = conv_layer(images, [20, 20, 3, 10], "Conv1")
    #Output shape = (batch_size, 25, 25, 20)
    conv2 = conv_layer(conv1, [5, 5, 10, 20], "Conv2")
    conv2 = tf.reshape(conv2, [-1, 25 * 25 * 20])

    dense1 = dense_layer(conv2, 25 * 25 * 20, 100, "Dense1")
    logits = dense_layer(dense1, 100, NUM_CLASSES, "Softmax_Linear")

    return logits

def define_loss(logits, labels):
    labels = tf.to_int64(labels)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
    return loss

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_evaluation(session, eval_correct, testDataset, batch_size, images_placeholder, labels_placeholder):
    training_data, training_labels = testDataset
    batches = [(training_data[i:i+batch_size], training_labels[i:i+batch_size]) for i in range(0, len(training_data), batch_size)]
    totalCorrect = 0
    for batch in batches:
        image_data = batch[0]
        label_data = batch[1]

        feed_dict = {images_placeholder: image_data, labels_placeholder: label_data}
        totalCorrect += session.run(eval_correct, feed_dict=feed_dict)

    precision = float(totalCorrect) / len(training_data)
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (len(training_data), totalCorrect, precision))
    sys.stdout.flush()

def run_training(batch_size, learning_rate, epochs, run_number):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(batch_size)

        logits = define_model(images_placeholder)
        lossFunction = define_loss(logits, labels_placeholder)
        train_op = training(lossFunction, learning_rate)

        eval_correct = evaluation(logits, labels_placeholder)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            #session = tf_debug.LocalCLIDebugWrapperSession(session)

            summary_writer = tf.summary.FileWriter("log/"+str(run_number), session.graph)
            session.run(init)

            for step in range(epochs):
                training_data, training_labels = dataset.get_training_data()
                batches = [(training_data[i:i+batch_size], training_labels[i:i+batch_size]) for i in range(0, len(training_data), batch_size)]
                for batch in batches:
                    image_data = batch[0]
                    label_data = batch[1]

                    feed_dict = {images_placeholder: image_data, labels_placeholder: label_data}
                    activations, loss_value = session.run([train_op, lossFunction], feed_dict=feed_dict)
                if step % 2 == 0:
                    print('Step %d: loss = %.2f' % (step, loss_value))
                    sys.stdout.flush()
                    summary_str = session.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()


                if (step + 1) % 5 == 0 or (step + 1) == epochs:
                    validation_data = dataset.get_validation_data(batch_size)
                    print("Doing evaluation on validation Set")
                    sys.stdout.flush()
                    do_evaluation(session, eval_correct, validation_data, batch_size, images_placeholder, labels_placeholder)

run_training(59, 0.05, 20, 0)
