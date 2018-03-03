import tensorflow as tf
from sklearn.utils import shuffle
import math
import numpy as np
from Tools import find_labels
from Tools import decode_one_hot
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def create_model(x, dim, hiddens,  weights, biases, keep_prob):
    layer = x
    for i in range(len(dim) - 1):
        layer = tf.nn.xw_plus_b(layer, weights[i], biases[i])
        if i < len(hiddens):
            layer = tf.nn.elu(layer)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)
        else:
            layer = tf.nn.sigmoid(layer)
    return layer

def learn_neural_network(Xx, yy, Xt, yt, learning_rate, dropout_prob, biases, hiddens, batch_size, number_of_epochs,
                         encoder, type_of_f1_score, is_model_saved, name_of_model):
    in_dim = Xx.shape[1]
    tf.Variable("int", in_dim, name="in_dim_var")
    out_dim = yy.shape[1]
    x = tf.placeholder("float", [None, in_dim], name="input")
    y = tf.placeholder("float", [None, out_dim])
    keep_prob = tf.placeholder("float", name="keep_prob_var")
    dim = [in_dim, *hiddens, out_dim]
    tf.Variable(dim, name="dim_var")
    tf.Variable(hiddens, name="hiddens_var")
    weights = [tf.Variable(tf.random_normal([dim[i - 1], dim[i]]), name="weight_" + str(i)) for i in range(1, len(dim))]
    biases_tf = [tf.Variable(tf.constant(biases[i - 1], shape=[dim[i]]), name="bias_" + str(i)) for i in range(1, len(dim))]
    model_out = create_model(x, dim, hiddens, weights, biases_tf, keep_prob)
    loss = tf.nn.l2_loss(y - model_out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train_losses = []
    val_losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(number_of_epochs):
            tl = []
            for i, (batch_x, batch_y) in enumerate(generate_batches(Xx, yy, batch_size)):
                tl.append(
                    sess.run([optimizer, loss],
                             feed_dict={x: batch_x, y: batch_y, keep_prob: dropout_prob})[
                        1])
            train_loss = np.mean(tl)
            vl = []
            for i, (batch_x, batch_y) in enumerate(generate_batches(Xt, yt, batch_size)):
                vl.append(sess.run(loss, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}))
            val_loss = np.mean(vl)
            print("Epoch: {} training loss {} validation loss {}".format(epoch + 1, train_loss, val_loss))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        results = model_out.eval(feed_dict={x: Xt, keep_prob: 1.0})
        labels = find_labels(results)
        decoded_one_hot = decode_one_hot(labels)
        y_pred = encoder.inverse_transform(decoded_one_hot)
        yt = decode_one_hot(yt)
        yt = encoder.inverse_transform(yt)
        print("Test accuray = {} fi_score = {}".format(accuracy_score(y_pred, yt, normalize=True), f1_score(y_pred, yt,
                average=type_of_f1_score)))
        if is_model_saved:
            saver = tf.train.Saver()
            saver.save(sess, "./" + name_of_model)
    return train_losses, val_losses

def predict_output(Xx, model_path, check_point_path, encoder):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
        graph = tf.get_default_graph()
        #print([n.name for n in graph.as_graph_def().node])
        x = graph.get_tensor_by_name("input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob_var:0")
        dim = graph.get_tensor_by_name("dim_var:0").eval()
        hiddens = graph.get_tensor_by_name("hiddens_var:0").eval()
        weights = [graph.get_tensor_by_name("weight_" + str(i) + ":0").eval() for i in range(1, len(hiddens) + 2)]
        biases = [graph.get_tensor_by_name("bias_" + str(i) + ":0").eval() for i in range(1, len(hiddens) + 2)]
        model_out = create_model(x, dim, hiddens, weights, biases, keep_prob)
        results = sess.run(model_out, feed_dict={x: Xx, keep_prob: 1.0}) #model_out.eval(feed_dict={x: Xx, keep_prob: 1.0})
        label = find_labels(results)
        decoded_one_hot = decode_one_hot(label)
    return encoder.inverse_transform(decoded_one_hot)

def generate_batches(X, y, batch_size):
    X, y = shuffle(X, y)
    batches_num = math.ceil(y.shape[0] / batch_size)
    for i in range(0, batches_num):
        yield X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
