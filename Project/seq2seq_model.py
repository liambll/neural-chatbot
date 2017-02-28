# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:19:28 2017

@author: linhb
"""
# Seq2Se
import tensorflow as tf
import numpy as np

class Seq2Seq(object):
    # Initialize Computation Graph
    def __init__(self, source_len, target_len, 
            source_vocab_size, target_vocab_size,
            layer_size, num_layers, model_path, model_type,
            attention_heads=1,
            learning_rate=0.0001, 
            epochs=100000):

        self.source_len = source_len #length of source sentence
        self.target_len = target_len # length of target sentence
        self.model_path = model_path
        self.epochs = epochs
        self.model_type = model_type
        self.attention_heads = attention_heads

        # initialize computation graph
        tf.reset_default_graph()
        #  encoder inputs
        self.encoder_inputs = []
        for i in range(source_len):
            self.encoder_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                name="encoder{0}".format(i)))

        # target
        self.targets = []
        for i in range(target_len):
            self.targets.append(tf.placeholder(tf.int64, shape=[None],
                                                name="target{0}".format(i)))
            
        # decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [ tf.zeros_like(self.encoder_inputs[0], dtype=tf.int64, name='GO') ] \
                            + self.targets[:-1]

        # Basic LSTM cell with Dropout
        self.dropout = tf.placeholder(tf.float32)
        # define the basic cell
        basic_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(layer_size, state_is_tuple=True),
                output_keep_prob=self.dropout)
        
        # stack num_layers
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)

        # decoder based on model type
        if self.model_type == 1: # Seq2Seq model
            self.decoder_outputs, self.decoder_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.encoder_inputs,self.decoder_inputs, stacked_lstm,
                    source_vocab_size, target_vocab_size, layer_size)
            # share parameters
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # testing model, where output of previous timestep is fed as input to the next timestep
                self.decoder_outputs_test, self.decoder_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.encoder_inputs,self.decoder_inputs, stacked_lstm, source_vocab_size, 
                    target_vocab_size, layer_size, feed_previous=True)
        else:   #Seq2Seq Model with attention mechanism
            self.decoder_outputs, self.decoder_states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.encoder_inputs,self.decoder_inputs, stacked_lstm,
                    source_vocab_size, target_vocab_size, layer_size, self.attention_heads)
            # share parameters
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # testing model, where output of previous timestep is fed as input to the next timestep
                self.decoder_outputs_test, self.decoder_states_test = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.encoder_inputs,self.decoder_inputs, stacked_lstm, source_vocab_size, 
                    target_vocab_size, layer_size, self.attention_heads, feed_previous=True)

        
        # define loss fucntion
        loss_weights = []
        for target in self.targets:
            loss_weights.append(tf.ones_like(target, dtype=tf.float32))
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_outputs, self.targets, loss_weights, target_vocab_size)
        
        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        print("Finish initialization.")
        
    # train seq2seq model
    def train(self, train_set, valid_set, sess=None ):
        # model saver
        saver = tf.train.Saver()

        # create new session and initiate all variables
        if not sess:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        print("Start training.")
        prev_loss = float('INF')
        # run M epochs
        for i in range(self.epochs):
            try:
                # train batch
                train_source, train_target = train_set.next()                
                feed_dict = self.get_feed(train_source, train_target, 0.5)
                sess.run([self.optimizer, self.loss], feed_dict)
                
                # evaluate every 1000 epochs
                if i and i% (self.epochs//100) == 0:
                    # evaluate
                    losses = []
                    for j in range(16):
                        valid_source, valid_target = valid_set.next() 
                        feed_dict = self.get_feed(valid_source, valid_target, 1.0)
                        valid_loss, _ = sess.run([self.loss, self.decoder_outputs_test], feed_dict)
                        losses.append(valid_loss)
                    val_loss = np.mean(losses)
                    # print loss
                    print('Epoch %i - val_loss: %10.6f' %(i, val_loss))
                    # save best model
                    if (val_loss < prev_loss):
                        prev_loss = val_loss
                        saver.save(sess, self.model_path + 'model.ckpt')
                        print('---Best val_loss: %10.6f' %val_loss)
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess
        
    # get the feed dictionary
    def get_feed(self, X, Y, dropout):
        feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.source_len)}
        feed_dict.update({self.targets[t]: Y[t] for t in range(self.target_len)})
        feed_dict[self.dropout] = dropout # dropout
        return feed_dict
       
    # load pre-trained model
    def load_model(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        checkpoint = tf.train.get_checkpoint_state(self.model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        return sess

    # prediction
    def predict(self, sess, source):
        feed_dict = {self.encoder_inputs[t]: source[t] for t in range(self.source_len)}
        feed_dict[self.dropout] = 1.
        predictions = sess.run(self.decoder_outputs_test, feed_dict)
        # need to transpose 0,1 indices to interchange batch_size and timesteps dimensions
        predictions = np.array(predictions).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(predictions, axis=2)