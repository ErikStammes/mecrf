import os
import re
import uuid
import pickle
import logging

import tensorflow as tf
import numpy as np

from data_utils_csf import vectorize_data, vectorize_lexical_features, load_task, evaluate
from six.moves import range

tf.flags.DEFINE_string("model_loc", "tmp/model", "Location where model can be restored from")
tf.flags.DEFINE_string("test_set", "data/slots/sim-R/test.json", "Test set location")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory")
tf.flags.DEFINE_integer("max_sentence_size", 50, "Maximum sentence size")
tf.flags.DEFINE_integer("n_dialogues", -1, "Use the first n dialogues for evaluation only for quick prototyping, if below zero all dialogues are used")
FLAGS = tf.flags.FLAGS

def get_mini_batch_start_end(n_train, batch_size=None):
    mini_batch_size = n_train if batch_size is None else batch_size
    batches = zip(
        range(0, n_train, mini_batch_size),
        list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
    )
    return batches        

def main():
    # Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Load test dataset
    test = load_task(FLAGS.test_set)
    if FLAGS.n_dialogues > 0:
        test = test[:FLAGS.n_dialogues]
    test_flattened = [s for d in test for s in d]
    logger.info("Dataset loaded")
    
    # Load vocabs/one-hot encoded dicts
    with open('tmp/vocabs.pkl', 'r') as f:
        (word2idx, label2idx, idx2label) = pickle.load(f)
 
    slot_keys = np.unique([iob_tag[2:] for iob_tag in label2idx.keys() if iob_tag != 'O'])        

    # Vectorize data and features
    sentences, memories, _, mem_idx = vectorize_data(test, word2idx, FLAGS.max_sentence_size, FLAGS.memory_size, label2idx)    
    sent_lexical_features, mem_lexical_features = vectorize_lexical_features(test, FLAGS.max_sentence_size, FLAGS.memory_size, slot_keys)
    logger.info("Data vectorized")

    with tf.Session() as sess:
        # Restore model from meta file and variable values from checkpoint file
        new_saver = tf.train.import_meta_graph(FLAGS.model_loc + '.meta')
        new_saver.restore(sess, FLAGS.model_loc)
        logger.info("Model and session restored")

        # Restore operations needed for prediction
        unary_scores_op = tf.get_collection('unary_scores_op')[0]
        transition_params_op = tf.get_collection('transition_params_op')[0]
        sent_lens = tf.get_collection('sent_lens')[0]

        # Divide in batches and get scores
        batches = get_mini_batch_start_end(len(memories), FLAGS.batch_size)
        unary_scores, transition_params, sentence_lens = [], None, []
        for start, end in batches:
            feed_dict = {
                "memories:0": memories[start:end],
                "sentences:0": sentences[start:end], 
                "keep_prob:0": 1.0,
                "doc_start_index:0": mem_idx[start:end],
                "sentence_lexical_features:0": sent_lexical_features[start:end],
                "memory_lexical_features:0": mem_lexical_features[start:end]
            }
            uss, transition_params, sls = sess.run([unary_scores_op, transition_params_op, sent_lens], feed_dict=feed_dict)
            unary_scores.extend(uss)
            sentence_lens.extend(sls)
        predictions = []
        for unary_score, seq_len in zip(unary_scores, sentence_lens):            
            # Remove padding from the scores and tag sequence.
            us = unary_score[:seq_len]
            
            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                us, transition_params
            )
            predictions.append(viterbi_sequence)
        # Evaluate test set
        scores, acc, precision, recall, f_score = evaluate(test_flattened,
                    [[idx2label[p] for p in pred] for pred in predictions]
                )
        print('Test acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
        print('Test: ' + scores)                


if __name__ == "__main__":
    main()