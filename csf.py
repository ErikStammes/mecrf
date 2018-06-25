from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import operator
import pickle

from data_utils_csf import *
from mecrf_csf import *
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import cPickle as pickle

import logging
import gc

from _collections import defaultdict

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory.")
tf.flags.DEFINE_integer("max_sentence_size", 50, "Maximum size of sentence.")
tf.flags.DEFINE_integer("random_state", 101, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/conll03-ner/", "Directory containing CoNLL-03-NER data")
tf.flags.DEFINE_integer("rnn_hidden_size", 20, "RNN hidden size [20]")
tf.flags.DEFINE_string("embedding_file", None, "Pre-trained word embedding file path [None]")
tf.flags.DEFINE_boolean("update_embeddings", False, "Update embeddings [False]")
tf.flags.DEFINE_boolean("bilinear", False, "Use bilinear [False]")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep prob [1.0]")
tf.flags.DEFINE_integer("mlp_hidden_size", 64, "MLP hidden state size [64]")
tf.flags.DEFINE_integer("rnn_memory_hidden_size", 0, "RNN memory hidden size [0]")
tf.flags.DEFINE_string("nonlin", "tanh", "Non-linearity [tanh]")
tf.flags.DEFINE_string("dataset_size", None, "Dataset size (same as filename)")
tf.flags.DEFINE_integer("epochs_without_improvement", 20, "Quit after this number of epochs without improvement")

FLAGS = tf.flags.FLAGS

def get_label_dict(data):
    label2idx = {}
    for document in data:
        for sentence in document:
            for _, label, _ in sentence:
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
    return label2idx

def load_embeddings(data, in_file, binary=False, load_full_vocab=False):
    emb = {}
    unk = []
    with open(in_file) as in_f:
        for line in in_f:
            line = line.strip()
            attrs = line.split(' ')
            if len(attrs) == 2:
                continue
            word = attrs[0]
            word_emb = map(float, attrs[1:])
            emb[word] = word_emb
            unk.append(word_emb)
    unk = np.mean(np.array(unk), axis=0)
    ret_emb = []
    ret_emb.append(np.zeros(len(unk))) # padding
    ret_emb.append(unk) # = 1 in ret_word2idx
    ret_word2idx = {}
    if load_full_vocab:
        for word in emb:
            ret_word2idx[word] = len(ret_emb)
            ret_emb.append(emb[word])
        return np.asarray(ret_emb, dtype=np.float32), ret_word2idx
    for document in data:
        for sentence in document:
            for word, _, _ in sentence:
                if word.lower() in emb:
                    if word not in ret_word2idx:
                        ret_word2idx[word] = len(ret_emb)
                        ret_emb.append(emb[word.lower()])
                else:
                    ret_word2idx[word] = 1 # unk
    return np.asarray(ret_emb, dtype=np.float32), ret_word2idx

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(" ".join(sys.argv))

    if FLAGS.dataset_size is not None:
        train_name = 'train-' + FLAGS.dataset_size + '.json'
    else:
        train_name = 'train.json'
    dev_name = 'dev.json'
    test_name = 'test.json'


    train = load_task(
        os.path.join(FLAGS.data_dir, train_name),
        POS=False
    )
    train_flattened = [s for d in train for s in d]
    val = load_task(
        os.path.join(FLAGS.data_dir, dev_name),
        POS=False
    )
    val_flattened = [s for d in val for s in d]
    
    data = train + val #+ test
    data = np.asarray(data, dtype=np.object)

    logger.info("Loaded data")
    
    assert FLAGS.embedding_file is not None
    embedding_mat, word2idx = load_embeddings(
        data, 
        FLAGS.embedding_file,
        load_full_vocab=True
    )
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    embedding_size = embedding_mat.shape[1]
    
    logger.info('Embedding_mat size: ' + str(embedding_mat.shape))

    np.random.seed(FLAGS.random_state)
    
    max_story_size = max([sum([len(s) for s in d]) for d in data])
    mean_story_size = int(np.mean([sum([len(s) for s in d]) for d in data]))
    max_sentence_size = FLAGS.max_sentence_size
    memory_size = FLAGS.memory_size
    
    label2idx = get_label_dict(data)
    idx2label = dict(zip(label2idx.values(), label2idx.keys()))

    with open('tmp/vocabs.pkl', 'w') as f:
        pickle.dump((word2idx, label2idx, idx2label), f)
    
    vocab_size = embedding_mat.shape[0]

    logger.info("Vocabulary size: %d" % vocab_size)

    answer_size = len(label2idx)
    
    logger.info("Longest sentence length %d" % max_sentence_size)
    logger.info("Longest story length %d" % max_story_size)
    logger.info("Average story length %d" % mean_story_size)
    
    # train/validation/test sets
    train_sentences, train_memories, train_answers, train_mem_idx = vectorize_data(train, word2idx, max_sentence_size, memory_size, label2idx)
    val_sentences, val_memories, val_answers, val_mem_idx = vectorize_data(val, word2idx, max_sentence_size, memory_size, label2idx)
    
    slot_keys = np.unique([iob_tag[2:] for iob_tag in label2idx.keys() if iob_tag != 'O'])

    train_sentence_lexical_features, train_memory_lexical_features = vectorize_lexical_features(train, max_sentence_size, memory_size, slot_keys)
    val_sentence_lexical_features, val_memory_lexical_features = vectorize_lexical_features(val, max_sentence_size, memory_size, slot_keys)

    lexical_features_size = train_sentence_lexical_features.shape[2]

    logger.info("Training set title shape " + str(train_sentences.shape))
    logger.info("Training set text shape " + str(train_memories.shape))
    
    n_train = train_sentences.shape[0]
    n_val = val_sentences.shape[0]
    
    logger.info("Training Size %d" % n_train)
    logger.info("Validation Size %d" % n_val)
    
    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    
    batches = zip(range(0, n_train, batch_size), list(range(batch_size, n_train, batch_size)) + [n_train])
    batches = [(start, end) for start, end in batches]
    
    nonlin = None
    if FLAGS.nonlin == 'tanh':
        nonlin = tf.nn.tanh
    elif FLAGS.nonlin == 'relu':
        nonlin = tf.nn.relu
    else:
        raise
    
    best_score = -1
    best_perf = None

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4,
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
    )

    with tf.Session(config=session_conf) as sess:
        tf.set_random_seed(seed=FLAGS.random_state)
        model = MECRF(
            batch_size,
            vocab_size,
            answer_size,
            max_sentence_size,
            memory_size,
            embedding_size,
            session=sess,
            max_grad_norm=FLAGS.max_grad_norm,
            optimizer=optimizer,
            embedding_mat=embedding_mat,
            rnn_hidden_size=FLAGS.rnn_hidden_size,
            mlp_hidden_size=FLAGS.mlp_hidden_size,
            rnn_memory_hidden_size=FLAGS.rnn_memory_hidden_size,
            nonlin=nonlin,
            lexical_features_size=lexical_features_size,
        )
        
        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            
            for start, end in batches:
                m = train_memories[start:end]
                s = train_sentences[start:end]
                a = train_answers[start:end]
                mi = train_mem_idx[start:end]
                slf = train_sentence_lexical_features[start:end]
                mlf = train_memory_lexical_features[start:end]
                cost_t = model.batch_fit(
                    m, s, a, FLAGS.keep_prob, mi, slf, mlf
                )
                total_cost += cost_t
                
            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    m = train_memories[start:end]
                    s = train_sentences[start:end]
                    a = train_answers[start:end]
                    mi = train_mem_idx[start:end]
                    slf = train_sentence_lexical_features[start:end]
                    mlf = train_memory_lexical_features[start:end]
                    pred = model.predict(m, s, mi, slf, mlf)
                    train_preds += list(pred)
    
                train_scores, acc, precision, recall, f_score = evaluate(
                    train_flattened,
                    [[idx2label[p] for p in pred] for pred in train_preds]
                )
                
                logger.info('-----------------------')
                logger.info('Epoch %d' % t)
                logger.info('Total Cost: %f' % total_cost)
                logging.info('Training acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
                logging.info('Training: ' + train_scores)
    
                val_preds = model.predict(val_memories, val_sentences, val_mem_idx, val_sentence_lexical_features, val_memory_lexical_features)
                val_scores, acc, precision, recall, f_score = evaluate(
                    val_flattened,
                    [[idx2label[p] for p in pred] for pred in val_preds]
                )
                logging.info('Validation acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
                logging.info('Validation: ' + val_scores)
                
                val_f_score = f_score
                val_perf = (acc, precision, recall, f_score, t)

                if val_f_score > best_score:
                    best_score = val_f_score
                    best_perf = val_perf
                    epochs_since_best = 0
                    save_path = model.save_session('tmp/model')
                    logger.info("Model saved in path: %s" % save_path)
                else:
                    epochs_since_best += 1
                    if epochs_since_best > FLAGS.epochs_without_improvement:
                        logger.info('No improvements after %s epochs. Quitting now..' % FLAGS.epochs_without_improvement)
                        sess.close()
                        quit()

                logger.info('-----------------------')
                logger.info('Best val acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f, epoch: %d' % (best_perf))