import pickle

import tensorflow as tf
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

from data_utils_csf import load_task, vectorize_data, vectorize_lexical_features, evaluate


app = Flask(__name__)

tf.flags.DEFINE_string("model_loc", "tmp/model", "Location where model can be restored from")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory")
tf.flags.DEFINE_integer("max_sentence_size", 50, "Maximum sentence size")
FLAGS = tf.flags.FLAGS

@app.route("/classify", methods=['POST'])
def classify():
    output_data = model_api(request.data)
    return jsonify(output_data)


def get_model_api():
    # Load vocabs/one-hot encoded dicts
    with open('tmp/vocabs.pkl', 'r') as f:
        (word2idx, label2idx, idx2label) = pickle.load(f)
    slot_keys = np.unique([iob_tag[2:] for iob_tag in label2idx.keys() if iob_tag != 'O'])
           
    sess = tf.Session()
    # Restore model from meta file and variable values from checkpoint file
    new_saver = tf.train.import_meta_graph(FLAGS.model_loc + '.meta')
    new_saver.restore(sess, FLAGS.model_loc)

    # Restore operations needed for prediction
    unary_scores_op = tf.get_collection('unary_scores_op')[0]
    transition_params_op = tf.get_collection('transition_params_op')[0]
    sent_lens = tf.get_collection('sent_lens')[0]

    def model_api(input_data):
        dialogue = load_task(input_data, direct_input=True)
        d_flattened = [s for d in dialogue for s in d]

        sentences, memories, _, mem_idx = vectorize_data(dialogue, word2idx, FLAGS.max_sentence_size, FLAGS.memory_size, label2idx)
        sent_lexical_features, mem_lexical_features = vectorize_lexical_features(dialogue, FLAGS.max_sentence_size, FLAGS.memory_size, slot_keys)    
        feed_dict = {
            "memories:0": memories,
            "sentences:0": sentences, 
            "keep_prob:0": 1.0,
            "doc_start_index:0": mem_idx,
            "sentence_lexical_features:0": sent_lexical_features,
            "memory_lexical_features:0": mem_lexical_features
        }
        uss, transition_params, sls = sess.run([unary_scores_op, transition_params_op, sent_lens], feed_dict=feed_dict)
        predictions = []
        for unary_score, seq_len in zip(uss, sls):
            # Remove padding from the scores and tag sequence.
            us = unary_score[:seq_len]
            
            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                us, transition_params
            )
            predictions.append(viterbi_sequence)
        input_sequences = [[token[0] for token in sentence if token[0] != '<START>'] for sentence in d_flattened]
        pred_sequences = [[idx2label[p] for p in pred] for pred in predictions]
        scores, acc, precision, recall, f_score = evaluate(d_flattened, pred_sequences)
        aligned_sequences = {}
        for i in range(len(input_sequences)):
            aligned_sequences['%02d_uttr' % i] = ' '.join(input_sequences[i])
            aligned_sequences['%02d_tags' % i] = ' '.join(pred_sequences[i])
        return {"accuracy": acc, "precision": precision, "recall": recall, "f_score": f_score, "results": aligned_sequences}
    return model_api

model_api = get_model_api()
