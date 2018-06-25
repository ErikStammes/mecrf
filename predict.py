import os
import re
import uuid
import pickle

import tensorflow as tf
import numpy as np

from data_utils_csf import vectorize_data, vectorize_lexical_features, load_task
from itertools import chain
from six.moves import range

''' TODO: use flags for:
- model location
- test file location
- embedding file location
- memory size (or get it from restoration?)
- max sentence size
'''

# TODO: move this to data_utils_csf.py?
def output_conll(Gold, Pred, out_F, eval_sys):
    with open(out_F, 'w+') as f:
        assert len(Gold) == len(Pred)
        for gold, pred in zip(Gold, Pred):
            if not eval_sys and gold[0][2]['turn_type'] != 'user':
                continue
            assert len(gold) == len(pred)
            for g, p in zip(gold, pred):
                f.write(' '.join([g[0], g[1], p]))
                f.write('\n')
            f.write('\n')    

regex_pattern = r'accuracy:\s+([\d]+\.[\d]+)%; precision:\s+([\d]+\.[\d]+)%; recall:\s+([\d]+\.[\d]+)%; FB1:\s+([\d]+\.[\d]+)'
def eval(gold, pred, eval_sys=True):
    out_filename = str(uuid.uuid4())
    cur_dir = os.path.dirname(__file__)
    out_abs_filepath = os.path.abspath(os.path.join(cur_dir, 'output', out_filename))
    try:
        output_conll(gold, pred, out_abs_filepath, eval_sys)
        cmd_process = os.popen(
            "perl " + os.path.abspath(os.path.join(cur_dir, "conlleval.pl")) + " < " + out_abs_filepath)
        cmd_ret = cmd_process.read()
        cmd_ret_str = str(cmd_ret)
        m = re.search(regex_pattern, cmd_ret)
        assert m is not None
        acc = float(m.group(1))
        precision = float(m.group(2))
        recall = float(m.group(3))
        f_score = float(m.group(4))
        return cmd_ret_str, acc, precision, recall, f_score
    except:
        return '', 0., 0., 0., 0.
    finally:
        # pass
        os.remove(out_abs_filepath)    

def get_mini_batch_start_end(n_train, batch_size=None):
    '''
    Args:
        n_train: int, number of training instances
        batch_size: int (or None if full batch)
    
    Returns:
        batches: list of tuples of (start, end) of each mini batch
    '''
    mini_batch_size = n_train if batch_size is None else batch_size
    batches = zip(
        range(0, n_train, mini_batch_size),
        list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
    )
    return batches        

def main():
    #TODO divide this in functions
    test = load_task('data/slots/sim-R/test.json')
    test_flattened = [s for d in test for s in d]
    max_sentence_size = 500
    max_story_size = max([sum([len(s) for s in d]) for d in test])
    memory_size = min(500, max_story_size)
    
    with open('tmp/vocabs.pkl', 'r') as f:
        (word2idx, label2idx, idx2label) = pickle.load(f)

    slot_keys = np.unique([iob_tag[2:] for iob_tag in label2idx.keys() if iob_tag != 'O'])        

    sentences, memories, answers, mem_idx = vectorize_data(test, word2idx, max_sentence_size, memory_size, label2idx)    
    sent_lexical_features, mem_lexical_features = vectorize_lexical_features(test, max_sentence_size, memory_size, slot_keys)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('tmp/model.meta')
        new_saver.restore(sess, 'tmp/model')
        unary_scores_op = tf.get_collection('unary_scores_op')[0]
        transition_params_op = tf.get_collection('transition_params_op')[0]
        sent_lens = tf.get_collection('sent_lens')[0]

        n_train = len(memories)
        batches = get_mini_batch_start_end(n_train, 32)
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
        scores, acc, precision, recall, f_score = eval(test_flattened,
                    [[idx2label[p] for p in pred] for pred in predictions]
                )
        print('Test acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
        print('Test: ' + scores)                


if __name__ == "__main__":
    main()