from __future__ import absolute_import
from __future__ import division

import os
import re
import string
import json
from copy import copy
import numpy as np

def load_task(in_file, POS=False):
    with open(in_file) as f:
        return parse_conversations(f.read(), POS=POS)

def extract_tags_from_slots(utterance):
    tags_list = ['O'] * len(utterance['tokens'])
    if 'slots' not in utterance:
        return tags_list
    for slot in utterance['slots']:
        start, end = slot['start'], slot['exclusive_end']
        tags_list[start] = 'B-'+slot['slot']
        for i in range(1, end-start):
            tags_list[start+i] = 'I-'+slot['slot']
    return tags_list

# TODO: properly do this!
# use actual values from acts
def extract_acts_from_turn(turn, act_key):
    features = {}
    if act_key not in turn:
        return features
    acts = turn[act_key]
    for act in acts:
        act_type = act['type']
        if act_type not in features:
            features[act_type] = []
        if 'slot' in act:
            act_slot = act['slot']
            features[act_type].append(act_slot)
        else:
            features[act_type].append(0)
    return features


def extract_features_from_utterance(POS, turn_type, turn, turn_ratio):
    utterance_key = turn_type + '_utterance'
    act_key = turn_type + '_acts'
    tags = extract_tags_from_slots(turn[utterance_key])
    sentence = []
    turn_acts = extract_acts_from_turn(turn, act_key)
    sentence_features = {'turn_type': turn_type,
                         'turn_ratio': turn_ratio,
                         'turn_acts': turn_acts}
    sentence.append(('<START>', 'O', sentence_features))
    if POS:
        pos_tags = turn[turn_type + '_pos']
    for j, word in enumerate(turn[utterance_key]['tokens']):
        features = {}
        if POS:
            features['pos_tag'] = pos_tags[j]
        sentence.append((word, tags[j], features))
    return sentence

def parse_conversations(data, POS=False):
    '''Parse conversations from the simulated dialogue (Google) dataset in json
    format '''
    dialogues = json.loads(data)
    data = []
    for dialogue in dialogues:
        conversation = []
        total_turns = len(dialogue['turns'])
        for i, turn in enumerate(dialogue['turns']):
            if 'system_utterance' in turn:
                sentence = extract_features_from_utterance(POS, 'system', turn, i/total_turns)
                conversation.append(sentence)
            if 'user_utterance' in turn:
                sentence = extract_features_from_utterance(POS, 'user', turn, i/total_turns)
                conversation.append(sentence)
        data.append(conversation)   
    return data

def vectorize_data(data, word2idx, sentence_size, memory_size, ner2idx):
    nb_sentence = map(len, data)
    nb_sentences = sum(nb_sentence)
    ret_sentences = np.zeros((nb_sentences, sentence_size))
    ret_memories = np.zeros((nb_sentences, memory_size))
    ret_answers = np.zeros((nb_sentences, sentence_size))
    ret_mem_idx = np.zeros((nb_sentences, sentence_size))

    for i, document in enumerate(data):
        memory = []
        for j, sentence in enumerate(document):
            for k, (word, ner, _) in enumerate(sentence):
                idx = sum(nb_sentence[:i]) + j
                ret_sentences[idx, k] = word2idx[word] if word in word2idx else 1 # 1 for unk
                ret_answers[idx, k] = ner2idx[ner]
                ret_mem_idx[idx, k] = sum([len(s) for s in document[:j]]) + k # memory accessible to the current word inclusively
                memory.append(ret_sentences[idx, k])
        memory = memory[:memory_size]
        idx_start = sum(nb_sentence[:i])
        for j, sentence in enumerate(document):
            ret_memories[idx_start + j, :len(memory)] = memory

    return ret_sentences, ret_memories, ret_answers, ret_mem_idx

class AbstractFeature(object):
    def generate_feature(self, word, features):
        raise NotImplementedError("Not implemented")

    def feature_size(self):
        raise NotImplementedError("Not implemented")

class CapitalizationFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if word[0].isupper() else 0
    def feature_size(self):
        return 1

class AllCapitalizedFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if word.isupper() else 0
    def feature_size(self):
        return 1

class AllLowerFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if word.islower() else 0
    def feature_size(self):
        return 1

class NonInitialCapFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if any([c.isupper() for c in word[1:]]) else 0
    def feature_size(self):
        return 1

class MixCharDigitFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if any([c.isalpha() for c in word]) and any([c.isdigit() for c in word]) else 0
    def feature_size(self):
        return 1

class HasPunctFeature(AbstractFeature):
    def __init__(self):
        self._punct_set = set(string.punctuation)
    def generate_feature(self, word, features):
        return 1 if any([c in self._punct_set for c in word]) else 0
    def feature_size(self):
        return 1

class PreSuffixFeature(AbstractFeature):
    def __init__(self, window_size, is_prefix):
        self._vocab = {}
        self._window_size = window_size
        self._is_prefix = is_prefix
    def generate_feature(self, word, features):
        w = word.lower()
        fix = w[:self._window_size] if self._is_prefix else w[-self._window_size:]
        if fix in self._vocab:
            return self._vocab[fix]
        else:
            self._vocab[fix] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[fix]
    def feature_size(self):
        return len(self._vocab) + 1

class HasApostropheFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if word.lower()[-2:] == "'s" else 0
    def feature_size(self):
        return 1

class LetterOnlyFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}

    def generate_feature(self, word, features):
        w = filter(lambda x: x.isalpha(), word)
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]

    def feature_size(self):
        return len(self._vocab) + 1

class NonLetterOnlyFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}

    def generate_feature(self, word, features):
        w = filter(lambda x: not x.isalpha(), word)
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]

    def feature_size(self):
        return len(self._vocab) + 1

class WordPatternFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}
    def generate_feature(self, word, features):
        w = []
        for c in word:
            if c.isalpha() and c.islower():
                w.append('a')
            elif c.isalpha() and c.isupper():
                w.append('A')
            elif c.isdigit():
                w.append('0')
            else:
                w.append
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]
    def feature_size(self):
        return len(self._vocab) + 1

class WordPatternSummarizationFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}
    def generate_feature(self, word, features):
        w = []
        for c in word:
            if c.isalpha() and c.islower():
                if len(w) == 0 or w[-1] != 'a':
                    w.append('a')
            elif c.isalpha() and c.isupper():
                if len(w) == 0 or w[-1] != 'A':
                    w.append('A')
            elif c.isdigit():
                if len(w) == 0 or w[-1] != '0':
                    w.append('0')
            else:
                w.append
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]
    def feature_size(self):
        return len(self._vocab) + 1

class TurnTypeIsUserFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if 'turn_type' in features and features['turn_type'] == 'user' else 0
    def feature_size(self):
        return 1

class TurnRatioFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return features['turn_ratio'] if 'turn_ratio' in features else 0
    def feature_size(self):
        return 1

class PosTagIsVerbFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if 'pos_tag' in features and features['pos_tag'] in ['VERB', 'VB', 'VBD', 'VBG', 'VBN', \
         'VBP', 'VBZ' , 'MD', 'HVS', 'BES'] else 0
    def feature_size(self):
        return 1

class PosTagIsSymbolFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if 'pos_tag' in features and features['pos_tag'] == ['SYM'] else 0
    def feature_size(self):
        return 1

class PosTagIsPunctuationFeature(AbstractFeature):
    def generate_feature(self, word, features):
        return 1 if 'pos_tag' in features and features['pos_tag'] in ['PUNCT', '-LRB-', '-RRB-', ',', \
         ':', '.', '\'\'', '""', '#', '``', '$'] else 0
    def feature_size(self):
        return 1

class SpeechActSlotFeature(AbstractFeature):
    def __init__(self, targets, act_type):
        self._targets = targets
        self._act_type = act_type
    def generate_feature(self, word, features):
        rvalue = [0] * len(self._targets)
        if 'turn_type' not in features or \
                features['turn_type'] != 'system' or \
                'turn_acts' not in features or \
                self._act_type not in features['turn_acts']:
            return rvalue
        act_values = features['turn_acts'][self._act_type]
        for i, target in enumerate(self._targets):
            if target in act_values:
                rvalue[i] = 1
        return rvalue

    def feature_size(self):
        return len(self._targets)

class SpeechActFeature(AbstractFeature):
    def __init__(self):
        self._acts = ['REQUEST', 'SELECT', 'CONFIRM', 'NOTIFY_SUCCESS',
                      'NEGATE', 'NOTIFY_FAILURE', 'OFFER']
    def generate_feature(self, word, features):
        rvalue = [0] * len(self._acts)
        if 'turn_type' not in features or \
                features['turn_type'] != 'system' or \
                'turn_acts' not in features:
            return rvalue
        acts = features['turn_acts']
        for i, act in enumerate(self._acts):
            if act in acts:
                rvalue[i] = 1
        return rvalue
    def feature_size(self):
        return len(self._acts)

'''
When the system lets the user select a value for a certain slot the speech act is SELECT
The value is the proposed slot value.
'''
class SelectSlotFeature(AbstractFeature):
    def generate_feature(self, word, features):
        if 'turn_type' not in features or features['turn_type'] != 'system' or \
                'turn_acts' not in features or 'SELECT' not in features['turn_acts']:
            return 0
        proposed_slot_values = features['turn_acts']['SELECT']
        for value in proposed_slot_values:
            if word in value.split(): #e.g. the value is '7.15 pm' then both words '7.15' and 'pm' match
                return 1
        return 0
    def feature_size(self):
        return 1

# TODO: Caveat: only matches single word regexes!
class RegexFeature(AbstractFeature):
    def __init__(self, regexes):
        self._regexes = regexes
    def generate_feature(self, word, features):
        rvalue = [0] * len(self._regexes)
        for i, regex in enumerate(self._regexes):
            match = re.search(regex, word)
            if match is not None:
                rvalue[i] = 1
        return rvalue
    def feature_size(self):
        return len(self._regexes)


def vectorize_lexical_features(data, sentence_size, memory_size, targets):
    feature_list = []
    mx_char_digit_feature = MixCharDigitFeature()
    has_punct_feature = HasPunctFeature()
    non_letter_feature = NonLetterOnlyFeature()
    turntype_feature = TurnTypeIsUserFeature()
    turnratio_feature = TurnRatioFeature()
    pos_verb_feature = PosTagIsVerbFeature()
    pos_symbol_feature = PosTagIsSymbolFeature()
    pos_punctuation_feature = PosTagIsPunctuationFeature()
    select_slot_feature = SelectSlotFeature()
    feature_list = [
        mx_char_digit_feature,
        has_punct_feature,
        non_letter_feature,
        turntype_feature,
        turnratio_feature,
        #pos_verb_feature,
        #pos_symbol_feature,
        #pos_punctuation_feature,
        select_slot_feature
    ]
    act_slot_features = [SpeechActSlotFeature(targets, act) for act in ['REQUEST', 'SELECT', 'CONFIRM', 'OFFER']]
    multidim_feature_list = act_slot_features + [SpeechActFeature()] + [RegexFeature([r'[ap]m', r'\d'])]
    lexical_feature_size = sum([f.feature_size() for f in feature_list + multidim_feature_list])
    nb_sentence = map(len, data)
    nb_sentences = sum(nb_sentence)
    sentence_lexical_features = np.zeros((nb_sentences, sentence_size, lexical_feature_size))
    memory_lexical_features = np.zeros((nb_sentences, memory_size, lexical_feature_size))
    for i, document in enumerate(data):
        mlf = []
        for j, sentence in enumerate(document):
            for k, (word, _, word_features) in enumerate(sentence):
                idx = sum(nb_sentence[:i]) + j
                features = [f.generate_feature(word, word_features) for f in feature_list]
                mdfeatures = [f.generate_feature(word, word_features) for f in multidim_feature_list]
                map(features.extend, mdfeatures) #extend the original features list w/ the multidimensional features
                sentence_lexical_features[idx, k] = features
                mlf.append(features)
        mlf = mlf[:memory_size]
        idx_start = sum(nb_sentence[:i])
        for j, sentence in enumerate(document):
            memory_lexical_features[idx_start + j, :len(mlf), :] = mlf
    return sentence_lexical_features, memory_lexical_features
