from __future__ import absolute_import

import os, sys
import re
import numpy as np
import xml.etree.ElementTree
from collections import defaultdict
import nltk
# from vocab_processor import *
import operator
import json

def vectorize_data(sentences, max_sentence_len, max_target_len, max_aspect_len, 
                   word_processor, label_processor):
    ret_sentences = word_processor.transform(
        [text for _, text, _, _, _ in sentences]
    )
    # [None, max_sentence_len]
    assert ret_sentences.shape[1] == max_sentence_len

    ret_loc_indicator = np.zeros((len(sentences), 1), dtype=np.int32)
    for i, (_, _, target, _, _) in enumerate(sentences):
        assert target.lower() in ['location1', 'location2']
        ret_loc_indicator[i, :] = [0 if target.lower() == 'location1' else 1]
    
    ret_targets = word_processor.transform(
        [[target] for _, _, target, _, _ in sentences]
    )
    assert ret_targets.shape[1] == max_sentence_len
    ret_targets = ret_targets[:, :max_target_len]

    ret_aspects = word_processor.transform(
        [aspect_term for _, _, _, aspect_term, _ in sentences]
    )
    assert ret_aspects.shape[1] == max_sentence_len
    ret_aspects = ret_aspects[:, :max_aspect_len]

    ret_label = label_processor.transform(
        [label for _, _, _, _, label in sentences]
    )
    # [None, 1]

    ret_ids = [sent_id for sent_id, _, _, _, _ in sentences]
    return ret_sentences, ret_targets, ret_loc_indicator, ret_aspects, ret_label, np.array(ret_ids, dtype=np.object)

def load_task(data_dir, aspect2idx):
    in_file = os.path.join(data_dir, 'sentihood-train.json')
    train = parse_sentihood_json(in_file)
    in_file = os.path.join(data_dir, 'sentihood-dev.json')
    dev = parse_sentihood_json(in_file)
    in_file = os.path.join(data_dir, 'sentihood-test.json')
    test = parse_sentihood_json(in_file)
    
    train = convert_input(train, aspect2idx)
    train_aspect_idx = get_aspect_idx(train, aspect2idx)
    train = tokenize(train)
    dev = convert_input(dev, aspect2idx)
    dev_aspect_idx = get_aspect_idx(dev, aspect2idx)
    dev = tokenize(dev)
    test = convert_input(test, aspect2idx)
    test_aspect_idx = get_aspect_idx(test, aspect2idx)
    test = tokenize(test)

    return (train, train_aspect_idx), (dev, dev_aspect_idx), (test, test_aspect_idx)

def get_aspect_idx(data, aspect2idx):
    ret = []
    for _, _, _, aspect, _ in data:
        ret.append(aspect2idx[aspect])
    assert len(data) == len(ret)
    return np.array(ret)

def remove_replacement(data, replacement):
    ret_data = []
    ret_indices = []
    for sent in data:
        text = sent[0]
        assert replacement in text
        index = text.index(replacement)
        new_text = text[:index] + text[index+1:]
        ret_data.append((
            new_text, sent[1], sent[2]
        ))
        ret_indices.append(index)
    return ret_data, ret_indices

def lower_case(data):
    ret = []
    for sent_id, text, target, aspect, sentiment in data:
        new_text = map(lambda x: x.lower(), text)
        new_aspect = map(lambda x: x.lower(), aspect)
        ret.append((sent_id, new_text, target.lower(), new_aspect, sentiment))
    return ret

def parse_sentihood_json(in_file):
    with open(in_file) as f:
        data = json.load(f)
    ret = []
    for d in data:
        text = d['text']
        sent_id = d['id']
        opinions = []
        targets = set()
        for opinion in d['opinions']:
            sentiment = opinion['sentiment']
            aspect = opinion['aspect']
            target_entity = opinion['target_entity']
            targets.add(target_entity)
            opinions.append((target_entity, aspect, sentiment))
        ret.append((sent_id, text, opinions))
    return ret

def get_all_aspects(data):
    aspects = set()
    for sent_id, text, opinions in data:
        for target_entity, aspect, sentiment in opinions:
            aspects.add(aspect)
    return aspects

def convert_input(data, all_aspects):
    ret = []
    for sent_id, text, opinions in data:
        for target_entity, aspect, sentiment in opinions:
            if aspect not in all_aspects:
                continue
            ret.append((sent_id, text, target_entity, aspect, sentiment))
        assert 'LOCATION1' in text
        targets = set(['LOCATION1'])
        if 'LOCATION2' in text:
            targets.add('LOCATION2')
        for target in targets:
            aspects = set([a for t, a, _ in opinions if t == target])
            none_aspects = [a for a in all_aspects if a not in aspects]
            for aspect in none_aspects:
                ret.append((sent_id, text, target, aspect, 'None'))
    return ret
        
def tokenize(data):
    ret = []
    for sent_id, text, target_entity, aspect, sentiment in data:
        new_text = nltk.word_tokenize(text)
        new_aspect = aspect.split('-')
        ret.append((sent_id, new_text, target_entity, new_aspect, sentiment))
    return ret
