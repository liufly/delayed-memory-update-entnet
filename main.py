from __future__ import absolute_import
from __future__ import print_function

from data_utils_sentihood import *
from vocab_processor import *
from sklearn import metrics
from delayed_entnet_sentihood import Delayed_EntNet_Sentihood
from itertools import chain
from six.moves import range
from collections import defaultdict

import tensorflow as tf
import numpy as np

import sys
import random
import logging
import cPickle as pickle

import pprint
pp = pprint.PrettyPrinter()

tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate for the optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 800, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("sentence_len", 50, "Maximum len of sentence.")
tf.flags.DEFINE_string("task", "Sentihood", "Sentihood")
tf.flags.DEFINE_integer("random_state", 67, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/sentihood/", "Directory containing Sentihood data")
tf.flags.DEFINE_string("opt", "ftrl", "Optimizer [ftrl]")
tf.flags.DEFINE_string("embedding_file_path", None, "Embedding file path [None]")
tf.flags.DEFINE_boolean("update_embeddings", False, "Update embeddings [False]")
tf.flags.DEFINE_boolean("case_folding", True, "Case folding [True]")
tf.flags.DEFINE_integer("n_cpus", 6, "N CPUs [6]")
tf.flags.DEFINE_integer("n_keys", 7, "Number of keys [7]")
tf.flags.DEFINE_integer("n_tied", 2, "Number of tied keys [2]")
tf.flags.DEFINE_float("entnet_input_keep_prob", 0.8, "entnet input keep prob [0.8]")
tf.flags.DEFINE_float("entnet_output_keep_prob", 1.0, "entnet output keep prob [1.0]")
tf.flags.DEFINE_float("entnet_state_keep_prob", 1.0, "entnet state keep prob [1.0]")
tf.flags.DEFINE_float("final_layer_keep_prob", 0.8, "final layer keep prob [0.8]")
tf.flags.DEFINE_float("l2_final_layer", 1e-3, "Lambda L2 final layer [1e-3]")

FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(" ".join(sys.argv))
    logger.info("Started Task: %s" % FLAGS.task)
    
    logger.info(pp.pformat(FLAGS.__flags))

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=FLAGS.n_cpus,
        inter_op_parallelism_threads=FLAGS.n_cpus,
    )

    aspect2idx = {
        'general': 0,
        'price': 1,
        'transit-location': 2,
        'safety': 3,
    }

    assert FLAGS.n_keys >= 2
    assert FLAGS.n_tied == 2

    with tf.Session(config=session_conf) as sess:

        np.random.seed(FLAGS.random_state)

        # task data
        (train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task(FLAGS.data_dir, aspect2idx)

        if FLAGS.case_folding:
            train = lower_case(train)
            val = lower_case(val)
            test = lower_case(test)

        data = train + val + test

        max_sentence_len = max(map(lambda x: len(x[1]), data))
        max_sentence_len = min(FLAGS.sentence_len, max_sentence_len)
        logger.info('Max sentence len: %d' % max_sentence_len)
        max_target_len = 1 # should be one
        max_aspect_len = max(map(lambda x: len(x), [d[3] for d in data]))
        assert max_aspect_len == 2
        logger.info('Max target size: %d' % max_target_len)
        logger.info('Max aspect size: %d' % max_aspect_len)

        assert FLAGS.embedding_file_path is not None
        word_vocab = EmbeddingVocabulary(
            in_file=FLAGS.embedding_file_path,
        )
        word_vocab_processor = EmbeddingVocabularyProcessor(
            max_document_length=max_sentence_len,
            vocabulary=word_vocab,
        )
        embedding_mat = word_vocab.embeddings
        embedding_size = word_vocab.embeddings.shape[1]

        label_vocab = LabelVocabulary()
        label_vocab_processor = LabelVocabularyProcessor(
            vocabulary=label_vocab,
            min_frequency=0,
        )

        positive_idx = label_vocab.get('Positive')
        negative_idx = label_vocab.get('Negative')
        none_idx = label_vocab.get('None')

        train_sentences, train_targets, train_loc_indicators, train_aspects, train_labels, train_ids = vectorize_data(
            train,
            max_sentence_len,
            max_target_len,
            max_aspect_len,
            word_vocab_processor,
            label_vocab_processor,
        )
        
        val_sentences, val_targets, val_loc_indicators, val_aspects, val_labels, val_ids = vectorize_data(
            val,
            max_sentence_len,
            max_target_len,
            max_aspect_len,
            word_vocab_processor,
            label_vocab_processor,
        )
        
        test_sentences, test_targets, test_loc_indicators, test_aspects, test_labels, test_ids = vectorize_data(
            test,
            max_sentence_len,
            max_target_len,
            max_aspect_len,
            word_vocab_processor,
            label_vocab_processor,
        )

        target_terms = [['location1'], ['location2']]
        target_terms = word_vocab_processor.transform(target_terms)[:, :max_target_len]
        
        sentence_len = max_sentence_len
        vocab_size = len(word_vocab)
        answer_size = len(label_vocab)

        logger.info("Training sentences shape " + str(train_sentences.shape))
        logger.info("Training targets shape " + str(train_targets.shape))
        logger.info("Training aspects shape " + str(train_aspects.shape))
        logger.info("Validation sentences shape " + str(val_sentences.shape))
        logger.info("Validation targets shape " + str(val_targets.shape))
        logger.info("Validation aspects shape " + str(val_aspects.shape))
        logger.info("Test sentences shape " + str(test_sentences.shape))
        logger.info("Test targets shape " + str(test_targets.shape))
        logger.info("Test aspects shape " + str(test_aspects.shape))
        
        # params
        n_train = train_sentences.shape[0]
        n_val = val_sentences.shape[0]
        n_test = test_sentences.shape[0]
        
        logger.info("Training Size %d" % n_train)
        logger.info("Validation Size %d" % n_val)
        logger.info("Testing Size %d" % n_test)
        
        tf.set_random_seed(FLAGS.random_state)
        batch_size = FLAGS.batch_size
        
        global_step = None
        optimizer = None

        train_positive_idx = np.where(train_labels == positive_idx)[0]
        train_negative_idx = np.where(train_labels == negative_idx)[0]
        train_none_idx = np.where(train_labels == none_idx)[0]

        train_positive_sentences = train_sentences[train_positive_idx]
        train_positive_targets = train_targets[train_positive_idx]
        train_positive_aspects = train_aspects[train_positive_idx]
        train_positive_labels = train_labels[train_positive_idx]

        train_negative_sentences = train_sentences[train_negative_idx]
        train_negative_targets = train_targets[train_negative_idx]
        train_negative_aspects = train_aspects[train_negative_idx]
        train_negative_labels = train_labels[train_negative_idx]

        train_none_sentences = train_sentences[train_none_idx]
        train_none_targets = train_targets[train_none_idx]
        train_none_aspects = train_aspects[train_none_idx]
        train_none_labels = train_labels[train_none_idx]

        assert len(train_none_idx) > len(train_positive_idx)
        assert len(train_positive_idx) > len(train_negative_idx)

        n_positive_train = len(train_positive_idx)
        n_negative_train = len(train_negative_idx)
        n_none_train = len(train_none_idx)
        n_train = n_negative_train # down-sampling

        logger.info("Positive training Size %d" % n_positive_train)
        logger.info("Negative training Size %d" % n_negative_train)
        logger.info("None training Size %d" % n_none_train)

        if FLAGS.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
        elif FLAGS.opt == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate=FLAGS.learning_rate
            )

        batches = zip(
            range(0, max(1, n_train-batch_size), batch_size), 
            range(batch_size, max(batch_size + 1, n_train), batch_size)
        )
        batches = [(start, end) for start, end in batches]
        
        model = Delayed_EntNet_Sentihood(
            batch_size, 
            vocab_size, 
            max_target_len,
            max_aspect_len,
            sentence_len, 
            answer_size,
            embedding_size, 
            session=sess,
            embedding_mat=word_vocab.embeddings,
            update_embeddings=FLAGS.update_embeddings,
            n_keys=FLAGS.n_keys,
            tied_keys=target_terms,
            l2_final_layer=FLAGS.l2_final_layer,
            max_grad_norm=FLAGS.max_grad_norm, 
            optimizer=optimizer,
            global_step=global_step
        )
        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            total_training_instances = 0
            
            for start, end in batches:
                # train negative
                sentences = train_negative_sentences[start:end]
                targets = train_negative_targets[start:end]
                aspects = train_negative_aspects[start:end]
                answers = train_negative_labels[start:end]
                cost_t = model.fit(sentences, targets, aspects, answers,
                                   FLAGS.entnet_input_keep_prob,
                                   FLAGS.entnet_output_keep_prob,
                                   FLAGS.entnet_state_keep_prob,
                                   FLAGS.final_layer_keep_prob)
                total_cost += cost_t
                total_training_instances += len(train_negative_sentences[start:end])

                # train positive
                positive_start = random.randint(0, n_positive_train - batch_size)
                positive_end = positive_start + batch_size
                sentences = train_positive_sentences[positive_start:positive_end]
                targets = train_positive_targets[positive_start:positive_end]
                aspects = train_positive_aspects[positive_start:positive_end]
                answers = train_positive_labels[positive_start:positive_end]
                cost_t = model.fit(sentences, targets, aspects, answers, 
                                   FLAGS.entnet_input_keep_prob,
                                   FLAGS.entnet_output_keep_prob,
                                   FLAGS.entnet_state_keep_prob,
                                   FLAGS.final_layer_keep_prob)
                total_cost += cost_t
                total_training_instances += len(train_positive_sentences[positive_start:positive_end])

                # train none
                none_start = random.randint(0, n_none_train - batch_size)
                none_end = none_start + batch_size
                sentences = train_none_sentences[none_start:none_end]
                targets = train_none_targets[none_start:none_end]
                aspects = train_none_aspects[none_start:none_end]
                answers = train_none_labels[none_start:none_end]
                cost_t = model.fit(sentences, targets, aspects, answers, 
                                   FLAGS.entnet_input_keep_prob,
                                   FLAGS.entnet_output_keep_prob,
                                   FLAGS.entnet_state_keep_prob,
                                   FLAGS.final_layer_keep_prob)

                total_cost += cost_t
                total_training_instances += len(train_none_sentences[none_start:none_end])
    
            if t % FLAGS.evaluation_interval == 0:
                train_preds, train_preds_prob = model.predict(
                    train_sentences, train_targets, train_aspects, 
                    batch_size=batch_size,
                )
                
                train_acc = metrics.accuracy_score(
                    train_labels, np.array(train_preds)
                )

                val_preds, val_preds_prob = model.predict(
                    val_sentences, val_targets, val_aspects, 
                    batch_size=batch_size,
                )
    
                val_acc = metrics.accuracy_score(
                    val_labels, np.array(val_preds)
                )

                test_preds, test_preds_prob = model.predict(
                    test_sentences, test_targets, test_aspects,
                    batch_size=batch_size
                )
                test_acc = metrics.accuracy_score(
                    test_labels, np.array(test_preds)
                )

                assert total_training_instances != 0

                logger.info('-----------------------')
                logger.info('Epoch %d' % t)
                logger.info('Avg Cost: %f' % (total_cost / total_training_instances))
                logger.info('Training Accuracy: %f' % train_acc)
                logger.info('Validation Accuracy: %f' % val_acc)
                logger.info('Test Accuracy: %f' % test_acc)
                logger.info('-----------------------')
