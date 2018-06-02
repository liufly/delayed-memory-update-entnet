from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

from tensorflow import name_scope

from functools import partial

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn.ops import gen_gru_ops
from tensorflow.python.ops import init_ops


class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.
    """

    def __init__(self,
                 num_blocks,
                 num_units_per_block,
                 keys,
                 initializer=None,
                 recurrent_initializer=None,
                 activation=tf.nn.relu,):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._keys = keys
        self._activation = activation # \phi
        self._initializer = initializer
        self._recurrent_initializer = recurrent_initializer

    @property
    def state_size(self):
        "Return the total state size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block * 2

    @property
    def output_size(self):
        "Return the total output size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block

    def zero_state(self, batch_size, dtype):
        "Initialize the memory to the key values."
        zero_state = tf.concat([tf.expand_dims(key, axis=0) for key in self._keys], axis=1)
        zero_state_batch = tf.tile(zero_state, [batch_size, 1])
        return tf.concat(
            values=[
                zero_state_batch,
                tf.zeros(
                    shape=[batch_size, self._num_blocks * self._num_units_per_block],
                    dtype=tf.float32,
                ),
            ],
            axis=1
        )

    def get_gate(self, state_j, key_j, inputs, v=None, prev_a=None):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = tf.reduce_sum(inputs * state_j, axis=1)
        b = tf.reduce_sum(inputs * key_j, axis=1)
        assert v is not None
        c = tf.reduce_sum(prev_a * v, axis=1)
        return tf.sigmoid(a + b + c)

    def get_candidate(self, state_j, key_j, inputs, U, V, W, U_bias):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = tf.matmul(key_j, V)
        state_U = tf.matmul(state_j, U) + U_bias
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + inputs_W + key_V)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)
            V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)
            W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)

            U_bias = tf.get_variable('U_bias', [self._num_units_per_block])

            state, state_a = tf.split(
                value=state,
                num_or_size_splits=[
                    self._num_blocks * self._num_units_per_block,
                    self._num_blocks * self._num_units_per_block
                ],
                axis=1,
            )
            state_a = tf.split(state_a, self._num_blocks, axis=1)
            assert len(state_a) == self._num_blocks

            # Split the hidden state into blocks (each U, V, W are shared across blocks).
            state = tf.split(state, self._num_blocks, axis=1)
            assert len(state) == self._num_blocks

            next_states = []
            next_a_states = []
            for j, state_j in enumerate(state): # Hidden State (j)
                key_j = tf.expand_dims(self._keys[j], axis=0)
                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W, U_bias)

                reuse = False
                if j != 0:
                    reuse = True
                with tf.variable_scope("entnet_gru", reuse=reuse) as gru_scope:
                    w_ru = tf.get_variable(
                        "w_ru", 
                        [self._num_units_per_block * 2, self._num_units_per_block * 2]
                    )
                    b_ru = tf.get_variable(
                        "b_ru", [self._num_units_per_block * 2],
                        initializer=init_ops.constant_initializer(1.0))
                    w_c = tf.get_variable("w_c",
                        [self._num_units_per_block * 2, self._num_units_per_block]
                    )
                    b_c = tf.get_variable(
                        "b_c", [self._num_units_per_block],
                        initializer=init_ops.constant_initializer(0.0))
                    _gru_block_cell = gen_gru_ops.gru_block_cell  # pylint: disable=invalid-name
                    _, _, _, new_a = _gru_block_cell(
                        x=candidate_j, h_prev=state_a[j], 
                        w_ru=w_ru, w_c=w_c, b_ru=b_ru, b_c=b_c)
                    
                    v_a = tf.get_variable(
                        "v_a", [self._num_units_per_block],
                        initializer=self._initializer,
                    )
                
                next_a_states.append(new_a)

                gate_j = self.get_gate(state_j, key_j, inputs, v_a, new_a)

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # Equation 5: h_j <- h_j / \norm{h_j}
                # Forget previous memories by normalization.
                state_j_next_norm = tf.norm(
                    tensor=state_j_next,
                    ord='euclidean',
                    axis=-1,
                    keep_dims=True)
                state_j_next_norm = tf.where(
                    tf.greater(state_j_next_norm, 0.0),
                    state_j_next_norm,
                    tf.ones_like(state_j_next_norm))
                state_j_next = state_j_next / state_j_next_norm

                next_states.append(state_j_next)
            state_next = tf.concat(next_states, axis=1)
            state_a_next = tf.concat(next_a_states, axis=1)
            return state_next, tf.concat(values=[state_next, state_a_next], axis=1)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with name_scope(values=[t], name=name, default_name="zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(
            axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name
        )

def prelu(features, alpha, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU'):
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg


class Delayed_EntNet_Sentihood(object):
    def __init__(self, 
        batch_size, vocab_size, target_len, aspect_len, sentence_len, 
        answer_size, embedding_size,
        weight_tying="adj",
        hops=3,
        embedding_mat=None,
        update_embeddings=False,
        softmax_mask=True,
        max_grad_norm=5.0,
        n_keys=6,
        tied_keys=[],
        l2_final_layer=0.0,
        initializer=tf.contrib.layers.xavier_initializer(),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        global_step=None,
        session=None,
        name='Delayed_EntNet_Sentihood'):

        print name

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._target_len = target_len
        self._aspect_len = aspect_len
        self._sentence_len = sentence_len
        self._embedding_size = embedding_size
        self._answer_size = answer_size
        self._max_grad_norm = max_grad_norm
        self._init = initializer
        self._opt = optimizer
        self._global_step = global_step
        self._name = name
        self._embedding_mat = embedding_mat
        self._update_embeddings = update_embeddings

        assert len(tied_keys) <= n_keys
        self._n_keys = n_keys
        self._tied_keys = tied_keys
        self._l2_final_layer = l2_final_layer

        self._build_inputs()
        self._build_vars()

        logits = self._inference_adj(
            self._sentences, 
            self._targets,
            self._aspects,
            self._entnet_input_keep_prob,
            self._entnet_output_keep_prob,
            self._entnet_state_keep_prob,
            self._final_layer_keep_prob,
        )
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(self._answers_one_hot, tf.float32), 
            name="cross_entropy"
        )
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name="cross_entropy_mean"
        )

        # l2 regularization
        trainable_variables = tf.trainable_variables()
        l2_loss_final_layer = 0.0
        assert self._l2_final_layer >= 0

        if self._l2_final_layer > 0:
            final_layer_weights = [ tf.nn.l2_loss(v) for v in trainable_variables
                                    if 'R:0' in v.name]
            assert len(final_layer_weights) == 1
            l2_loss_final_layer = self._l2_final_layer * tf.add_n(final_layer_weights)

        # loss op
        loss_op = cross_entropy_mean + l2_loss_final_layer

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)

        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, global_step=self._global_step, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op, feed_dict={self._input_embedding: self._embedding_mat})

    def _build_inputs(self):
        self._sentences = tf.placeholder(
            tf.int32, [None, self._sentence_len], 
            name="sentences"
        )
        self._targets = tf.placeholder(
            tf.int32, [None, self._target_len],
            name="targets"
        )
        self._aspects = tf.placeholder(
            tf.int32, [None, self._aspect_len],
            name="aspects"
        )
        self._answers = tf.placeholder(
            tf.int32, [None], 
            name="answers"
        )
        self._answers_one_hot = tf.one_hot(
            indices=self._answers,
            depth=self._answer_size,
        )
        self._input_embedding = tf.placeholder(
            tf.float32, shape=self._embedding_mat.shape,
            name="input_embedding"
        )
        self._entnet_input_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="entnet_input_keep_prob"
        )
        self._entnet_output_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="entnet_output_keep_prob"
        )
        self._entnet_state_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="entnet_state_keep_prob"
        )
        self._final_layer_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="final_layer_keep_prob"
        )

    def _build_vars(self):
        with tf.variable_scope(self._name):
            self._embedding = tf.get_variable(
                name="embedding",
                dtype=tf.float32,
                initializer=self._input_embedding,
                trainable=self._update_embeddings,
            )

            self._free_keys_embedding = tf.get_variable(
                name="free_keys_embedding",
                dtype=tf.float32,
                shape=[self._n_keys - len(self._tied_keys), self._embedding_size],
                initializer=self._init,
                trainable=True,
            )

        self._nil_vars = set([self._embedding.name])

    def _mask_embedding(self, embedding):
        vocab_size, embedding_size = self._embedding_mat.shape
        embedding_mask = tf.constant(
            value=[0 if i == 0 else 1 for i in range(vocab_size)],
            shape=[vocab_size, 1],
            dtype=tf.float32,
            name="embedding_mask",
        )
        return embedding * embedding_mask

    def _inference_adj(self, sentences, targets, aspects, 
                       entnet_input_keep_prob, entnet_output_keep_prob, 
                       entnet_state_keep_prob, final_layer_keep_prob):
        with tf.variable_scope(self._name):
            masked_embedding = self._mask_embedding(self._embedding)

            batch_size = tf.shape(sentences)[0]
            
            targets_emb = tf.nn.embedding_lookup(masked_embedding, targets)
            # [None, entity_size, emb_size]
            targets_emb = tf.reduce_mean(
                input_tensor=targets_emb,
                axis=1,
                keep_dims=True,
            )
            # [None, 1, emb_size]
            aspects_emb = tf.nn.embedding_lookup(masked_embedding, aspects)
            # [None, aspect_size, emb_size]
            aspects_emb = tf.reduce_mean(
                input_tensor=aspects_emb,
                axis=1,
                keep_dims=True,
            )
            # [None, 1, emb_size]

            sentences_emb = tf.nn.embedding_lookup(masked_embedding, sentences)
            # [None, memory_size, emb_size]

            sentences_len = self._sentence_length(sentences_emb)
            # [None]

            tied_keys_emb = tf.nn.embedding_lookup(masked_embedding, self._tied_keys)
            # [len(self._tied_keys), max_key_len, emb_size]
            tied_keys_emb = tf.reduce_mean(
                input_tensor=tied_keys_emb,
                axis=1,
            )
            # [len(self._tied_keys), emb_size]
            free_keys_emb = self._free_keys_embedding
            # [n_keys - len(self._tied_keys), emb_size]

            keys_emb = tf.concat(
                values=[tied_keys_emb, free_keys_emb],
                axis=0,
                name="keys_emb",
            )
            # [n_keys, emb_size]

            batched_keys_emb = tf.tile(
                input=tf.expand_dims(input=keys_emb, axis=0),
                multiples=[batch_size, 1, 1]
            )
            # [None, n_keys, emb_size]

            keys = tf.split(keys_emb, self._n_keys, axis=0)
            # list of [1, emb_size]
            keys = [tf.squeeze(key, axis=0) for key in keys]
            # list of [emb_size]

            alpha = tf.get_variable(
                name='alpha',
                shape=self._embedding_size,
                initializer=tf.constant_initializer(1.0)
            )
            activation = partial(prelu, alpha=alpha)

            cell_fw = DynamicMemoryCell(
                num_blocks=self._n_keys,
                num_units_per_block=self._embedding_size,
                keys=keys,
                initializer=self._init,
                recurrent_initializer=self._init,
                activation=activation,
            )
            initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            sentences_emb_shape = sentences_emb.get_shape()
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell=cell_fw,
                input_keep_prob=entnet_input_keep_prob,
                output_keep_prob=entnet_output_keep_prob,
                state_keep_prob=entnet_state_keep_prob,
                variational_recurrent=True,
                input_size=(sentences_emb_shape[2]),
                dtype=tf.float32,
            )

            cell_bw = DynamicMemoryCell(
                num_blocks=self._n_keys,
                num_units_per_block=self._embedding_size,
                keys=keys,
                initializer=self._init,
                recurrent_initializer=self._init,
                activation=activation,
            )
            initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell=cell_bw,
                input_keep_prob=entnet_input_keep_prob,
                output_keep_prob=entnet_output_keep_prob,
                state_keep_prob=entnet_state_keep_prob,
                variational_recurrent=True,
                input_size=(sentences_emb_shape[2]),
                dtype=tf.float32,
            )
            (_, _), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=sentences_emb,
                sequence_length=sentences_len,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
            )

            last_state_fw, _ = tf.split(
                value=last_state_fw,
                num_or_size_splits=[
                    self._n_keys * self._embedding_size, 
                    self._n_keys * self._embedding_size,
                ],
                axis=1
            )
            last_state_bw, _ = tf.split(
                value=last_state_bw,
                num_or_size_splits=[
                    self._n_keys * self._embedding_size, 
                    self._n_keys * self._embedding_size,
                ],
                axis=1
            )
            # last_state_f/bw: [None, emb_size * n_keys]

            last_state_fw = tf.stack(
                tf.split(last_state_fw, self._n_keys, axis=1), axis=1)
            # [None, n_keys, emb_size]
            last_state_bw = tf.stack(
                tf.split(last_state_bw, self._n_keys, axis=1), axis=1)
            # [None, n_keys, emb_size]

            last_state = last_state_fw + last_state_bw
            # [None, n_keys, emb_size]
            
            asp_att = tf.concat(values=[targets_emb, aspects_emb], axis=2)
            # [None, 1, emb_size * 2]
            W_asp_att = tf.get_variable(
                name='W_asp_att',
                shape=[self._embedding_size, self._embedding_size * 2],
                dtype=tf.float32,
                initializer=self._init,
            )
            temp = tf.tensordot(
                batched_keys_emb, W_asp_att, [[2], [0]]
            )
            # [None, n_keys, emb_size * 2]
            attention = tf.reduce_sum(temp * asp_att, axis=2)
            # [None, n_keys]
            attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
            # [None, 1]
            attention = tf.nn.softmax(attention - attention_max)
            # [None, n_keys]
            attention = tf.expand_dims(attention, axis=2)
            # [None, n_keys, 1]

            u = tf.reduce_sum(last_state * attention, axis=1)
            # [None, emb_size]
            
            R = tf.get_variable('R', [self._embedding_size, self._answer_size])
            H = tf.get_variable('H', [self._embedding_size, self._embedding_size])

            a = tf.squeeze(aspects_emb, axis=1)
            # [None, emb_size]
            hidden = activation(a + tf.matmul(u, H))
            # [None, emb)size]
            hidden = tf.nn.dropout(x=hidden, keep_prob=final_layer_keep_prob)
            # [None, emb_size]
            y = tf.matmul(hidden, R)
            # [None, 1]

            return y

    def _get_mini_batch_start_end(self, n_train, batch_size=None):
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

    def fit(self, sentences, targets, aspects, answers, entnet_input_keep_prob, 
            entnet_output_keep_prob, entnet_state_keep_prob, 
            final_layer_keep_prob, batch_size=None):
        assert len(sentences) == len(targets)
        assert len(sentences) == len(aspects)
        assert len(sentences) == len(answers)
        batches = self._get_mini_batch_start_end(len(sentences), batch_size)
        total_loss = 0.
        for start, end in batches:
            feed_dict = {
                self._sentences: sentences[start:end], 
                self._targets: targets[start:end],
                self._aspects: aspects[start:end],
                self._answers: answers[start:end], 
                self._entnet_input_keep_prob: entnet_input_keep_prob,
                self._entnet_output_keep_prob: entnet_output_keep_prob,
                self._entnet_state_keep_prob: entnet_state_keep_prob,
                self._final_layer_keep_prob: final_layer_keep_prob,
            }
            loss, _ = self._sess.run(
                [self.loss_op, self.train_op], 
                feed_dict=feed_dict
            )
            total_loss = loss * len(sentences[start:end])
        return total_loss

    def predict(self, sentences, targets, aspects, batch_size=None):
        assert len(sentences) == len(targets)
        assert len(sentences) == len(aspects)
        batches = self._get_mini_batch_start_end(len(sentences), batch_size)
        predictions, predictions_prob = [], []
        for start, end in batches:
            feed_dict = {
                self._sentences: sentences[start:end], 
                self._targets: targets[start:end],
                self._aspects: aspects[start:end],
                self._entnet_input_keep_prob: 1.0,
                self._entnet_output_keep_prob: 1.0,
                self._entnet_state_keep_prob: 1.0,
                self._final_layer_keep_prob: 1.0,
            }
            prediction, prediction_prob = self._sess.run(
                [self.predict_op, self.predict_proba_op],
                feed_dict=feed_dict
            )
            predictions.extend(prediction)
            predictions_prob.extend(prediction_prob)
        return predictions, np.array(predictions_prob)

    def _sentence_length(self, sentences):
        '''
        sentences: (None, sentence_len, embedding_size)
        '''
        used = tf.sign(tf.reduce_max(tf.abs(sentences), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
