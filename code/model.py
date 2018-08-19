import os
import warnings
import mxnet

from mxnet import init, nd, cpu, autograd, gluon
from mxnet.gluon import nn, Block, rnn
from gluonnlp.model.utils import apply_weight_drop, _get_rnn_layer



class MOSRNN(Block):
    """Mos language model by Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen.

    Reference: https://github.com/zihangdai/mos

    License: MIT License

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    hidden_size_last : int
        Number of last hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    dropout : float
        Dropout rate to use for encoder output.
    weight_drop : float
        Dropout rate to use on encoder h2h weights.
    drop_h : float
        Dropout rate to on the output of intermediate layers of encoder.
    drop_i : float
        Dropout rate to on the output of embedding.
    drop_e : float
        Dropout rate to use on the embedding layer.
    drop_l : float
        Dropout rate to use on the latent layer.
    num_experts : int
        Number of softmax.
    """
    def __init__(self, mode, vocab_size, embed_size=280, hidden_size=960, hidden_size_last=620,
                 num_layers=3, tie_weights=False, dropout=0.2, weight_drop=0.5, drop_h=0.3,
                 drop_i=0.55, drop_e=0.1, drop_l=0.3, num_experts=15, **kwargs):
        super(MOSRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._hidden_size_last = hidden_size_last
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._drop_l = drop_l
        self._num_experts = num_experts
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.prior = self._get_prior()
            self.latent = self._get_latent()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1))
            if self._drop_e:
                apply_weight_drop(embedding_block, 'weight', self._drop_e, axes=(1,))
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                                           self._hidden_size, self._hidden_size if
                                           l != self._num_layers - 1 else self._hidden_size_last,
                                           0, self._weight_drop))
        return encoder

    def _get_prior(self):
        # num_experts as output size, in_units will be inferred as last hid size
        prior = nn.HybridSequential()
        with prior.name_scope():
            prior.add(nn.Dense(self._num_experts, in_units=self._hidden_size_last,
                               use_bias=False, flatten=False))
        return prior

    def _get_latent(self):
        latent = nn.HybridSequential()
        with latent.name_scope():
            latent.add(nn.Dense(self._num_experts * self._embed_size, 'tanh',
                                in_units=self._hidden_size_last, flatten=False))
        return latent

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def forward(self, inputs, begin_state=None):
        """Implement forward computation.

        Parameters
        ----------
        inputs : NDArray
            The training dataset.
        begin_state : list
            The initial hidden states.

        Returns
        -------
        out: NDArray
            The output of the model.
        out_states: list
            The list of output states of the model's encoder.
        """
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        out_states = []
        encoded_raw = []
        encoded_dropped = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder)-1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        states = out_states
        encoded_dropped.append(encoded)

        latent = nd.Dropout(self.latent(encoded), p=self._drop_l, axes=(0,))
        logit = self.decoder(latent.reshape(-1, self._embed_size))
        prior_logit = self.prior(encoded).reshape(-1, self._num_experts)
        prior = nd.softmax(prior_logit)

        prob = nd.softmax(logit.reshape(-1, self._vocab_size))
        prob = prob.reshape(-1, self._num_experts, self._vocab_size)
        prob = (prob * prior.expand_dims(2).broadcast_to(prob.shape)).sum(axis=1)
        out = nd.log(nd.add(prob, 1e-8)).reshape(-1, inputs.shape[1], self._vocab_size)

        return out, out_states, encoded_raw, encoded_dropped


class WDRNN(Block):
    """AWD language model (w/o asgd). Support (emb_size != last_hid_size), but with an extra dense layer
    Reference: https://github.com/salesforce/awd-lstm-lm by salesforce.
    License: BSD 3-Clause
    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    dropout : float
        Dropout rate to use for encoder output.
    weight_drop : float
        Dropout rate to use on encoder h2h weights.
    drop_h : float
        Dropout rate to on the output of intermediate layers of encoder.
    drop_i : float
        Dropout rate to on the output of embedding.
    drop_e : float
        Dropout rate to use on the embedding layer.
    """
    def __init__(self, mode, vocab_size, embed_size=400, hidden_size=1150, hidden_size_last=620, num_layers=3,
                 tie_weights=True, dropout=0.4, weight_drop=0.5, drop_h=0.2,
                 drop_i=0.65, drop_e=0.1, **kwargs):
        super(WDRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._hidden_size_last = hidden_size_last
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1))
            if self._drop_e:
                apply_weight_drop(embedding_block, 'weight', self._drop_e, axes=(1,))
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                                           self._hidden_size, self._hidden_size if
                                           l != self._num_layers - 1 else self._hidden_size_last,
                                           0, self._weight_drop))
        return encoder

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                if self._hidden_size_last != self._embed_size:
                    output.add(nn.Dense(self._embed_size, 'tanh', flatten=False))
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def state_info(self, *args, **kwargs):
        return [c.state_info(*args, **kwargs) for c in self.encoder]

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Implement the forward computation that the awd language model and cache model use.
        Parameters
        -----------
        inputs : NDArray
                input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`
        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        encoded_raw: list
            The list of outputs of the model's encoder with length equals to num_layers.
            the shape of every encoder's output `(sequence_length, batch_size, num_hidden)`
        encoded_dropped: list
            The list of outputs with dropout of the model's encoder with length equals
            to num_layers. The shape of every encoder's dropped output
            `(sequence_length, batch_size, num_hidden)`
        """
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        out_states = []
        encoded_raw = []
        encoded_dropped = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder) - 1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        encoded_dropped.append(encoded)
        with autograd.predict_mode():
            out = self.decoder(encoded)

        return out, out_states, encoded_raw, encoded_dropped

class RNN(gluon.Block):
    ''' Vanilla multi-layers RNN with an extra dense layer, modified from
    https://zh.gluon.ai/chapter_recurrent-neural-networks/rnn-gluon.html'''

    def __init__(self, mode, vocab_size, embed_size=400, hidden_size=1150, hidden_size_last=280, num_layers=3,
                 tie_weights=True, dropout=0.4, weight_drop=0, drop_h=0,
                 drop_i=0, drop_e=0, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._hidden_size_last = hidden_size_last
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()


    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1))
            if self._drop_e:
                apply_weight_drop(embedding_block, 'weight', self._drop_e, axes=(1,))
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                                           self._hidden_size, self._hidden_size if
                                           l != self._num_layers - 1 else self._hidden_size_last,
                                           0, self._weight_drop))
        return encoder

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                if self._hidden_size_last != self._embed_size:
                    output.add(nn.Dense(self._embed_size, 'tanh', flatten=False))
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def state_info(self, *args, **kwargs):
        return [c.state_info(*args, **kwargs) for c in self.encoder]

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Implement the forward computation that the awd language model and cache model use.
        Parameters
        -----------
        inputs : NDArray
                input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`
        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        encoded_raw: list
            The list of outputs of the model's encoder with length equals to num_layers.
            the shape of every encoder's output `(sequence_length, batch_size, num_hidden)`
        encoded_dropped: list
            The list of outputs with dropout of the model's encoder with length equals
            to num_layers. The shape of every encoder's dropped output
            `(sequence_length, batch_size, num_hidden)`
        """
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        out_states = []
        encoded_raw = []
        encoded_dropped = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder) - 1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        encoded_dropped.append(encoded)
        with autograd.predict_mode():
            out = self.decoder(encoded)

        return out, out_states, encoded_raw, encoded_dropped
