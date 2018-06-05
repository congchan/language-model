import os
import warnings

from mxnet.gluon.model_zoo.model_store import get_model_file
from mxnet import init, nd, cpu, autograd
from mxnet.gluon import nn, Block
from mxnet.gluon.model_zoo import model_store
from gluonnlp.model.utils import apply_weight_drop, _get_rnn_layer


class AWDRNN(Block):
    """AWD language model by salesforce.

    Reference: https://github.com/salesforce/awd-lstm-lm

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
    def __init__(self, mode, vocab_size, embed_size, hidden_size, num_layers,
                 tie_weights=False, dropout=0.5, weight_drop=0, drop_h=0.5, drop_i=0.5, drop_e=0,
                 **kwargs):
        super(AWDRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
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
                                           l != self._num_layers - 1 or not self._tie_weights
                                           else self._embed_size, 0, self._weight_drop))
        return encoder

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
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder)-1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        with autograd.predict_mode():
            out = self.decoder(encoded)
        return out, out_states
