import os
import warnings
import mxnet

from mxnet import init, nd, cpu, autograd, gluon
from mxnet.gluon import nn, Block, rnn
from gluonnlp.model.utils import apply_weight_drop, _get_rnn_layer

class RNN(gluon.Block):
    ''' Vanilla RNN, modified version from
    https://zh.gluon.ai/chapter_recurrent-neural-networks/rnn-gluon.html'''

    def __init__(self, vocab_size, args, **kwargs):
        super(RNN, self).__init__(**kwargs)

        with self.name_scope():
            self.dropout = nn.Dropout(args.dropout)
            self.embedding = nn.HybridSequential()
            self.embedding.add(nn.Embedding(vocab_size, args.emb_size,
                                    weight_initializer = mxnet.init.Uniform(0.1)))

            self.decoder = nn.HybridSequential()
            if args.tied:
                ''' Optionally tie weights as in:
                "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                https://arxiv.org/abs/1608.05859
                and
                "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                https://arxiv.org/abs/1611.01462
                When using the tied weight, hid_size must be equal to emb_size
                '''
                # output tensor (sequence_length, batch_size, num_hidden) when layout is “TNC”
                self.encoder = rnn.LSTM(args.hid_size, args.n_layers)

                self.in_units = args.emb_size
                self.decoder.add(nn.Dense(vocab_size, flatten=False,
                                            in_units=self.in_units,
                                            params=self.embedding[0].params))
            else:
                self.encoder = rnn.LSTM(args.emb_size, args.n_layers)
                self.in_units = args.last_hid_size
                self.decoder.add(nn.Dense(vocab_size, flatten=False,
                                            in_units=self.in_units))


    def forward(self, inputs, state):
        embedding = self.dropout(self.embedding(inputs))
        output, state = self.encoder(embedding, state)
        output = self.dropout(output)
        # since flatten=False require input data shape (..., in_units).
        output = self.decoder(output)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)



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
    n_layers : int
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
    def __init__(self, mode, vocab_size, embed_size, hidden_size, hidden_size_last, n_layers,
                 tie_weights=False, dropout=0.5, weight_drop=0, drop_h=0.5, drop_i=0.5, drop_e=0.1,
                 l_dropout=0.5, n_experts=10, **kwargs):
        super(MOSRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._hidden_size_last = hidden_size_last
        self._n_layers = n_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._l_dropout = l_dropout
        self._dropout_l = l_dropout
        self._n_experts = n_experts
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.prior = nn.Dense(n_experts, use_bias=False, flatten=False) # n_experts as output size, in_units will be inferred as last hid size
            self.latent = nn.Dense(n_experts * embed_size, 'tanh', flatten=False)
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
            for l in range(self._n_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                                           self._hidden_size, self._hidden_size if
                                           l != self._n_layers - 1 else self._hidden_size_last,
                                           0, self._weight_drop))
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

    def forward(self, inputs, begin_state=None, return_h=False, return_prob=False):
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
        # if not begin_state:
        #     begin_state = self.begin_state(batch_size=inputs.shape[0])
        out_states = []
        raw_encodeds = []
        encodeds = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            raw_encodeds.append(encoded)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder)-1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
                encodeds.append(encoded)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        states = out_states
        encodeds.append(encoded)

        latent = nd.Dropout(self.latent(encoded), p=self._dropout_l, axes=(0,))
        logit = self.decoder(latent.reshape(-1, self._embed_size))
        prior_logit = self.prior(encoded).reshape(-1, self._n_experts)
        prior = nd.softmax(prior_logit)
        prob = nd.softmax(logit.reshape(-1, self._vocab_size)).reshape(-1, self._n_experts, self._vocab_size)
        prob = (prob * prior.expand_dims(2).broadcast_to(prob.shape)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = nd.log(nd.add(prob, 1e-8))
            model_output = log_prob

        model_output = model_output.reshape(inputs.shape[0], -1, self._vocab_size)

        if return_h:
            return model_output, out_states, raw_encodeds, encodeds
        return model_output, out_states
