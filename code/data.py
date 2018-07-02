import os
from mxnet import nd
import numpy as np
import mxnet as mx
from collections import Counter

class Config(object):
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, data, debug=0, predict_only=False):
        self.dictionary = Dictionary()
        path = os.path.join('data', data)
        self.debug = debug
        self.train = [] if predict_only else self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = [] if predict_only else self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = [] if debug else self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path, ctx=None):
        """ Tokenizes a text file into a list of indexes of tokens."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = nd.empty(self.debug if self.debug else tokens , ctx, dtype=np.int64)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                    if self.debug and token >= self.debug:
                        return ids

        return ids

class SentCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, ctx=None):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ['<eos>']
                sent = nd.empty(len(words), ctx, dtype=np.int64)
                for i, word in enumerate(words):
                    sent[i] = self.dictionary.word2idx[word]
                sents.append(sent)

        return sents

class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False, ctx=None):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id
        self.ctx = ctx

    def __next__(self):
        if self.idx >= len(self.sort_sents):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents)-self.idx)
        batch = self.sort_sents[self.idx:self.idx+batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = mx.nd.full((max_len, batch_size), self.pad_id, self.ctx, dtype=np.int64)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0),i].copy_(s)
        if self.cuda:
            tensor = tensor.cuda()

        self.idx += batch_size

        return tensor

    next = __next__

    def __iter__(self):
        self.idx = 0
        return self

if __name__ == '__main__':
    corpus = SentCorpus('../penn')
    loader = BatchSentLoader(corpus.test, 10)
    for i, d in enumerate(loader):
        print(i, d.size())
