import math, time, argparse, os, sys, logging, gluonnlp, mxnet
import numpy as np
import data, model, utils
import gluon_utils as gu
from mxnet import gluon, nd, init, autograd
from utils import batchify, get_batch, detach, create_exp_dir, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='PennTreeBank/WikiText2 RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='penn',
                        help='which data corpus')
    parser.add_argument('--model', type=str, default='Mos',
                        help='Model, options (MOS, StandardRNN, AWDRNN)')
    parser.add_argument('--exprm', type=str, default='',
                        help='experiment suffix')
    parser.add_argument('--rnn_cell', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
    parser.add_argument('--emb_size', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--hid_size', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--last_hid_size', type=int, default=-1,
                        help='number of hidden units for the last rnn layer')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--drop_h', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--drop_i', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--drop_e', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--drop_l', type=float, default=-0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--w_drop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--tied', action='store_false',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='Experiments',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue train from a checkpoint')
    parser.add_argument('--n_experts', type=int, default=10,
                        help='number of experts')
    parser.add_argument('--small_batch_size', type=int, default=-1,
                        help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                         In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                         until batch_size is reached. An update step is then performed.')
    parser.add_argument('--max_seq_len_delta', type=int, default=40,
                        help='max sequence length')
    parser.add_argument('--single_gpu', default=False, action='store_true',
                        help='use single GPU')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    if not args.continue_train:
        path = utils.make_dir([args.save, args.model+'-'+args.rnn_cell+args.exprm])

    logging.basicConfig(level=logging.INFO,
                        handlers = [
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(path, "log.log"))
                        ])

    ctxs = gu.try_all_gpus()
    logging.info('Computation on: {}'.format(ctxs))

    if args.last_hid_size < 0:
        args.last_hid_size = args.emb_size
    if args.drop_l < 0:
        args.drop_l = args.drop_h
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size



    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    mxnet.random.seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size).as_in_context(ctxs)
    val_data = batchify(corpus.valid, eval_batch_size).as_in_context(ctxs)
    test_data = batchify(corpus.test, test_batch_size).as_in_context(ctxs)

    ###############################################################################
    # Build the model
    ###############################################################################
    '''Model parameters aved path: /train/experiment/model.params.
    If parameters exists, load the saved parameters, else initialize'''

    ntokens = len(corpus.dictionary)

    params = os.path.join(path, 'model.params')

    if args.model == 'MOS':
        model = model.MOS(args.rnn_cell, ntokens, args.emb_size, args.hid_size, args.last_hid_size, args.nlayers,
                           args.dropout, args.drop_h, args.drop_i, args.drop_e, args.w_drop,
                           args.tied, args.drop_l, args.n_experts)
    elif args.model == 'AWDRNN':
        model = model.AWDRNN(args.rnn_cell, ntokens, args.emb_size, args.hid_size, args.n_layers,
                     tie_weights=args.tied, dropout=args.dropout, weight_drop=args.w_drop,
                     drop_h=args.drop_h, drop_i=args.drop_i, drop_e=args.drop_e)
    elif args.model == 'StandardRNN':
        model = gluonnlp.model.StandardRNN(args.rnn_cell, ntokens, args.emb_size, args.hid_size,
                                        args.n_layers, dropout=args.dropout, tie_weights=args.tied)


    if os.path.exists(params) and os.path.isfile(params) and os.path.getsize(params)>0:
        model.load_params(params, ctx=ctxs)
        logging.info("Loaded parameters from : {}".format(params))
    else:
        model.initialize(init.Xavier(), ctx=ctxs)

    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    train()
    #total_params = sum(x.data.nelement() for x in model.parameters())
    #logging('Args: {}'.format(args))
    #logging('Model total parameters: {}'.format(total_params))


def evaluate(data_source, batch_size=10):
    '''https://mxnet.incubator.apache.org/api/python/autograd/autograd.html#train-mode-and-predict-mode'''
    cost_sum = nd.array([0], ctx=ctxs)
    n = 0
    state = model.begin_state(func=nd.zeros, batch_size=batch_size, ctx=ctxs)
    for i in range(0, data_source.shape[0] - 1, args.bptt):
        X, Y = get_batch(data_source, i)
        output, state = model(X, state)
        # cost tensor with shape (batch_size,).
        # Dimenions other than batch_axis are averaged out.
        cost = loss(output, Y)
        cost_sum += cost.sum()
        n += cost.size
    return cost_sum / n

def train():
    '''If gluon trainer recognizes multi-devices,
    it will automatically aggregate the gradients and synchronize the parameters.'''

    assert (len(ctxs) == num_gpus or num_gpus == 0), '# of GPUs not matched!'

    model.initialize(init=init.Normal(sigma=0.01), ctx=ctxs, force_reinit=True)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr})

    # Loop over epochs.
    bess_loss = float("Inf")
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            tic = time.time()
            epoch_train()
            val_loss = evaluate(val_data, eval_batch_size)

            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - tic),
                                               val_loss, math.exp(val_loss)))
            logging('-' * 89)

            if val_loss < bess_loss:
                save_checkpoint(model, trainer, path)
                logging('Saving Normal!')
                bess_loss = val_loss

    except KeyboardInterrupt:
        logging('-' * 89)
        logging('Exiting from training early')

def epoch_train():
    ''' Train all the batches within one epoch'''

    total_loss = nd.array([0], ctx=ctxs)
    costs = [nd.array([0], ctx=ctx) for ctx in ctxs]
    m = batch_size // len(ctxs)
    state = [model.begin_state(batch_size=m, ctx=ctx) for ctx in ctxs]
    # state = nd.zeros(shape=(args.batch_size, args.hid_size), ctx=ctxs) # （bsz, hidden_size)

    ############################################################################
    # Loop all batches
    batch, i = 0, 0
    while i < train_data.shape(0) - 1 - 1:
        #######################################################################
        # Control seq_len cited from origin paper
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Normal distribution (mean, variance): Prevent extreme sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5))) #
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        ########################################################################

        # Schedual learning rate
        trainer.set_learning_rate(trainer.learning_rate * seq_len / args.bptt)

        '''Each batch shape(seq_len, batch_size), split data to each device.
        m is the # of samples for each device, devided along batch_size axis.'''
        Xs, Ys = get_batch(train_data, i, args, seq_len=seq_len)
        assert args.batch_size == Xs.shape[1], 'data shape[1] should be batch_size'
        Xs = gluon.utils.split_and_load(Xs, ctxs, 1)
        Ys = gluon.utils.split_and_load(Ys, ctxs, 1)
        tic_b = time.time()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # state = detach(state)

        for i, X in enumerate(Xs):
            with autograd.record(): # train_mode
                 output, state[i] = model(X, state[i].detach()) # state（n_layers, bsz, hidden_size)
                 costs[i]= loss(output, Ys[i]).sum() / (seq_len * m)  # loss (seq_len * m,)

        for c in costs:
            c.backward()

        # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
        # 因此我们将 clipping_theta 乘以 seq_len 和 batch_size。
        grads = [p.grad(ctx) for ctx in ctxs for p in model.collect_params().values()]
        gluon.utils.clip_global_norm(grads, args.clipping_theta * seq_len * batch_size)
        trainer.step(batch_size)

        total_loss += sum(costs).asscalar() # this will synchronize all GPUs
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, trainer.learning_rate,
                (time.time() - tic_b) * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            tic_b = time.time()

        batch += 1
        i += seq_len

    nd.waitall()
    ############################################################################
