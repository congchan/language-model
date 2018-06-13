# -*- coding: utf-8 -*-
import math, time, argparse, os, sys, logging, gluonnlp, mxnet
import numpy as np
import data, model, utils
from mxnet import gluon, nd, init, autograd
from utils import batchify, get_batch, detach, create_exp_dir, save_checkpoint

def configuration():
    '''Setting configuration'''
    parser = argparse.ArgumentParser(description='PennTreeBank/WikiText2 RNN/LSTM Language Model')
    parser.add_argument('--continue_exprm', type=str, default=None,
                        help='continue experiment from a checkpoint')
    parser.add_argument('--data', type=str, default='penn',
                        help='which data corpus: penn, wikitext-2')
    parser.add_argument('--model', type=str, default='Mos',
                        help='Model, options (RNN, MOS, StandardRNN, AWDRNN)')
    parser.add_argument('--exprm', type=str, default='',
                        help='experiment suffix')
    parser.add_argument('--rnn_cell', type=str, default='lstm',
                        help='type of recurrent net (lstm, gru)')
    parser.add_argument('--emb_size', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--hid_size', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--last_hid_size', type=int, default=-1,
                        help='number of hidden units for the last rnn layer\
                        by default equal to hid_size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=3,
                        help='initial learning rate')
    parser.add_argument('--clipping_theta', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--debug', type=int, default=0,
                        help='debug mode sepcify the tokenize length for faster debugging')
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
    parser.add_argument('--predict_only', action='store_true',
                        help='predict only, default not')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only, default not')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='Experiments',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer in trainer: SGD, Adam, RMSProp, ... ' )
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
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

def predict():
    ''' Evalute on test data'''
    test_loss = evaluate(test_data, test_batch_size)
    logging.info('-' * 89)
    logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logging.info('-' * 89)

def evaluate(data_source, batch_size=10):
    '''https://mxnet.incubator.apache.org/api/python/autograd/autograd.html#train-mode-and-predict-mode'''
    cost_sum = 0
    n = 0
    state = model.begin_state(func=nd.zeros, batch_size=batch_size)
    for i in range(0, data_source.shape[0] - 1, args.bptt):
        X, Y = get_batch(data_source, i, args)
        output, state = model(X, state)
        # cost tensor with shape (batch_size,).
        # Dimenions other than batch_axis are averaged out.
        cost = loss(output, Y)
        cost_sum += cost.sum().asscalar()
        n += cost.size
        state = detach(state)

    return cost_sum / n

def train():
    '''If gluon trainer recognizes multi-devices,
    it will automatically aggregate the gradients and synchronize the parameters.'''

    costs_container = [nd.array([0], ctx=ctx) for ctx in ctxs]

    logging.info('-' * 40 + "Begin training" + '-' * 40)
    # Loop over epochs.
    best_loss = float("Inf")
    for epoch in range(args.epochs):
        tic = time.time()
        train_one_epoch(epoch, costs_container)
        val_loss = evaluate(val_data, eval_batch_size)
        toc = time.time()
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | '
                'valid ppl {:8.2f}'.format(epoch, toc - tic, val_loss, math.exp(val_loss)))
        epoch_info.append([epoch, toc - tic, val_loss, math.exp(val_loss) ])
        utils.save_info(epoch_info, epoch_file)
        logging.info('-' * 89)

        if val_loss < best_loss:
            save_checkpoint(model, trainer, path)
            logging.info('Performance improving, saving Model')
            best_loss = val_loss



def train_one_epoch(epoch, costs):
    ''' Train all the batches within one epoch.
    costs is the container created once and reuse for efficiency'''

    total_loss = 0
    states = [model.begin_state(batch_size=m, ctx=ctx) for ctx in ctxs]

    # Loop all batches
    batch, cursor = 0, 0
    tic_log_interval = time.time()
    while cursor < train_data.shape[0] - 1 - 1:
        #######################################################################
        # Control seq_len cited from origin paper
        random_bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Normal distribution (mean, variance): Prevent extreme sequence lengths
        seq_len = max(5, int(np.random.normal(random_bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        ########################################################################

        # Schedual learning rate
        # trainer.set_learning_rate(trainer.learning_rate * seq_len / args.bptt)

        '''Each batch shape(seq_len, batch_size), split data to each device.
        m is the # of samples for each device, devided along batch_size axis.'''
        Xs, Ys = get_batch(train_data, cursor, args, seq_len=seq_len)
        assert args.batch_size == Xs.shape[1], 'data shape[1] should be batch_size'
        Xs = gluon.utils.split_and_load(Xs, ctxs, 1)
        Ys = gluon.utils.split_and_load(Ys, ctxs, 1)
        tic_b = time.time()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        states = detach(states)

        for i, X in enumerate(Xs):
            with autograd.record(): # train_mode
                 output, states[i] = model(X, states[i]) # state(n_layers, bsz, hidden_size)
                 costs[i]= loss(output, Ys[i]).mean()  # loss (m,)

        for c in costs:
            c.backward()

        # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
        # 因此我们将 clipping_theta 乘以 seq_len 和 batch_size。
        grads = [p.grad(ctx) for ctx in ctxs for p in model.collect_params().values()]
        gluon.utils.clip_global_norm(grads, args.clipping_theta * seq_len *args.batch_size)
        trainer.step(1)

        batch_loss = (sum(costs)/len(costs)).asscalar() # this will synchronize all GPUs

        batch_info.append([epoch, batch, trainer.learning_rate, seq_len,
                      (time.time() - tic_b) * 1000, batch_loss, math.exp(batch_loss) ])

        total_loss += batch_loss

        if batch % args.log_interval == 0 and batch > 0:
            utils.save_info(batch_info, batch_file)

            toc_log_interval = time.time()
            total_loss = total_loss / args.log_interval

            logging.info('| epoch {:3d} ({}/{})%| batch {:3d} | lr {:02.2f} | seq_len {:3d} | ms/batch {:5.2f} | '
                    'loss {:5.3f} | ppl {:5.2f}'.format(
                epoch, cursor, train_data.shape[0], batch, trainer.learning_rate, seq_len,
                (toc_log_interval - tic_log_interval) * 1000 / args.log_interval, total_loss,
                math.exp(total_loss)))

            total_loss = 0
            tic_log_interval = time.time()

        batch += 1
        cursor += seq_len

    nd.waitall()
    ############################################################################

if __name__ == "__main__":
    ###############################################################################
    # prepare configuration, files for saving data and logging information
    ###############################################################################
    args = configuration()

    if args.continue_exprm:
        path = utils.make_dir([args.save, args.continue_exprm])
        configfile = os.path.join(path, 'config.json')

        try: args = data.Config(utils.read_config(configfile,
                                        {'continue_exprm':args.continue_exprm}))
        except FileNotFoundError: raise

    else:
        # By default, use argparse for configuration
        if args.tied: args.hid_size = args.emb_size
        if args.debug: args.log_interval = 2
        path = utils.make_dir([args.save, args.model+'-'+args.rnn_cell+args.exprm])
        args = data.Config(utils.save_config(vars(args), os.path.join(path, 'config.json')))

    logging.basicConfig(level=logging.INFO,
                        handlers = [
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(path, "log.log"))
                        ])

    # set the header of csv logging files
    epoch_info = []
    epoch_file = os.path.join(path, 'epoch_results.csv')
    utils.save_info(['epoch', 'time/s', 'val_loss', 'perplexity'], epoch_file)

    batch_info = []
    batch_file = os.path.join(path, 'batch_results.csv')
    utils.save_info(['epoch', 'batch', 'learning_rate', 'seq_len', 'time/ms', 'val_loss', 'perplexity'], batch_file)

    # config the conputation resources
    if args.cpu:
        ctxs = [mxnet.cpu()]
    else:
        ctxs = utils.try_all_gpus()
    m = args.batch_size // len(ctxs)
    logging.info("Split batch samples (batch size={}) to {}, each device loaded {} samples".format(
                args.batch_size, ctxs, m))

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    mxnet.random.seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data, args.debug)
    logging.info("Load {} train_tokens, {} valid_tokens, {} test_tokens".format(
                    len(corpus.train), len(corpus.valid), len(corpus.test)))

    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size).as_in_context(ctxs[0])
    val_data = batchify(corpus.valid, eval_batch_size).as_in_context(ctxs[0])
    test_data = batchify(corpus.test, test_batch_size).as_in_context(ctxs[0])

    ###############################################################################
    # Build the model
    ###############################################################################
    '''Model parameters aved path: /train/experiment/model.params.
    If parameters exists, load the saved parameters, else initialize'''

    vocab_size = len(corpus.dictionary)

    params = os.path.join(path, 'model.params')
    trainer_states = os.path.join(path, 'trainer.states')

    if args.last_hid_size < 0:
        args.last_hid_size = args.emb_size
    if args.drop_l < 0:
        args.drop_l = args.drop_h
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size

    if args.model == 'MOS':
        model = model.MOS(args.rnn_cell, vocab_size, args.emb_size, args.hid_size, args.last_hid_size, args.nlayers,
                           args.dropout, args.drop_h, args.drop_i, args.drop_e, args.w_drop,
                           args.tied, args.drop_l, args.n_experts)
    elif args.model == 'AWDRNN':
        model = model.AWDRNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size, args.n_layers,
                     tie_weights=args.tied, dropout=args.dropout, weight_drop=args.w_drop,
                     drop_h=args.drop_h, drop_i=args.drop_i, drop_e=args.drop_e)
    elif args.model == 'RNN':
        model = model.RNN(vocab_size, args)
    elif args.model == 'StandardRNN':
        model = gluonnlp.model.StandardRNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size,
                                        args.n_layers, dropout=args.dropout, tie_weights=args.tied)


    loss = gluon.loss.SoftmaxCrossEntropyLoss(batch_axis=1)

    if args.continue_exprm and utils.check_file(params):
        model.load_params(params, ctx=ctxs)
        logging.info("Loading parameters from : {}".format(params))
    else:
        model.initialize(init.Xavier(), ctx=ctxs)

    trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                {'learning_rate': args.lr, 'wd': args.wdecay})

    if args.continue_exprm and utils.check_file(trainer_states):
        trainer.load_states(trainer_states)
        logging.info("Loading training states from : {}".format(trainer_states))

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if not args.predict_only: train()
    except KeyboardInterrupt:
            logging.info('-' * 89)
            logging.info('Exiting from training early')
    finally:
        if not args.debug:
            logging.info('Start evaluation on test_data')
            predict()

    #total_params = sum(x.data.nelement() for x in model.parameters())
    #logging.info('Args: {}'.format(args))
    #logging.info('Model total parameters: {}'.format(total_params))
