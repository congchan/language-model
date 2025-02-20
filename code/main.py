# -*- coding: utf-8 -*-
import math, time, argparse, os, sys, logging, gluonnlp, mxnet
import numpy as np
import data, model, utils
from JointActivationRegularizationLoss import JointActivationRegularizationLoss
from mxnet import gluon, nd, init, autograd
from utils import batchify, get_batch, detach, create_exp_dir

def configuration():
    '''Setting configuration'''
    parser = argparse.ArgumentParser(description='PennTreeBank/WikiText2 RNN/LSTM Language Model')
    parser.add_argument('--continue_exprm', type=str, default=None,
                        help='continue experiment from a checkpoint')
    parser.add_argument('--data', type=str, default='penn',
                        help='which data corpus: penn, wikitext-2')
    parser.add_argument('--model', type=str, default='MOSRNN',
                        help='Model, options (RNN, MOSRNN, StandardRNN, AWDRNN)')
    parser.add_argument('--exprm', type=str, default='',
                        help='experiment suffix')
    parser.add_argument('--rnn_cell', type=str, default='lstm',
                        help='type of recurrent net (lstm, gru)')
    parser.add_argument('--emb_size', type=int, default=280,
                        help='size of word embeddings')
    parser.add_argument('--hid_size', type=int, default=960,
                        help='number of hidden units per layer')
    parser.add_argument('--last_hid_size', type=int, default=620,
                        help='number of hidden units for the last rnn layer\
                        by default equal to hid_size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=3,
                        help='initial learning rate')
    parser.add_argument('--clipping_theta', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--debug', type=int, default=0,
                        help='debug mode sepcify the tokenize length for faster debugging')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--drop_h', type=float, default=0.225,
                        help='dropout for rnn layers (0 = no dropout), context vector V-drop 0.3')
    parser.add_argument('--drop_i', type=float, default=0.4,
                        help='dropout for input embedding layers (0 = no dropout), emb V-drop 0.55')
    parser.add_argument('--drop_e', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout), word level V-drop 0.1')
    parser.add_argument('--drop_l', type=float, default=0.29,
                        help='dropout applied to latent layers (0 = no dropout)')
    parser.add_argument('--w_drop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--tied', action='store_false',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--predict_only', action='store_true',
                        help='predict only, default not')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='Non-monotonically Triggered interval')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only, default not')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='number of GPUs should be no more than the actual request gpus')
    parser.add_argument('--log_interval', type=int, default=0, metavar='N',
                        help='report interval')
    parser.add_argument('--log_freq', type=int, default=5, metavar='N',
                        help='report frequency per epoch')
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
    parser.add_argument('--schedual_rate', type=float, default=0,
                        help='schedual_rate update')
    parser.add_argument('--num_experts', type=int, default=15,
                        help='number of experts')
    parser.add_argument('--small_batch_size', type=int, default=-1,
                        help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                         In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                         until batch_size is reached. An update step is then performed.')
    parser.add_argument('--max_seq_len_delta', type=int, default=40,
                        help='max sequence length')
    parser.add_argument('--single_gpu', default=False, action='store_true',
                        help='use single GPU')
    parser.add_argument('--early_stop', type=int, default=4,
                        help='number of no improtment epoch to trigger early stop')

    args = parser.parse_args()

    return args

def predict():
    ''' Evalute on test data'''
    test_loss, test_time = evaluate(test_data, test_batch_size)
    logging.info('-' * 89)
    logging.info('| End of training | test time {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}'.format(
                  test_time, test_loss, math.exp(test_loss)))
    logging.info('-' * 89)


def evaluate(data_source, batch_size):
    '''https://mxnet.incubator.apache.org/api/python/autograd/autograd.html#train-mode-and-predict-mode'''
    tic = time.time()
    total_loss = 0
    N = 0
    states = model.begin_state(batch_size, ctx=ctxs[0])
    for cursor in range(0, data_source.shape[0] - 1, args.bptt):
        Xs, Ys = get_batch(data_source, cursor, args)
        # By default, MXNet is in predict_mode
        output, states, _, _ = model(Xs, states) # state(num_layers, bsz, hidden_size)
        states = detach(states)
        total_loss += nd.sum(batch_size * loss(output, Ys)).asscalar() # loss (seq_len,)
        N += batch_size * len(output)

    return (total_loss / N), time.time() - tic


def train(load_best_loss):
    '''If gluon trainer recognizes multi-devices,
    it will automatically aggregate the gradients and synchronize the parameters.'''

    logging.info('-' * 50 + "Begin training" + '-' * 50)
    # Loop over epochs.
    best_loss = float("Inf")
    not_improves_times = 0
    best_epoch = 0
    for epoch in range(start_epoch, args.epochs):

        cur_lr = trainer.learning_rate
        tic = time.time()
        train_one_epoch(epoch, cur_lr)
        val_loss, val_time = evaluate(val_data, eval_batch_size)
        toc = time.time()
        trainer.set_learning_rate(cur_lr)

        logging.info('-' * 120)
        logging.info('| end of epoch {:3d} with lr {:2.4f} | train time: {:5.2f}s | val time: {:5.2f}s | '
                     ' valid loss {:5.3f} | valid ppl {:8.2f}'.format(
                                      epoch, cur_lr, toc - tic, val_time, val_loss, math.exp(val_loss)))
        epoch_info.append([epoch, cur_lr, toc - tic, val_time, val_loss, math.exp(val_loss) ])
        utils.save_info(epoch_info, epoch_file)
        logging.info('-' * 120)

        ''' If no pre-trained model loaded, the load_best_loss is float("Inf"),
                then any improvment will be saved, and update the load_best_loss
            if loaded from pre-trained model, there is valid real value load_best_loss
                But still need val_loss to help find a good downward direction along the loss surface,
        '''
        if val_loss < best_loss:
            if val_loss < load_best_loss:
                load_best_loss = val_loss
                model.save_parameters(os.path.join(path, 'model.params'))
                utils.read_kvstore(trainingfile, update={'lr': cur_lr})
                logging.info('Performance improving; Save the best model!')
            else:
                logging.info('Performance improving, but not the best one.')
            best_loss = val_loss
            best_epoch = epoch
            not_improves_times = 0
        else:
            not_improves_times += 1
            if not_improves_times == args.early_stop and args.schedual_rate:
                not_improves_times = 0
                load_model()
                new_lr = args.schedual_rate * cur_lr
                trainer.set_learning_rate(new_lr)
                logging.info('No improvement, anneal lr to {:2.4f}, rolling back to epoch {}'.format(
                                new_lr, best_epoch))
                epoch_info.append(["roll_back_to", None, None, None, None, None, best_epoch])
                batch_info.append(["roll_back_to", best_epoch])
        utils.read_kvstore(trainingfile, update={'epoch': epoch})

def train_one_epoch(epoch, cur_lr):
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
        # Rescale learning rate depending on the variable length w.r.t bptt
        trainer.set_learning_rate(cur_lr * seq_len / args.bptt)
        ########################################################################


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
        loss_list = []
        with autograd.record(): # train_mode
            for i, X in enumerate(Xs):
                 output, states[i], encoded_raw, encoded_dropped = model(X, states[i]) # state(num_layers, bsz, hidden_size)
                 device_loss = joint_loss(output, Ys[i], encoded_raw, encoded_dropped)
                 loss_list.append(device_loss.as_in_context(ctxs[0]) / X.size)
        for l in loss_list:
            l.backward()

        ''' trainer.allreduce_grads()
            For each parameter, reduce the gradients from different contexts.
            Should be called after autograd.backward(), outside of record() scope, and before trainer.update().
            For normal parameter updates, step() should be used, which internally calls allreduce_grads() and then update().
            However, in gradient clipping, manually call allreduce_grads() and update() separately.
        '''
        # trainer.allreduce_grads()
        # grads = [p.grad(ctxs[0]) for p in parameters]
        grads = [p.grad(ctx) for ctx in ctxs for p in parameters]
        gluon.utils.clip_global_norm(grads, args.clipping_theta)
        trainer.step(1)
        # trainer.update(1)

        batch_loss = sum([nd.sum(l).asscalar() for l in loss_list]) / len(ctxs)
        toc_b = time.time()
        batch_info.append([epoch, batch, trainer.learning_rate, seq_len, (toc_b - tic_b) * 1000,
                      args.batch_size * seq_len // (toc_b - tic_b), batch_loss, math.exp(batch_loss) ])

        total_loss += batch_loss

        if batch % args.log_interval == 0 and batch > 0:
            utils.save_info(batch_info, batch_file)

            toc_log_interval = time.time()
            total_loss = total_loss / args.log_interval

            logging.info('| epoch {:4d} ({:5.2f}%)| batch {:4d} | lr {:7.4f} | seq_len {:2d} | {:4.0f} ms/batch | '
                         '{:5d} tokens/s | loss {:6.3f} | ppl {:5.2f}'.format(
                epoch, cursor / train_data.shape[0] * 100, batch, trainer.learning_rate, seq_len,
                (toc_log_interval - tic_log_interval) * 1000 / args.log_interval,
                int(args.batch_size * args.log_interval * seq_len / (toc_log_interval - tic_log_interval)), total_loss,
                math.exp(total_loss)))

            total_loss = 0
            tic_log_interval = time.time()

        batch += 1
        cursor += seq_len

        global parameters_count
        if not parameters_count:
            logging.info('Parameters (except embeding): {}'.format(sum(p.data(ctxs[0]).size for p in parameters)))
            parameters_count = 1

    nd.waitall() # synchronize batch data
    ############################################################################

def load_model(load_states=True):
    if utils.check_file(params):
        model.load_params(params, ctx=ctxs)
        logging.info("Loading parameters from : {}".format(params))

    if load_states and utils.check_file(trainingfile):
        trainer.set_learning_rate(float(utils.read_kvstore(trainingfile)['lr']))
        logging.info("Loading lr from : {}".format(trainingfile))



if __name__ == "__main__":
    ###############################################################################
    # prepare configuration, files for saving data and logging information
    ###############################################################################
    args = configuration()
    ''' If continue training a pre-trained model,
        the lr and start epoch is read from the trainingfile
    '''
    if args.continue_exprm:
        path = utils.make_dir([args.save, args.continue_exprm])
        configfile = os.path.join(path, 'config.json')
        trainingfile = os.path.join(path, 'training.json')
        if not utils.check_file(trainingfile):
            utils.save_kvstore({'epoch' : 0, 'lr' : args.lr}, trainingfile)
            logging.info("trainingfile not found, create a new one.")

        try:
            args = data.Config(utils.read_kvstore(configfile,
                                        {'continue_exprm' : args.continue_exprm,
                                        'predict_only' : args.predict_only,
                                        'seed' : args.seed,
                                        'early_stop' : args.early_stop,
                                        'debug' : args.debug}))
            training_states = utils.read_kvstore(trainingfile)
            args.lr = float(training_states['lr'])
            start_epoch = training_states['epoch'] + 1
        except FileNotFoundError: raise

    else:
        # By default, use argparse for configuration
        if args.tied and args.model == 'StandardRNN':
            args.hid_size = args.emb_size
        path = utils.make_dir([args.save, args.model+'-'+args.rnn_cell+args.exprm])
        args = data.Config(utils.save_kvstore(vars(args), os.path.join(path, 'config.json')))
        start_epoch = 0
        trainingfile = os.path.join(path, 'training.json')
        utils.save_kvstore({'epoch' : 0, 'lr' : args.lr}, trainingfile)

    logging.basicConfig(level=logging.INFO,
                        handlers = [
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(path, "log.log"))
                        ])

    # config the conputation resources
    if args.cpu:
        ctxs = [mxnet.cpu()]
    else:
        ctxs = utils.try_all_gpus(args.num_gpus)
    assert args.batch_size % len(ctxs) == 0, \
    'Total batch size must be multiple of the number of devices'
    m = args.batch_size // len(ctxs)
    logging.info("Split batch samples (batch size={}) to {}, each device loaded {} samples".format(
                args.batch_size, ctxs, m))

    # Set the random seed manually for reproducibility.
    if args.seed:
        np.random.seed(args.seed)
        mxnet.random.seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    tic = time.time()
    corpus = data.Corpus(args.data, args.debug, args.predict_only)
    vocab_size = len(corpus.dictionary)
    logging.info('Cost {:5.2f}s to load {} train_tokens, {} valid_tokens, {} test_tokens. Vicabulary size {}. '
                  'Around {} batches/epoch'.format(time.time() - tic,
                    len(corpus.train), len(corpus.valid), len(corpus.test), vocab_size,
                    len(corpus.train) // (args.batch_size * args.bptt)))

    if not args.log_interval:
         args.log_interval = len(corpus.train) // (args.batch_size * args.bptt) // args.log_freq

    eval_batch_size = 10 if args.data == 'penn' else m
    test_batch_size = 1

    if not args.predict_only:
        train_data = batchify(corpus.train, args.batch_size).as_in_context(ctxs[0])
    val_data = batchify(corpus.valid, eval_batch_size).as_in_context(ctxs[0])
    test_data = batchify(corpus.test, test_batch_size).as_in_context(ctxs[0])

    ###############################################################################
    # Build the model
    ###############################################################################
    '''Model parameters aved path: /train/experiment/model.params.
    If parameters exists, load the saved parameters, else initialize'''

    params = os.path.join(path, 'model.params')

    if args.last_hid_size < 0:
        args.last_hid_size = args.emb_size
    if args.drop_l < 0:
        args.drop_l = args.drop_h
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size

    if args.model == 'MOSRNN' or args.model == 'MOS':
        model = model.MOSRNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size, args.last_hid_size, args.num_layers,
                           tie_weights=args.tied, dropout=args.dropout, weight_drop=args.w_drop, drop_h=args.drop_h,
                           drop_i=args.drop_i, drop_e=args.drop_e, drop_l=args.drop_l, num_experts=args.num_experts)
    elif args.model == 'AWDRNN' or args.model == 'AWD' :
        ''' AWDRNN model from gluonnlp (actually w/o asgd), not support (emb_size != last_hid_size)'''
        model = gluonnlp.model.train.AWDRNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size, args.num_layers,
                     tie_weights=args.tied, dropout=args.dropout, weight_drop=args.w_drop,
                     drop_h=args.drop_h, drop_i=args.drop_i, drop_e=args.drop_e)
    elif args.model == 'WDRNN' or args.model == 'WD' :
        ''' Weight-decay RNN (w/o asgd), support (emb_size != last_hid_size), but with an extra dense layer '''
        model = model.WDRNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size, args.last_hid_size, args.num_layers,
                     tie_weights=args.tied, dropout=args.dropout, weight_drop=args.w_drop,
                     drop_h=args.drop_h, drop_i=args.drop_i, drop_e=args.drop_e)
    elif args.model == 'RNN':
        ''' RNN , support (emb_size != last_hid_size), but with an extra dense layer '''
        model = model.RNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size, args.last_hid_size, args.num_layers,
                     tie_weights=args.tied, dropout=args.dropout, weight_drop=0,
                     drop_h=0, drop_i=0, drop_e=0)
    elif args.model == 'StandardRNN':
        model = gluonnlp.model.StandardRNN(args.rnn_cell, vocab_size, args.emb_size, args.hid_size,
                                        args.num_layers, dropout=args.dropout, tie_weights=args.tied)


    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    ar_loss = gluonnlp.loss.ActivationRegularizationLoss(args.alpha)
    tar_loss = gluonnlp.loss.TemporalActivationRegularizationLoss(args.beta)
    joint_loss = JointActivationRegularizationLoss(loss, ar_loss, tar_loss)

    model.initialize(init.Xavier(), ctx=ctxs)
    model.hybridize()
    if args.optimizer == 'SGD':
        trainer_params = {'learning_rate': args.lr,
                      'momentum': 0,
                      'wd': args.wdecay}
    elif args.optimizer == 'Adam':
        trainer_params = {'learning_rate': args.lr,
                      'wd': args.wdecay,
                      'beta1': 0,
                      'beta2': 0.999,
                      'epsilon': 1e-9}
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params)

    load_best_loss = float("Inf")
    if args.continue_exprm:
        load_model()
        load_best_loss, val_time = evaluate(val_data, eval_batch_size)
        load_best_ppl = math.exp(load_best_loss)
        logging.info("Loaded model: val_time {:5.2f}, valid loss {}, ppl {}\
                     ".format(val_time, load_best_loss, load_best_ppl))

    # At any point you can hit Ctrl + C to break out of training early.
    # logging.info(model.summary(nd.zeros((args.bptt, m))))

    try:
        if not args.predict_only:
            # set the header of csv logging files
            epoch_info = []
            epoch_file = os.path.join(path, 'epoch_results.csv')
            utils.save_info(['epoch', 'lr', 'train_time(s)', 'val_time(s)', 'val_loss', 'perplexity'], epoch_file)

            batch_info = []
            batch_file = os.path.join(path, 'batch_results.csv')
            utils.save_info(['epoch', 'batch', 'learning_rate', 'seq_len', 'ms/batch', 'tokens/s',
                             'val_loss', 'perplexity'], batch_file)
            parameters = model.collect_params().values()
            parameters_count = 0
            train(load_best_loss)
    except KeyboardInterrupt:
            logging.info('-' * 89)
            logging.info('Exiting from training early')
    finally:
        logging.info('Start evaluation on test_data')
        predict()
