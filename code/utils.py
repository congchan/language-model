import os, shutil
import torch

def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


def batchify(data, batch_size):
    '''Reshape the index list into matrix of shape(num_batches, batch_size)
    [[ 0.  4.  8.]
    [ 1.  5.  9.]
    [ 2.  6. 10.]
    [ 3.  7. 11.]]'''
    num_batches = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:num_batches*batch_size]
    # Evenly divide the data across the bsz batches.
    data = data.reshape((batch_size, num_batches)).T
    return data


def get_batch(source, i, seq_len=None):
    ''''input: a batchify data source. '''
    seq_len = min(seq_len if seq_len else args.bptt, source.shape[0]-1-i)
    X = source[i : i+seq_len]
    Y = source[i+1 : i+1+seq_len]
    return X, Y.reshape((-1,))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, trainer, path, finetune=False):
    """Block support .save_params() and .load_params()"""
    if finetune:
        model.save_params(os.path.join(path, 'finetune_model.params'))
        trainer.save_states(os.path.join(path, 'finetune_trainer.states'))
    else:
        model.save_params(os.path.join(path, 'model.params'))
        trainer.save_states(os.path.join(path, 'trainer.states'))


def get_params(params, ctx):
    ''' Copy parameters to specific GPU
    Usage: new_params = get_params(params, mx.gpu(0))'''
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params


def allreduce(data):
    '''reduce data from all GPU and broadcast
    Usage:  data = [nd.ones((1,2), ctx=mx.gpu(i)) * (i + 1) for i in range(2)]
            allreduce(data)'''
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])


def split_and_load(data, ctx):
    ''' Split data to each GPUs
    Usage:  batch = nd.arange(24).reshape((6, 4))
            ctx = [mx.gpu(0), mx.gpu(1)]
            splitted = split_and_load(batch, ctx) '''
    n, k = data.shape[0], len(ctx)
    m = n // k
    assert m * k == n, '# examples is not divided by # devices.'
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]
