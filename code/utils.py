import os, json, csv, mxnet

def check_file(file):
    ''' Check if file is valid'''
    return os.path.exists(file) and os.path.isfile(file) and os.path.getsize(file)>0

def detach(state):
    if isinstance(state[0], (tuple, list)):
        state = [ [i.detach() for i in s] for s in state]
    elif isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


def batchify(data, batch_size):
    '''Work out how cleanly we can divide the dataset into bsz parts.
    By reshaping the index list into seq order major batch(num_batches, batch_size)
    [[ 0.  3.  6.  9. 12. 15.]
     [ 1.  4.  7. 10. 13. 16.]
     [ 2.  5.  8. 11. 14. 17.]]'''
    num_batches = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:num_batches*batch_size]
    # Evenly divide the data across the bsz batches.
    data = data.reshape((batch_size, num_batches)).T
    return data


def get_batch(source, i, args, seq_len=None):
    ''' seq_len acts as random shuffler, and introduces random sequence length
    input: a batchify data source.
    return: data(seq_len, batch_size)
    '''
    seq_len = min(seq_len if seq_len else args.bptt, source.shape[0]-1-i)
    X = source[i : i+seq_len]
    Y = source[i+1 : i+1+seq_len]
    return X, Y

def make_dir(path):
    make_path = os.path.join(*path)
    if not os.path.exists(make_path):
        os.makedirs(make_path)
    return make_path

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)

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

def save_config(config, file):
    ''' config: dict '''

    with open(file, 'w') as configfile:
        json.dump(config, configfile)

    return config


def read_config(file, update=None):
    ''' ONLY when continue_exprm.
    update provide {K:V} pair to update'''
    with open(file, 'r') as configfile:
        config = json.load(configfile)
        configfile.close()

    if update:
        config.update(update)
        save_config(config, file)

    return config


def save_info(results, file):
    ''' save the logging info to csv file, and clear the buffer'''
    with open(file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', dialect='excel')
        if isinstance(results[0], list):
            writer.writerows(results)
        else:
            writer.writerow(results)
        csvfile.close()

    del results[:]


def get_params(params, ctx):
    ''' Copy parameters to specific GPU
    Usage: new_params = get_params(params, mxnet.gpu(0))'''
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params


def allreduce(data):
    '''reduce data from all GPU and broadcast
    Usage:  data = [mxnet.ndarray.ones((1,2), ctx=mxnet.gpu(i)) * (i + 1) for i in range(2)]
            allreduce(data)'''
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])


def split_and_load(data, ctx, batch_axis=1):
    ''' Split data(seq_len, batch_size) into len(ctx_list) slices along batch_axis to each GPUs.
    Usage:  ctx = [mxnet.gpu(0), mxnet.gpu(1)]
            splitted = split_and_load(data, ctx) '''
    n, k = data.shape[batch_axis], len(ctx)
    m = n // k
    assert m * k == n, '# examples is not divided by # devices.'
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]

def get_mem():
    ''' Monitor memory usage, only works in Linux or MacOS
    Usage: start = get_mem()
            ...
           get_mem() - start '''
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3

def try_all_gpus(num_GPUs):
    """Return all available GPUs, or [mxnet.cpu()] if there is no GPU"""
    ctxes = []
    try:
        for i in range(num_GPUs):
            ctx = mxnet.gpu(i)
            _ = mxnet.ndarray.array([0], ctx=ctx)
            ctxes.append(ctx)
    except:
        raise ValueError('Failed to get the request number of GPUs')

    if not ctxes:
        ctxes = [mxnet.cpu()]
    return ctxes
