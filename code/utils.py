import os, shutil
import torch

def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


def batchify(data, batch_size):
    num_batches = data.shape[0] // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:num_batches*batch_size]
    # Evenly divide the data across the bsz batches.
    data = data.reshape((batch_size, num_batches)).T
    return data


def get_batch(source, i, seq_len=None):
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
