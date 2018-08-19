## Mixture of softmaxes (MoS) language model
MoS model by Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. https://github.com/zihangdai/mos

This is the code we used to reproduce the paper
>[Breaking the Softmax Bottleneck: A High-Rank RNN Language Model](https://arxiv.org/abs/1711.03953)

## Requirements
Python 3.0, MXNet 1.2.0, Gluonnlp 0.3.0

## Train the models

### Penn Treebank
```python code/main.py --batch_size 12 --lr 20.0 --epoch 1000 --num_gpus 1```

### WikiText-2
```python code/main.py --epochs 1000 --data wikitext-2 --drop_h 0.2 --hid_size 600 --last_hid_size 400 --emsize 280 --batch_size 15 --lr 15.0 --drop_l 0.29 --max_seq_len_delta 20 --drop_i 0.55 --num_gpus 5```

If you have large enough memory:
```python code/main.py --epochs 1000 --data wikitext-2 --drop_h 0.2 --hid_size 1150 --last_hid_size 650 --emsize 300 --batch_size 15 --lr 15.0 --drop_l 0.29 --max_seq_len_delta 20 --drop_i 0.55 --num_gpus 5```

## Play around with AWD model
`--model AWDRNN`
