# Notes for running experiments

We'll be using `open_lth.py lottery --default_hparams=MODEL --levels=LEVELS --replicate=NUMBER` to run lottery ticket experiments. I've implemented `mnist_simplecnn_N1[_N2...]` to be the same CNNs we're using in other experiments. `--levels` is the number of iterations to prune with the lottery ticket procedure, and `--replicate` is a number given to a particular run of a experiment. For example, `--replicate=5` will store the experiment under

> /home/jfrankle/open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_5/main

To get 5 datapoints for an experiment, we would run 5 commands with different replicate values.

For more on arguments and hyperparameters, use `--help`.

# Results

Data is stored in `../open_lth_data` and named with an MD5 hash, so you'll need to run `--display_output_location` to keep track of where a certain experiment is stored.

# TinyCNN on MNIST 

python3.7 open_lth.py lottery --default_hparams=mnist_simplecnn_16_32  --levels=35 --replicate=1