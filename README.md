# Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis

TensorFlow implementation of [Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis](https://openreview.net/forum?id=H1Xw62kRZ).

![introduction](./assets/introduction.png)


## Requirements

- Python 2.7+
- [TensorFlow](https://www.tensorflow.org/) 1.4.1
- [tqdm](https://github.com/tqdm/tqdm)

## Usage

To train a model:

    $ python main.py
    $ tensorboard --logdir=logs --host=0.0.0.0

To synthesize a Karel program given an input/output pair:

    $ python main.py --test=True --map=examples/test.map

To run Karel interpreter:

    $ python -m karel.interpreter KAREL MAP
    $ python -m karel.interpreter ./examples/simple.karel ./examples/simple.map


## Results

(in progress)


## References

- [Neural Program Meta-Induction](https://arxiv.org/abs/1710.04157)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
