# SSL_LGAN

This is the code for semi-supervised learning experiments described in the paper 'Global versus Localized Generative Adversarial Nets' [[pdf]](https://arxiv.org/pdf/1711.06020.pdf).

The code is modified from [the repository of 'Improved Techniques for Training GANs'](https://github.com/openai/improved-gan)

Current status: Initial release

Required Libraries: 
  * Theano
  * Lasagne
  * gpuarray

1. Semi-supervised Learning on Cifar-10

Please download all the files to your dictionary first. To conduct the semi-supervised learning on Cifar-10, please run the following commands:

```shell
THEANO_FLAGS='device=<cuda>,floatX=float32' python train_cifar10.py [--batch_size <100>|--count <400>|...]
```
To accelerate the training process, LGAN can be trained in two phases2. The first one is the training with only non-Jacobian related parameters by 

```shell
THEANO_FLAGS='device=<cuda>,floatX=float32' python train_cifar10_phase1.py [--batch_size <100>|--count <400>|--save_dir <./model>|...]
```

Then followed by the training with Jacobian related parameters by

```shell
THEANO_FLAGS='device=<cuda>,floatX=float32' python train_cifar10_phase2.py [--batch_size <100>|--count <400>|--phase1_model_dir <./model>|...]
```

2. Semi-supervised Learning on SVHN

Coming soon...
