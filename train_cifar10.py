import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import sys

sys.path.append('/path/to/home')

import nn
import plotting
import cifar10_data

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=400)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--sample_dim', type=int, default=10)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--AE_weight', type=float, default=1.)
parser.add_argument('--gen_lab_weight', type=float, default=1.)
parser.add_argument('--gen_jacobi_weight', type=float, default=1.)
parser.add_argument('--disc_lap_weight_lab', type=float, default=1.)
parser.add_argument('--disc_lap_weight_unl', type=float, default=1.)
parser.add_argument('--z_delta', type=float, default=0.0001)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/path/to/data/cifar-10-python')
parser.add_argument('--save_dir', type=str, default='/path/to/save_dir')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()

trainx_gen = trainx.copy()
trainx_unl3 = trainx.copy()

testx, testy = cifar10_data.load(args.data_dir, subset='test')
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# generator
x = T.tensor4()
n_dim = args.n_dim
sample_dim = args.sample_dim
noise_dim = (args.batch_size, n_dim)
noise = theano_rng.uniform(size=noise_dim)
gen_img_input = ll.InputLayer(shape=(None, 3, 32, 32))
n_batch = gen_img_input.shape[0]
gen_noise_input = ll.InputLayer(shape=noise_dim)
gen_layers = [nn.batch_norm(dnn.Conv2DDNNLayer(gen_img_input, 32, (5,5), stride=(2,2), pad=2, W=Normal(0.05), nonlinearity=nn.lrelu), g=None)] # 32 -> 16
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 64, (5,5), stride=(2,2), pad=2, W=Normal(0.05), nonlinearity=nn.lrelu), g=None)) # 16 -> 8
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 128, (5,5), stride=(2,2), pad=2, W=Normal(0.05), nonlinearity=nn.lrelu), g=None)) # 8 -> 4
gen_layers.append(ll.GlobalPoolLayer(gen_layers[-1]))
gen_layers.append(nn.batch_norm(ll.ConcatLayer([gen_noise_input, gen_layers[-1]], axis=1), g=None))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4)))
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu), g=None))
gen_layers.append(nn.batch_norm(ll.NINLayer(gen_layers[-1], num_units=512, W=Normal(0.05), nonlinearity=nn.lrelu)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (n_batch,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu), g=None))
gen_layers.append(nn.batch_norm(ll.NINLayer(gen_layers[-1], num_units=256, W=Normal(0.05), nonlinearity=nn.lrelu)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (n_batch,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu), g=None))
gen_layers.append(nn.batch_norm(ll.NINLayer(gen_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu)))
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (n_batch,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1], {gen_img_input: x, gen_noise_input: noise})

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()
x_jacobian = T.tensor4()
z_jacobian = T.matrix()
temp = ll.get_output(gen_layers[-1], {gen_img_input: x, gen_noise_input: noise}, deterministic=False, init=False)
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=False)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

# Gradient for disc

z_delta_disc = T.tile(z_jacobian, (args.batch_size, 1))*args.z_delta
z_d_disc = T.sum(z_jacobian, axis=1).dimshuffle('x', 0)*args.z_delta

x_disc_jacobian_lab = x_lab.repeat(sample_dim, axis=0)
labels_jacobian = labels.repeat(sample_dim)
gen_dat_del_lab = ll.get_output(gen_layers[-1], {gen_img_input: x_disc_jacobian_lab, gen_noise_input: z_delta_disc}, deterministic=False)
gen_dat_zero_lab = ll.get_output(gen_layers[-1], {gen_img_input: x_disc_jacobian_lab, gen_noise_input: T.zeros_like(z_delta_disc)}, deterministic=False)
disc_dat_delta_lab = ll.get_output(disc_layers[-1], gen_dat_del_lab, deterministic=False) 
disc_dat_zero_lab = ll.get_output(disc_layers[-1], gen_dat_zero_lab, deterministic=False)
l_delta_lab = disc_dat_delta_lab[T.arange(args.batch_size*sample_dim),labels_jacobian]
l_zero_lab = disc_dat_zero_lab[T.arange(args.batch_size*sample_dim),labels_jacobian]
disc_delta_lab = (l_delta_lab - nn.log_sum_exp(disc_dat_delta_lab)) - (l_zero_lab - nn.log_sum_exp(disc_dat_zero_lab))
disc_delta_lab = disc_delta_lab.reshape((args.batch_size, sample_dim))
grad_simu_disc_lab = 1.*disc_delta_lab/z_d_disc
grad_norm_disc_lab = T.sum(abs(grad_simu_disc_lab), axis=1) / sample_dim
loss_disc_jacobian_lab = T.mean(grad_norm_disc_lab)

x_disc_jacobian_unl = x_unl.repeat(sample_dim, axis=0)
gen_dat_del_unl = ll.get_output(gen_layers[-1], {gen_img_input: x_disc_jacobian_unl, gen_noise_input: z_delta_disc}, deterministic=False)
gen_dat_zero_unl = ll.get_output(gen_layers[-1], {gen_img_input: x_disc_jacobian_unl, gen_noise_input: T.zeros_like(z_delta_disc)}, deterministic=False)
disc_dat_delta_unl = ll.get_output(disc_layers[-1], gen_dat_del_unl, deterministic=False) 
disc_dat_zero_unl = ll.get_output(disc_layers[-1], gen_dat_zero_unl, deterministic=False)

disc_delta_unl = (disc_dat_delta_unl - nn.log_sum_exp(disc_dat_delta_unl).dimshuffle(0, 'x')) - (disc_dat_zero_unl - nn.log_sum_exp(disc_dat_zero_unl).dimshuffle(0, 'x'))
disc_delta_unl = disc_delta_unl.reshape((args.batch_size, sample_dim, -1))
z_d_disc_unl = T.tile(z_d_disc, (args.batch_size, 1)).dimshuffle(0, 1, 'x')
grad_simu_disc_unl = 1.*disc_delta_unl/z_d_disc_unl

grad_norm_disc_unl = T.sum(T.sum(abs(grad_simu_disc_unl), axis=1)/sample_dim, axis=1) / disc_delta_unl.shape[2]
loss_disc_jacobian_unl = T.mean(grad_norm_disc_unl)

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl + args.disc_lap_weight_lab*loss_disc_jacobian_lab + args.disc_lap_weight_unl*loss_disc_jacobian_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.8*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab, x], outputs=None, updates=init_updates) # data based initialization
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,x,z_jacobian,lr], outputs=[loss_lab, loss_unl, loss_disc_jacobian_lab, loss_disc_jacobian_unl, train_err], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)

# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-2], x_unl, deterministic=False)
output_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(abs(m1-m2)) # feature matching loss

# AutoEncoder loss
gen_dat_ae = ll.get_output(gen_layers[-1], {gen_img_input: x, gen_noise_input: T.zeros_like(noise)})
loss_ae = T.mean((x-gen_dat_ae)**2)

# lab maintance loss
gen_dat_lab = ll.get_output(gen_layers[-1], {gen_img_input: x_lab, gen_noise_input: T.zeros_like(noise)})
output_before_softmax_gen_lab = ll.get_output(disc_layers[-1], gen_dat_lab, deterministic=False)
l_gen_lab = output_before_softmax_gen_lab[T.arange(args.batch_size),labels]
loss_gen_lab = -T.mean(l_gen_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_gen_lab)))

# Jacobian for gen
x_gen_jacobian = x.repeat(sample_dim, axis=0)
z_delta_gen = T.tile(z_jacobian, (args.batch_size, 1))*args.z_delta
gen_dat_delta = ll.get_output(gen_layers[-1], {gen_img_input: x_gen_jacobian, gen_noise_input: z_delta_gen}) - ll.get_output(gen_layers[-1], {gen_img_input: x_gen_jacobian, gen_noise_input: -z_delta_gen})
gen_dat_delta = gen_dat_delta.reshape((args.batch_size, sample_dim, -1))
gen_dat_delta = gen_dat_delta.dimshuffle(0, 2, 1)
z_d = T.tile(T.sum(z_jacobian, axis=1), (args.batch_size, 1))*args.z_delta
z_d = z_d.dimshuffle(0, 'x', 1)
jacobian_simu = 1.*gen_dat_delta / (2*z_d)
identity = T.eye(jacobian_simu.shape[2]).dimshuffle('x', 0, 1)
jacobian_orth = T.batched_dot(jacobian_simu.dimshuffle(0, 2, 1), jacobian_simu)-identity
jacobian_norm = 1.*T.sum(T.sum(abs(jacobian_orth), axis=2), axis=1) / (jacobian_simu.shape[2]**2)
loss_gen_jacobian = T.mean(jacobian_norm)

gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen+args.AE_weight*loss_ae+args.gen_lab_weight*loss_gen_lab+args.gen_jacobi_weight*loss_gen_jacobian, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl,x,x_lab,labels,z_jacobian,lr], outputs=[loss_gen, loss_ae, loss_gen_lab, loss_gen_jacobian], updates=gen_param_updates)


# select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

best_err = 1.


# //////////// perform training //////////////
for epoch in range(1200):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))
    
    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

    trainx_gen = trainx_gen[rng.permutation(trainx_gen.shape[0])]
    trainx_unl3 = trainx_unl3[rng.permutation(trainx_unl3.shape[0])]
    
    trainx1 = []
    trainy1 = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds1 = rng.permutation(txs.shape[0])
        trainx1.append(txs[inds1])
        trainy1.append(tys[inds1])
    trainx1 = np.concatenate(trainx1, axis=0)
    trainy1 = np.concatenate(trainy1, axis=0)
    
    if epoch==0:
        print(trainx.shape)
        init_param(trainx[:500], trainx_gen[:args.batch_size]) # data based initialization

        test_err = 0.
        for t in range(nr_batches_test):
            test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
        test_err /= nr_batches_test
        print('Initial test err = %.4f' % (test_err))
        best_err = test_err

    # train
    loss_lab = 0.
    loss_unl = 0.
    loss_lap_lab = 0.
    loss_lap_unl = 0.
    train_err = 0.

    loss_gen = 0.
    loss_ae = 0.
    loss_prv = 0.
    loss_jacobi_gen = 0.

    for t in range(nr_batches_train):
        z_delta = np.zeros((sample_dim, n_dim))
        z_delta_ids = np.random.choice(range(n_dim), sample_dim, replace=False)
        z_delta_ids.sort()
        z_delta[np.arange(sample_dim), z_delta_ids] = 1.
        z_delta = np.cast[th.config.floatX](z_delta)

        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size
        ll, lu, ldl, ldu, te = train_batch_disc(trainx[ran_from:ran_to],trainy[ran_from:ran_to],
                                      trainx_unl[ran_from:ran_to],trainx_gen[ran_from:ran_to],
                                      z_delta, lr)
        loss_lab += ll
        loss_unl += lu
        loss_lap_lab += ldl
        loss_lap_unl += ldu
        train_err += te
        
        lg, lae, lp, lgj = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],trainx_unl3[t*args.batch_size:(t+1)*args.batch_size],
                                    trainx1[t*args.batch_size:(t+1)*args.batch_size],trainy1[t*args.batch_size:(t+1)*args.batch_size],
                                    z_delta, lr)

        loss_gen += lg
        loss_ae += lae
        loss_prv += lp
        loss_jacobi_gen += lgj

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    loss_lap_lab /=nr_batches_train
    loss_lap_unl /= nr_batches_train
    train_err /= nr_batches_train

    loss_gen /= nr_batches_train
    loss_ae /= nr_batches_train
    loss_prv /= nr_batches_train
    loss_jacobi_gen /= nr_batches_train
    
    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, loss_lap_lab = %.4f, loss_lap_unl = %.4f, loss_gen = %.4f, loss_ae = %.4f, loss_prv = %.4f, loss_j_gen = %.4f, train err = %.4f, test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, loss_lap_lab, loss_lap_unl, loss_gen, loss_ae, loss_prv, loss_jacobi_gen, train_err, test_err))
    sys.stdout.flush()

    # generate samples from the model
    if test_err < best_err:
        print 'Best found! Saving...'
        np.savez(args.save_dir+'/models/disc_avg_params.npz', *[p.get_value() for p in disc_param_avg])
        np.savez(args.save_dir+'/models/disc_params.npz', *lasagne.layers.get_all_param_values(disc_layers))
        np.savez(args.save_dir+'/models/gen_params.npz', *lasagne.layers.get_all_param_values(gen_layers))

        best_err = test_err
        
