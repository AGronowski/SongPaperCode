import tensorflow as tf
from tfutils import utils as tu
import click
from data.adult import create_adult_datasets
from methods import LagrangianFairTransferableAutoEncoder
from collections import namedtuple
import numpy as np

tfd = tf.contrib.distributions

# True for adult dataset, False for mental health dataset
adult_bool = True


# adding object in brackets does nothing in Python3, only used so code works in Python2
class VariationalEncoder(object):
    def __init__(self, z_dim=10):
        # x is input features combined with sensitive u
        def encoder_func(x):
            # softplus activation function: ln(1+e^x) similar to ReLu
            # 1st hidden layer
            fc1 = tf.layers.dense(x, 50, activation=tf.nn.softplus)

            # output is used as the mean and standard deviation
            mean = tf.layers.dense(fc1, z_dim, activation=tf.identity)
            # taking log of variance makes the range be all numbers instead of only positive
            logstd = tf.layers.dense(fc1, z_dim, activation=tf.identity)
            return mean, tf.exp(logstd)

        # z is the representation
        # used to try to predict protected variable u
        def discriminate_func(z):
            # 2nd hidden layer
            fc1 = tf.layers.dense(z, 50, activation=tf.nn.softplus)
            # logits : output that hasn't been normalized
            logits = tf.layers.dense(fc1, 1, activation=tf.identity)
            return logits

        def classify_func(z):
            fc1 = tf.layers.dense(z, 50, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, 1, activation=tf.identity)
            return logits

        # templates are functions that create variables and then reuse them
        self.encoder = tf.make_template('encoder/x', lambda x: encoder_func(x))
        self.discriminate = tf.make_template('disc_01/u', lambda z: discriminate_func(z))
        self.discriminate_0 = tf.make_template('disc_0/u', lambda z: discriminate_func(z))
        self.discriminate_1 = tf.make_template('disc_1/u', lambda z: discriminate_func(z))
        self.classify = tf.make_template('classify/y', lambda z: classify_func(z))

    # property lets a method be accessed as an attribute instead of a method with ()
    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'encoder' in var.name]

    @property
    def discriminate_vars(self):
        return [var for var in tf.global_variables() if 'disc' in var.name]

    @property
    def classify_vars(self):
        return [var for var in tf.global_variables() if 'classify' in var.name]

    def sample_and_log_prob(self, x, u, y):
        #combine input features with protected variable u, then input into encoder neural network
        #axis = 1 means new column added to right of the matrix

        loc1, scale1 = self.encoder(tf.concat([x, u], axis=1))
        #q(z|x) is multivariate normal distibution, with mean and diagonal of covariance matrix coming from the encoder neural network
        qzx = tfd.MultivariateNormalDiag(loc=loc1, scale_diag=scale1)  # q(z_1 | x, u)
        z1 = qzx.sample()
        #try to predict u by putting sample from q(z|x,u) into discriminator
        #put sample into discriminator(1 layer NN)
        # q(u) is bernoulli distribution. Its probability is the prediction of u with (x,u) as the input
        # this is done 3 times using the same sample
        logits_u = self.discriminate(z1)
        #logits is log probability
        qu = tfd.Bernoulli(logits=logits_u) #q(u|z)
        logits_u0 = self.discriminate_0(z1)
        qu0 = tfd.Bernoulli(logits=logits_u0)
        logits_u1 = self.discriminate_1(z1)
        qu1 = tfd.Bernoulli(logits=logits_u1)
        #get sample from q(u)
        u_ = qu.sample()

        #try to predict y from y
        logits_y = self.classify(z1)
        qy = tfd.Bernoulli(logits=logits_y)

        #log_prob returns the natural log of the pdf evaluated at the given sample value
        #axis = 1 means 1 column with each row the sum of all elements that were in the row
        return z1, u_, \
               qzx.log_prob(z1), \
               tf.reduce_sum(qu.log_prob(u_), axis=1), tf.reduce_sum(qu.log_prob(u), axis=1), \
               qy, tf.reduce_sum(qu0.log_prob(u), axis=1), tf.reduce_sum(qu1.log_prob(u), axis=1),


class VariationalDecoder(object):

    #dimensions are 102 for adult, 30 for mental health
    if adult_bool:
        x_dim = 102
    else:
        x_dim = 30

    def __init__(self, z_dim=10, x_dim=x_dim):
        self.z_dim = z_dim

        #predict x from z
        def decoder_func(z):
            fc1 = tf.layers.dense(z, 50, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, x_dim, activation=tf.identity)
            return logits

        self.decoder = tf.make_template('decoder/x', lambda z: decoder_func(z))

    def sample_and_log_prob(self, z, x, u):
        z1 = z
        #standard normal distribution
        #zeros_like and ones_like returns tensor of same type and shape as
        #loc is mean, and scale_diag is the diagonal of the covariance matrix
        pz1 = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z1), scale_diag=tf.ones_like(z1))
        #u column added to right of z matrix, then decoded
        x_ = self.decoder(tf.concat([z1, u], axis=1))  # p(x | z_1, u)
        #102 dimensional bernoulli distribution
        pxz = tfd.Bernoulli(logits=x_)
        return x_, tf.reduce_sum(pxz.log_prob(x), axis=1), pz1.log_prob(z1)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'decoder' in var.name]


@click.command()
@click.option('--mi', type=click.FLOAT, default=1.0)
@click.option('--e1', type=click.FLOAT, default=1.0)
@click.option('--e2', type=click.FLOAT, default=0.0)
@click.option('--e3', type=click.FLOAT, default=0.0)
@click.option('--e4', type=click.FLOAT, default=0.0)
@click.option('--e5', type=click.FLOAT, default=0.0)
@click.option('--disc', type=click.INT, default=1)
@click.option('--lag', is_flag=True, flag_value=True)
@click.option('--test', is_flag=True, flag_value=True)
#in original code default is set to -1
#here setting default to -1 makes code run without terminating
@click.option('--gpu', type=click.INT, default=0)


# e1: Upper bound for MI
# e2: Adversarial approximation to Demographic parity
# e4: Adversarial approximation to Equalized odds
# e5: Adversarial approximation to Equalized opportunity
# disc: discriminator iterations

# lag = True - L_MIFR (lagrangian optimization - e parameters represent epsilon constraints)
# lag = False - MIFR (fixed multipliers - e parameters represent lambdas)


def main(mi, e1, e2, e3, e4, e5, disc, lag, test, gpu):
    test_bool = test

    # print('test is ' + str(test))
    # if lag:
    #     print('lag is True - running L-MIFR (lagrangian optimization)')
    # else:
    #     print('lag is False - running MIFR (fixed multipliers)')

    import time
    start_time = time.time()

    #run multiple experiments with the following e1 and e2 values
    points = np.linspace(0,2,11)
    combinations = [(a,b) for a in points for b in points]
    for i, j in combinations:
        e1 = i
        e2 = j

        import os
        if gpu == -1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            device_id = tu.find_avaiable_gpu()
        else:
            device_id = gpu

        try:
            jobid = os.environ['SLURM_JOB_ID']
        except:
            jobid = '0'
        print('Using device {}'.format(device_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        z_dim = 10

        #train and test dataset, p(u) is bernoulli distibution of protected gender variable
        train, test, pu = create_adult_datasets(batch=64)
        print('created datasets')

        #named tuple is like dictionary, values can be accessed either by key or index
        Datasets = namedtuple('datasets', ['train', 'test'])
        datasets = Datasets(train=train, test=test)

        #trainable is false if variables don't need differentiation
        global_step = tf.Variable(0, trainable=False, name='global_step')
        starter_learning_rate = 0.001
        #learning rate is lowered as training progresses
        #global step keeps track of the current learning rate
        #current learning rate multiplied by 0.98 every 1000 steps
        #staircase False means no integer division
        #decayed_learning_rate = _starter_learning_rate * decay_rate ^ (global_step / decay_steps)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   1000, 0.98, staircase=False)
        #default beta1 for adam optimizer is 0.9
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        encoder = VariationalEncoder(z_dim=z_dim)
        decoder = VariationalDecoder(z_dim=z_dim)
        if lag:
            print('lag is True')
            #found in utils in tfutils. currectly path is hardcoded
            logdir = tu.obtain_log_path('fair/lmifr_n/adult/{}-{}-{}-{}-{}-{}-{}/'.format(mi, e1, e2, e3, e4, e5, disc))
        else:
            print('lag is False')
            logdir = tu.obtain_log_path('fair/mifr_n/adult/{}-{}-{}-{}-{}-{}-{}/'.format(mi, e1, e2, e3, e4, e5, disc))

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        with open(os.path.join(logdir, 'jobid'), 'w') as f:
            f.write(jobid)

        lvae = LagrangianFairTransferableAutoEncoder(encoder, decoder, datasets, optimizer, logdir, pu, mi, e1, e2, e3, e4, e5, lag, disc)
        lvae.global_step = global_step

        #how many epochs to train over
        #default is 2000
        if not test_bool:
            #train function is defined in vae.py
            lvae.train(num_epochs=1000)
        # lvae.test()
        lvae.evaluate_classifier()
        tf.reset_default_graph()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

