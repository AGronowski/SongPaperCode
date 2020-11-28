import tensorflow as tf
from tfutils import utils as tu
import click
from data.adult import create_adult_datasets
from methods import LagrangianFairTransferableAutoEncoder
from collections import namedtuple

tfd = tf.contrib.distributions


class VariationalEncoder(object):
    def __init__(self, z_dim=10):
        #x is input images
        def encoder_func(x):
            #50 node hidden layer of neural network
            fc1 = tf.layers.dense(x, 50, activation=tf.nn.softplus)
            #10 node output layer
            mean = tf.layers.dense(fc1, z_dim, activation=tf.identity)
            logstd = tf.layers.dense(fc1, z_dim, activation=tf.identity)
            return mean, tf.exp(logstd)

        #z is the representation
        def discriminate_func(z):
            fc1 = tf.layers.dense(z, 50, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, 1, activation=tf.identity)
            return logits

        def classify_func(z):
            fc1 = tf.layers.dense(z, 50, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, 1, activation=tf.identity)
            return logits

        self.encoder = tf.make_template('encoder/x', lambda x: encoder_func(x))
        self.discriminate = tf.make_template('disc_01/u', lambda z: discriminate_func(z))
        self.discriminate_0 = tf.make_template('disc_0/u', lambda z: discriminate_func(z))
        self.discriminate_1 = tf.make_template('disc_1/u', lambda z: discriminate_func(z))
        self.classify = tf.make_template('classify/y', lambda z: classify_func(z))

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
        loc1, scale1 = self.encoder(tf.concat([x, u], axis=1))
        #q(z|x) is multivariate normal distibution, with mean and diagonal of covariance matrix coming from the encoder neural network
        qzx = tfd.MultivariateNormalDiag(loc=loc1, scale_diag=scale1)  # q(z_1 | x, u)
        z1 = qzx.sample()
        #put sample into discriminator(1 layer NN)
        logits_u = self.discriminate(z1)
        #q(u) is bernoulli distribution. Its probability is the prediction of u with (x,u) as the input
        #this is done 3 times using the same sample
        qu = tfd.Bernoulli(logits=logits_u)
        logits_u0 = self.discriminate_0(z1)
        qu0 = tfd.Bernoulli(logits=logits_u0)
        logits_u1 = self.discriminate_1(z1)
        qu1 = tfd.Bernoulli(logits=logits_u1)
        #get sample from q(u)
        u_ = qu.sample()
        #classify the sample
        logits_y = self.classify(z1)

        qy = tfd.Bernoulli(logits=logits_y)

        #log_prob returns the log of the pdf evaluated at the given sample value
        return z1, u_, \
               qzx.log_prob(z1), \
               tf.reduce_sum(qu.log_prob(u_), axis=1), tf.reduce_sum(qu.log_prob(u), axis=1), \
               qy, tf.reduce_sum(qu0.log_prob(u), axis=1), tf.reduce_sum(qu1.log_prob(u), axis=1),


class VariationalDecoder(object):
    def __init__(self, z_dim=10, x_dim=102):
        self.z_dim = z_dim

        def decoder_func(z):
            fc1 = tf.layers.dense(z, 50, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, x_dim, activation=tf.identity)
            return logits

        self.decoder = tf.make_template('decoder/x', lambda z: decoder_func(z))

    def sample_and_log_prob(self, z, x, u):
        z1 = z
        #standard normal distribution
        pz1 = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z1), scale_diag=tf.ones_like(z1))
        x_ = self.decoder(tf.concat([z1, u], axis=1))  # p(x | z_1, u)
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

    Datasets = namedtuple('datasets', ['train', 'test'])
    datasets = Datasets(train=train, test=test)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.98, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
    encoder = VariationalEncoder(z_dim=z_dim)
    decoder = VariationalDecoder(z_dim=z_dim)
    if lag:
        print('lag is True')
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


    #import time
    #start_time = time.time()

    #how many epochs to train over
    #default is 2000
    if not test_bool:
        lvae.train(num_epochs=2000)
    lvae.test()
    lvae.evaluate_classifier()

    #print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()

