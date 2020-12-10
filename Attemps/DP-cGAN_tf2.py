from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from tensorflow.keras import layers, models
import time
from IPython import display
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve, auc
from absl import logging
import collections
from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query

print(tf.__version__)
ConvergenceWarning('ignore')


def parseArg():
    parser = argparse.ArgumentParser(description='DP-cGAN, a generative adversarial model.')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for DP-cGAN.'
                        )

    conf_file = os.path.abspath("conf/mnist_cDCGAN.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


conf_file = parseArg()

# 从conf中读进超参数
with open(conf_file, "r") as f:
    conf_dict = eval(f.read())
    result_dir = conf_dict['result_dir']
    checkpoint_dir = result_dir + conf_dict['checkpoint_dir']
    images_dir = result_dir + conf_dict['images_dir']
    isTrain = conf_dict['is_train']
    isTest = conf_dict['is_test']
    isLoad = conf_dict['is_load']
    percentage = conf_dict['percentage']
    input_dataset = conf_dict['dataset']
    learning_rate_initial = conf_dict['learning_rate_initial']
    decay_steps = conf_dict['decay_steps']
    learning_rate_end = conf_dict['learning_rate_end']
    optimizer_type = conf_dict['optimizer_type']
    BATCH_SIZE = conf_dict['batch_size']
    NR_MICROBATCHES = conf_dict['nr_microbatch']
    NORM_CLIP = conf_dict['norm_clip']
    NOISE_MULT = conf_dict['noise_mult']
    DP_DELTA = conf_dict['dp_delta']
    EPOCHS = conf_dict['epochs']
    N_DISC = conf_dict['disc_train_step']
    Z_DIM = conf_dict['hidden_dim']
    N_GEN = conf_dict['pic_gen_num']
    COND_num_classes = conf_dict['cond_num']

del conf_dict, conf_file

'''
Dataset
Loading MNIST and creating hot encoding for labels.
'''
# 加载数据集
if input_dataset == 'MNIST':
    (train_images, train_labels), (X_test_org, Y_test_org) = tf.keras.datasets.mnist.load_data()
    pic_dim = 1
elif input_dataset == "FASHION_MNIST":
    (train_images, train_labels), (X_test_org, Y_test_org) = tf.keras.datasets.fashion_mnist.load_data()
    pic_dim = 1
elif input_dataset == 'CIFAR10':
    (train_images, train_labels), (X_test_org, Y_test_org) = tf.keras.datasets.cifar10.load_data()
    pic_dim = train_images.shape[3]
else:
    raise ValueError('UnKnown Dataset')

pic_num = train_images.shape[0]
pic_size = train_images.shape[1]
train_images = train_images.reshape(pic_num, pic_size, pic_size, pic_dim).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
train_labels = train_labels.reshape((pic_num, 1))
train_labels_vec = np.zeros((len(train_labels), COND_num_classes), dtype='float32')

X_test_org = X_test_org.reshape(X_test_org.shape[0], pic_size, pic_size, pic_dim).astype('float32')
X_test_org = (X_test_org - 127.5) / 127.5  # Normalize the images to [-1, 1]
Y_test_org = [int(y) for y in Y_test_org]
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_test_org = label_binarize(Y_test_org, classes=classes)

for i, label in enumerate(train_labels):
    train_labels_vec[i, int(train_labels[i])] = 1.0
BUFFER_SIZE = int(pic_num * percentage / 100)  # 整个训练集大小

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)


def checkpoint_name(title):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt__" + str(title))
    return (checkpoint_prefix)


def _random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    # ind = tf.compat.v1.multinomial(uniform_log_prob, n_samples)
    ind = tf.random.categorical(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")


# Method obtained from https://github.com/reihaneh-torkzadehmahani/DP-CGAN
def compute_fpr_tpr_roc(Y_test, Y_score):
    n_classes = Y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr],
                                                                                       Y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc


'''
Modified Optimizer for DP
The optimizer below is a modification of the original from TF Privacy,
available here to allow setting different values of noise multipliers and clipping factor on different steps of the optimization.
The main modification lies on the compute_gradients method, which now includes:
    curr_noise_mult: Current noise_multiplier
    curr_norm_clip: Current L2 norm clipping factor
On every step of the optimization we now additionally pass these parameters to control the privacy effects.
'''


def make_optimizer_class(cls):
    """Constructs a DP optimizer class from an existing one."""
    parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
    child_code = cls.compute_gradients.__code__
    GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
    if child_code is not parent_code:
        logging.warning(
            'WARNING: Calling make_optimizer_class() on class %s that overrides '
            'method compute_gradients(). Check to ensure that '
            'make_optimizer_class() does not interfere with overridden version.',
            cls.__name__)

    class DPOptimizerClass(cls):
        """Differentially private subclass of given class cls."""

        _GlobalState = collections.namedtuple(
            '_GlobalState', ['l2_norm_clip', 'stddev'])

        def __init__(
                self,
                dp_sum_query,
                num_microbatches=None,
                unroll_microbatches=False,
                *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
                **kwargs):
            """Initialize the DPOptimizerClass.

            Args:
              dp_sum_query: DPQuery object, specifying differential privacy
                mechanism to use.
              num_microbatches: How many microbatches into which the minibatch is
                split. If None, will default to the size of the minibatch, and
                per-example gradients will be computed.
              unroll_microbatches: If true, processes microbatches within a Python
                loop instead of a tf.while_loop. Can be used if using a tf.while_loop
                raises an exception.
            """
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self._dp_sum_query = dp_sum_query
            self._num_microbatches = num_microbatches
            self._global_state = self._dp_sum_query.initial_global_state()
            # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
            # Beware: When num_microbatches is large (>100), enabling this parameter
            # may cause an OOM error.
            self._unroll_microbatches = unroll_microbatches

        def compute_gradients(self,
                              loss,
                              var_list,
                              gate_gradients=GATE_OP,
                              aggregation_method=None,
                              colocate_gradients_with_ops=False,
                              grad_loss=None,
                              gradient_tape=None,
                              curr_noise_mult=0,
                              curr_norm_clip=1):

            self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip,
                                                                 curr_norm_clip * curr_noise_mult)
            self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip,
                                                                      curr_norm_clip * curr_noise_mult)

            # TF is running in Eager mode, check we received a vanilla tape.
            if not gradient_tape:
                raise ValueError('When in Eager mode, a tape needs to be passed.')

            vector_loss = loss()
            if self._num_microbatches is None:
                self._num_microbatches = tf.shape(input=vector_loss)[0]
            sample_state = self._dp_sum_query.initial_sample_state(var_list)
            microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
            sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

            def process_microbatch(i, sample_state):
                """Process one microbatch (record) with privacy helper."""
                microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
                grads = gradient_tape.gradient(microbatch_loss, var_list)
                sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
                return sample_state

            for idx in range(self._num_microbatches):
                sample_state = process_microbatch(idx, sample_state)

            if curr_noise_mult > 0:
                grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
            else:
                grad_sums = sample_state

            def normalize(v):
                return v / tf.cast(self._num_microbatches, tf.float32)

            final_grads = tf.nest.map_structure(normalize, grad_sums)
            grads_and_vars = final_grads  # list(zip(final_grads, var_list))

            return grads_and_vars

    return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
    """Constructs a DP optimizer with Gaussian averaging of updates."""

    class DPGaussianOptimizerClass(make_optimizer_class(cls)):
        """DP subclass of given class cls using Gaussian averaging."""

        def __init__(
                self,
                l2_norm_clip,
                noise_multiplier,
                num_microbatches=None,
                ledger=None,
                unroll_microbatches=False,
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs):
            dp_sum_query = gaussian_query.GaussianSumQuery(
                l2_norm_clip, l2_norm_clip * noise_multiplier)

            if ledger:
                dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
                                                              ledger=ledger)

            super(DPGaussianOptimizerClass, self).__init__(
                dp_sum_query,
                num_microbatches,
                unroll_microbatches,
                *args,
                **kwargs)

        @property
        def ledger(self):
            return self._dp_sum_query.ledger

    return DPGaussianOptimizerClass


GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
DPGradientDescentGaussianOptimizer_NEW = make_gaussian_optimizer_class(GradientDescentOptimizer)


# DO NOT EDIT THIS
def generate_and_save_images(title, model, epoch, test_input, test_label):
    # Notice `training` is set to False: This is so all layers run in inference mode (batchnorm).
    predictions = model([test_input, test_label], training=False)

    # fig = plt.figure(figsize=(2, COND_num_classes))

    for i in range(predictions.shape[0]):
        plt.subplot(COND_num_classes, 1, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(images_dir + '/' + title + '___image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


'''
C-GAN Models
Both Generator and Discriminator follow simple architectures, with fully connected neural networks.
We emphasize the use of C-GAN, t
herefore conditioning the models to the label information - notice the additional input on both networks below for labels.
'''


def make_generator_model_FCC():
    # INPUT: label input
    in_label = layers.Input(shape=(COND_num_classes,))

    # INPUT: image generator input
    in_lat = layers.Input(shape=(Z_DIM,))

    # MERGE
    merge = layers.concatenate([in_lat, in_label], axis=1)

    ge1 = layers.Dense(128 * pic_dim, use_bias=True)(merge)
    ge1 = layers.ReLU()(ge1)

    ge2 = layers.Dense(pic_size * pic_size * pic_dim, use_bias=True, activation="tanh")(ge1)
    out_layer = layers.Reshape((pic_size, pic_size, pic_dim))(ge2)

    model = models.Model([in_lat, in_label], out_layer)

    return model


def make_discriminator_model_FCC():
    # INPUT: Label
    in_label = layers.Input(shape=(COND_num_classes,))

    # INPUT: Image
    in_image = layers.Input(shape=(pic_size, pic_size, pic_dim))
    in_image_b = layers.Flatten()(in_image)

    # MERGE
    merge = layers.concatenate([in_image_b, in_label], axis=1)

    ge1 = layers.Dense(128 * pic_dim, use_bias=True)(merge)
    ge1 = layers.ReLU()(ge1)

    out_layer = layers.Dense(1, use_bias=True)(ge1)

    model = models.Model([in_image, in_label], out_layer)

    return model


'''
Initiate and test models
'''

# 生成网络
generator = make_generator_model_FCC()
generator.summary()
# 判别网络
discriminator = make_discriminator_model_FCC()
discriminator.summary()


'''
Loss and Updates

Please note that, during the training step of the Discriminator train_step_DISC,
we combine gradients from both real and generated on a single update step into sanitized_grads_and_vars,
following the approach from Torkzadehmahani et al. 2019.
When learning from the real/training dataset we clip and add noise to the gradients of the Discriminator.
When learning from the generated data we only clip the gradients of the Discriminator.
'''
cross_entropy_DISC = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
cross_entropy_GEN = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step_DISC(images, labels, noise, labels_to_gen):
    with tf.GradientTape(persistent=True) as disc_tape_real:
        # This dummy call is needed to obtain the var list.
        dummy = discriminator([images, labels], training=True)
        var_list = discriminator.trainable_variables

        # In Eager mode, the optimizer takes a function that returns the loss.
        def loss_fn_real():
            real_output = discriminator([images, labels], training=True)
            disc_real_loss = cross_entropy_DISC(tf.ones_like(real_output), real_output)
            return disc_real_loss

        grads_and_vars_real = discriminator_optimizer.compute_gradients(loss_fn_real,
                                                                        var_list,
                                                                        gradient_tape=disc_tape_real,
                                                                        curr_noise_mult=NOISE_MULT,
                                                                        curr_norm_clip=NORM_CLIP)

        # In Eager mode, the optimizer takes a function that returns the loss.
        def loss_fn_fake():
            generated_images = generator([noise, labels_to_gen], training=True)
            fake_output = discriminator([generated_images, labels_to_gen], training=True)
            disc_fake_loss = cross_entropy_DISC(tf.zeros_like(fake_output), fake_output)
            return disc_fake_loss

        grads_and_vars_fake = discriminator_optimizer.compute_gradients(loss_fn_fake,
                                                                        var_list,
                                                                        gradient_tape=disc_tape_real,
                                                                        curr_noise_mult=0,
                                                                        curr_norm_clip=NORM_CLIP)
        disc_loss_r = loss_fn_real()
        disc_loss_f = loss_fn_fake()

        s_grads_and_vars = [(grads_and_vars_real[idx] + grads_and_vars_fake[idx])
                            for idx in range(len(grads_and_vars_real))]
        sanitized_grads_and_vars = list(zip(s_grads_and_vars, var_list))

        discriminator_optimizer.apply_gradients(sanitized_grads_and_vars)

    return (disc_loss_r, disc_loss_f)


# Notice the use of `tf.function`: This annotation causes the function to be "compiled".
@tf.function
def train_step_GEN(labels, noise):
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noise, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)
        # if the generator is performing well, the discriminator will classify the fake images as real (or 1)
        gen_loss = cross_entropy_GEN(tf.ones_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return (gen_loss)


'''
Train function definition

The Generator receives labels as input, in addition to noise, but since the labels are considered sensitive,
as part of the training data, the Generator will NOT see/receive them.
In this sense, we get uniform random samples of the possible labels to pass to the Generator.
Therefore, we do NOT use DP-SGD on the Generator, since only the Discriminator trains using the sensitive training data.
'''


def train(dataset, title, verbose):
    for epoch in range(EPOCHS):
        start = time.time()

        i_gen = 0
        for image_batch, label_batch in dataset:
            if verbose:
                print("Iteration: " + str(i_gen + 1))

            noise = tf.random.normal([BATCH_SIZE, Z_DIM])
            labels_to_gen = _random_choice(labels_gen_vec, BATCH_SIZE)

            d_loss_r, d_loss_f = train_step_DISC(image_batch, label_batch, noise, labels_to_gen)
            if verbose:
                print("Loss DISC Real: " + str(tf.reduce_mean(d_loss_r)))
                print("Loss DISC Fake: " + str(tf.reduce_mean(d_loss_f)))

            if (i_gen + 1) % N_DISC == 0:
                g_loss_f = train_step_GEN(labels_to_gen, noise)
                if verbose:
                    print("Loss GEN Fake:: " + str(g_loss_f))

            i_gen = i_gen + 1

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(title,
                                 generator,
                                 epoch + 1,
                                 seed,
                                 seed_labels)

        if isTest:
            images_GEN = generator([noise_GEN, labels_GEN], training=False)
            images_flat = layers.Flatten()(images_GEN)

            X_train = images_flat
            Y_train = labels_flat[:images_flat.shape[0]]

            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            Y_train_org = label_binarize(Y_train, classes=classes)
            Y_train_vec = layers.Flatten()(Y_train_org)

            X_test = layers.Flatten()(X_test_org)
            Y_test = layers.Flatten()(Y_test_org)

            # Vanilla Neural Network
            # tf.random.set_seed(1)
            classifier_NN = OneVsRestClassifier(MLPClassifier(random_state=2, alpha=1))
            NN_model2 = classifier_NN.fit(X_train, Y_train)
            # ROC per class
            Y_score = NN_model2.predict_proba(X_test)
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(np.array(Y_test), Y_score)
            res_MLP = [str(au) + " = " + str(roc_auc[au]) for au in roc_auc]
            # Logistic Regression
            # tf.random.set_seed(1)
            classifier_LR = OneVsRestClassifier(LogisticRegression(solver='lbfgs',
                                                                   multi_class='multinomial',
                                                                   random_state=2))
            LR_model2 = classifier_LR.fit(X_train, Y_train)
            # ROC per class
            Y_score = LR_model2.predict_proba(X_test)
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(np.array(Y_test), Y_score)
            res_LR = [str(au) + " = " + str(roc_auc[au]) for au in roc_auc]

            fp = open(images_dir + "/test_result.txt", 'a+')
            fp.write("Epoch {}\n".format(epoch + 1))
            fp.write("MLP Result: " + str(res_MLP) + '\n')
            fp.write("Logistic Regression: " + str(res_LR) + '\n')
            fp.close()
            del classifier_NN, classifier_LR

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Save the model
        if (EPOCHS + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_name(title + "__epoch=" + str(epoch) + "__"))


'''
Parameters
Specific parameters due to DP-SGD:
NR_MICROBATCHES (microbatches - int):
     Each batch of data (of size BATCH_SIZE) is split into smaller units called microbatches.
     So naturally NR_MICROBATCHES should evenly divide BATCH_SIZE.
     If NR_MICROBATCHES = BATCH_SIZE then every training example is a microbatch,
     clipped individually and with noise added to the average. As NR_MICROBATCHES decreases,
     we have more examples in a single microbatch,
     where averaged microbatches are clipped and noise is added to the average of averaged microbatches.
NORM_CLIP (l2_norm_clip - float) -
    The maximum Euclidean (L2) norm of each individual (or microbatch) gradient.
    To enforce such maximum norm gradients are clipped, which bounds the optimizer's sensitivity to individual training data.
NOISE_MULT (noise_multiplier - float) -
    The amount of noise sampled and added to gradients during training.
    Generally, more noise gives better privacy, which often, but not necessarily, lowers utility.
Please have in mind that the actual noise added in practice
is sampled from a Gaussian distribution with mean zero and standard deviation NORM_CLIP * NOISE_MULT.
Therefore, a larger NORM_CLIP may pass more signal from the data via gradients,
 but it also increases the noise added to the gradients.
TF Privacy's authors have already pointed out that setting NR_MICROBATCHES trades off performance
(e.g. NR_MICROBATCHES = 1) with utility (e.g. NR_MICROBATCHES = BATCH_SIZE).
DP_DELTA: Delta from the DP definition. We emphasize that DP_DELTA needs to be smaller than 1/BUFFER_SIZE.
'''

# Learning Rate for DISCRIMINATOR
LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=learning_rate_initial,
                                              global_step=tf.compat.v1.train.get_or_create_global_step(),
                                              decay_steps=decay_steps,
                                              end_learning_rate=learning_rate_end,
                                              power=1)
if BATCH_SIZE % NR_MICROBATCHES != 0:
    raise ValueError('Batch size should be an integer multiple of the number of microbatches')

'''
Get DP epsilon from parameters

Instead of updating and consulting the moments accountant on each step of training,
we just previously check the epsilon we obtain from the given parameters.
Therefore, we can just quickly keep manually adjusting the parameters above to reach our desired epsilon below,
and avoid extra computation during training.
Moreover, this allows a better understanding of the privacy implications of each parameter above
'''

# Obtain DP_EPSILON
compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=BUFFER_SIZE,
                                              batch_size=BATCH_SIZE,
                                              noise_multiplier=NOISE_MULT,
                                              epochs=EPOCHS,
                                              delta=DP_DELTA)

generator_optimizer = tf.keras.optimizers.Adam()

discriminator_optimizer = DPGradientDescentGaussianOptimizer_NEW(
    learning_rate=LR_DISC,
    l2_norm_clip=NORM_CLIP,
    noise_multiplier=NOISE_MULT,
    num_microbatches=NR_MICROBATCHES)

'''
TRAINING
We emphasize here that when batching our training dataset, DP requires random shuffling.
To help track the progress of our GAN, we fix some seeds for labels and noise for the generator,
and constantly plot the generated images. Below we create one seed for each of the 10 classes on MNIST.
'''

# 训练
# Create/reinitiate models
generator = make_generator_model_FCC()
discriminator = make_discriminator_model_FCC()
# Create checkpoint structure
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

tf.random.set_seed(1)

# Batch and random shuffle training data
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels_vec)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Fix some seeds to help visualize progress
seed = tf.random.normal([10, Z_DIM])
seed_labels = tf.Variable(np.diag(np.full(10, 1)).reshape((10, 10)), dtype='float32')

# To be used for sampling random labels to pass to generator
labels_gen_vec = np.zeros((10, COND_num_classes), dtype='float32')
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    labels_gen_vec[i, int(i)] = 1.0

# GIVES CURRENT TRIAL A NAME - Suggestion: from parameters used
training_title = 'eps9.6'

# 生成图片
N_GEN = 5000
N_GEN_per_CLASS = np.int(N_GEN / 10)

tf.random.set_seed(10)
noise_GEN = tf.random.normal([N_GEN, Z_DIM])
labels_GEN = tf.Variable(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] * N_GEN_per_CLASS +
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * N_GEN_per_CLASS,
                                  dtype='float32').reshape((N_GEN, 10)))
labels_flat = tf.Variable(np.array([0] * N_GEN_per_CLASS +
                                   [1] * N_GEN_per_CLASS +
                                   [2] * N_GEN_per_CLASS +
                                   [3] * N_GEN_per_CLASS +
                                   [4] * N_GEN_per_CLASS +
                                   [5] * N_GEN_per_CLASS +
                                   [6] * N_GEN_per_CLASS +
                                   [7] * N_GEN_per_CLASS +
                                   [8] * N_GEN_per_CLASS +
                                   [9] * N_GEN_per_CLASS,
                                   dtype='float32').reshape((N_GEN, 1)))

# STARTS TRAINING
if isTrain:
    train(train_dataset, training_title, False)

if isLoad:
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                     save_weights_only=True,
                                                     verbose=1)
    print(cp_callback)

images_GEN = generator([noise_GEN, labels_GEN], training=False)
images_flat = layers.Flatten()(images_GEN)
Y_train = labels_flat[:images_flat.shape[0]]
X_train = images_flat

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_train_org = label_binarize(Y_train, classes=classes)
Y_train_vec = layers.Flatten()(Y_train_org)

X_test = layers.Flatten()(X_test_org)
Y_test = layers.Flatten()(Y_test_org)

# Vanilla Neural Network
tf.random.set_seed(1)
classifier_NN = OneVsRestClassifier(MLPClassifier(random_state=2, alpha=1))
NN_model2 = classifier_NN.fit(X_train, Y_train)
# ROC per class
Y_score = NN_model2.predict_proba(X_test)
false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(np.array(Y_test), Y_score)
res_MLP = [str(au) + " = " + str(roc_auc[au]) for au in roc_auc]
# Logistic Regression
tf.random.set_seed(1)
classifier_LR = OneVsRestClassifier(LogisticRegression(solver='lbfgs',
                                                       multi_class='multinomial',
                                                       random_state=2))
LR_model2 = classifier_LR.fit(X_train, Y_train)
# ROC per class
Y_score = LR_model2.predict_proba(X_test)
false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(np.array(Y_test), Y_score)
res_LR = [str(au) + " = " + str(roc_auc[au]) for au in roc_auc]

fp = open(images_dir + "/test_result.txt", 'a+')
fp.write("MLP Result: " + str(res_MLP) + '\n')
fp.write("Logistic Regression: " + str(res_LR))
fp.close()
