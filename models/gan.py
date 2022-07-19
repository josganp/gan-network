"""Generative Adversarial Network for Training the Discriminator

"""

import numpy as np
import tensorflow as tf
import json
import csv

import utils.datasets as ds
import models.nnblocks as nn
import utils.tensorboard as tb


class GAN:
    """Anomaly detection object using a GAN

    Args:
        num_features (int): Number of input for classifier.
        batch_size (int = 100): Batch size.
        num_epochs (int = 10): Number of training epochs.
        debug (bool = False): Flag to print output.
        normalize (str = 'rescaling'): Normalization mode
            from the set {'rescaling', 'vector_norm', 'none'}
        display_step (int = 1): How often to debug epoch data
            during training.
        num_steps (int = 3): Number of time steps.
        latent_vector_size (int = 100): Size of the latent vector.
        adpt_l (float = 2): Parameter for adaptive learning of G and D.
        ns_param (float = 0.01): Node similarity parameter.
        batch_param (float = 0.1): Batch deviation parameter.
        dr_param (float = 1.): Discriminator real loss parameter.
        df_param (float = 1.): Discriminator fake loss parameter.
        res_depth (int = 1): Number of residual blocks per layer.
        learning_rate (float = 0.001): Learning rate.
        reg_param (float = 0.01): L2 Regularization parameter.
    """

    def __init__(self, num_features, **kwargs):
        defaults = {
            'num_epochs': 10,
            'display_step': 1,
            'batch_size': 100,
            'num_steps': 1,
            'debug': False,
            'normalize': 'rescaling',
            'latent_vector_size': 100,
            'adpt_l': 2.0,
            'res_depth': 1,
            'ns_param': 0.01,
            'batch_param': 0.1,
            'dr_param': 1.,
            'df_param': 1.,
            'd_learning_rate': .001,
            'g_learning_rate': .001,
            'reg_param': 0.01,
            'vector_norm': False
        }

        self.num_features = num_features

        vars(self).update({p: kwargs.get(p, d) for p, d in defaults.items()})

        ########################################
        # TensorFlow Variables                 #
        ########################################
        tf.compat.v1.disable_eager_execution()

        self.X = tf.compat.v1.placeholder(
            'float32',
            [None, num_features * self.num_steps],
            name='X'
        )

        self.Y = tf.compat.v1.placeholder('int64', [None], name='Y')

        self.Z = tf.compat.v1.placeholder(
            'float32',
            [None, self.latent_vector_size],
            name='Z'
        )

        self.keep_prob = tf.compat.v1.placeholder('float32', name='keep_prob')

        # for normalization
        self.feature_min = tf.Variable(
            np.zeros(num_features * self.num_steps),
            dtype=tf.float32
        )

        self.feature_max = tf.Variable(
            np.zeros(num_features * self.num_steps),
            dtype=tf.float32
        )

        ########################################
        # GAN Model                            #
        ########################################

        self.embedding_ops = []

        # def build_net(X, sizes):
        #     lrelu = nn.lrelu_gen(0.1)

        #     def block(x, in_dim, out_dim, i):
        #         with tf.variable_scope('block_{}'.format(i)):
        #             z = x
        #             for j in range(self.res_depth):
        #                 with tf.variable_scope('res_block_{}'.format(j)):
        #                     z = nn.build_residual_block(
        #                         z, lrelu, in_dim, self.reg_param
        #                     )
        #                     with tf.variable_scope('residual_block'):
        #                         self.embedding_ops.append(z)
        #                     z = tf.nn.dropout(z, self.keep_prob)

        #             z = nn.build_fc_layer(
        #                 z, lrelu, in_dim, out_dim, self.reg_param
        #             )

        #             with tf.variable_scope('fc_block'):
        #                 self.embedding_ops.append(z)

        #             if i < len(sizes) - 2:
        #                 z = tf.nn.dropout(z, self.keep_prob)

        #             return z
        #     z = X

        #     for i in range(1, len(sizes)):
        #         z = block(z, sizes[i-1], sizes[i], i-1)

        #     return z

        def build_net(x, sizes):
            lrelu = nn.lrelu_gen(0.1)
            sizes = [[sizes[i], sizes[i+1]] for i in range(len(sizes)-1)]

            def build_block(x, _in, _out):
                z = x

                for i in range(self.res_depth):
                    with tf.compat.v1.variable_scope('fc_layer_{}'.format(i)):
                        f = tf.nn.dropout(
                            nn.build_fc_layer(
                                z, lrelu, _in, _in, self.reg_param
                            ),
                            rate = 1 - (self.keep_prob)
                        )

                        z = f + z

                return tf.nn.dropout(
                    nn.build_fc_layer(
                        z, lrelu, _in, _out, self.reg_param
                    ),
                    rate = 1 - (self.keep_prob)
                )

            z = x
            i = 0

            for _in, _out in sizes:
                with tf.compat.v1.variable_scope('res_block_{}'.format(i)):
                    z = build_block(z, _in, _out)
                    i += 1

            return z

        vec_size = self.num_features * self.num_steps

        g_sizes = [self.latent_vector_size, 100, vec_size]
        d_sizes = [vec_size, 18, 9, 3, 2]

        # g_sizes = [self.latent_vector_size, vec_size]
        # d_sizes = [vec_size, 18, 9, 3, 2]

        with tf.compat.v1.variable_scope('generator'):
            G_sample = tf.nn.tanh(build_net(self.Z, g_sizes))

        with tf.compat.v1.variable_scope('discriminator'):
            D_logit_real = build_net(self.X, d_sizes)
            tf.compat.v1.get_variable_scope().reuse_variables()
            D_logit_fake = build_net(G_sample, d_sizes)

        D_fake = tf.nn.sigmoid(D_logit_fake)
        D_real = tf.nn.sigmoid(D_logit_real)

        # self.scores = tf.nn.sigmoid(D_logit_real)
        # self.scores = tf.squeeze(D_real)
        self.scores = D_logit_real

        ########################################
        # Losses & Optimizers                  #
        ########################################

        # D Loss
        D_loss_real = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=D_logit_real, labels=tf.ones_like(self.Y)
            )
        )

        D_loss_fake = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.zeros_like(self.Y)
            )
        )

        with tf.compat.v1.name_scope('D_loss_real'):
            tb.variable_summaries(D_loss_real)

        with tf.compat.v1.name_scope('D_loss_fake'):
            tb.variable_summaries(D_loss_fake)

        # Ensure differnce between nodes.
        # node_simiality_loss = -tf.reduce_mean(tf.add(
        #     tf.square(D_real[:, 1] - D_real[:, 0]),
        #     tf.square(D_fake[:, 1] - D_fake[:, 0])
        # ))

        node_simiality_loss = -tf.reduce_mean(input_tensor=tf.add(
            tf.nn.moments(x=D_real, axes=[1])[1],
            tf.nn.moments(x=D_fake, axes=[1])[1]
        ))

        # Punish stddev accross batch
        batch_loss = tf.reduce_mean(input_tensor=tf.add(
            tf.nn.moments(x=D_real, axes=[0])[1],
            tf.nn.moments(x=D_fake, axes=[0])[1]
        ))

        self.D_loss = tf.add_n([
            self.df_param * D_loss_fake,
            self.dr_param * D_loss_real,
            self.ns_param * node_simiality_loss,
            self.batch_param * batch_loss
        ])

        with tf.compat.v1.name_scope('D_loss'):
            tb.variable_summaries(self.D_loss)

        with tf.compat.v1.name_scope('batch_loss'):
            tb.variable_summaries(batch_loss)

        with tf.compat.v1.name_scope('node_simiality_loss'):
            tb.variable_summaries(node_simiality_loss)

        self.D_only_loss = tf.add_n([
            self.df_param * D_loss_fake,
            self.dr_param * D_loss_real
        ])

        self.D_loss += tf.add_n(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES,
            scope='discriminator'
        ))

        # G Loss
        self.G_loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=D_logit_fake, labels=tf.ones_like(self.Y)
            )
        )

        with tf.compat.v1.name_scope('G_loss'):
            tb.variable_summaries(self.G_loss)

        self.G_loss += tf.add_n(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES,
            scope='generator'
        ))

        # Optimizers
        self.D_solver = tf.compat.v1.train.AdamOptimizer(self.d_learning_rate).minimize(
            self.D_loss,
            var_list=tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope='discriminator'
            )
        )

        self.G_solver = tf.compat.v1.train.AdamOptimizer(self.g_learning_rate).minimize(
            self.G_loss,
            var_list=tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope='generator'
            )
        )

        ########################################
        # Evaluation Metrics                   #
        ########################################

        # negative_labels = tf.cast(tf.fill(tf.shape(self.Y), 0), 'int64')
        # positive_labels = tf.cast(tf.fill(tf.shape(self.Y), 1), 'int64')

        # pred_labels = tf.where(
        #     tf.greater(self.scores, tf.fill(tf.shape(self.Y), 0.5)),
        #     positive_labels,
        #     negative_labels

        # )

        pred_labels = tf.argmax(input=self.scores, axis=1)

        self.confusion_matrix = tf.math.confusion_matrix(
            labels=self.Y,
            predictions=pred_labels,
            num_classes=2
        )

        self.accuracy = tf.reduce_mean(
            input_tensor=tf.cast(tf.equal(pred_labels, self.Y), dtype=tf.float32)
        )

        # Variable ops
        self.init_op = tf.compat.v1.global_variables_initializer()
        self.saver = tf.compat.v1.train.Saver()
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.merged = tf.compat.v1.summary.merge_all()

    def train(self, X, Y):
        """Train the Classifier.

        Args:
            X (np.ndarray): Features with shape
                (num_samples * time_steps, features).
            Y (np.ndarray): Labels.
        """

        training_size = X.shape[0]

        # normalize X
        if self.normalize == 'rescaling':
            _min = X.min(axis=0)
            _max = X.max(axis=0)
            X = ds.rescale(X, _min, _max, -1, 1)

        elif self.normalize == 'vector_norm':
            X = ds.vector_norm(X, -1, 1)

        assert self.batch_size < training_size, (
            'batch size is larger than training_size'
        )

        with tf.compat.v1.Session(config=self.config) as sess:
            sess.run(self.init_op)

            # for tensorboard
            writer = tf.compat.v1.summary.FileWriter(
                logdir='logdir/train',
                graph=sess.graph
            )

            prev_diff_loss = 0

            batch = ds.random_batcher([X, Y], self.batch_size)

            count = 0

            for epoch in range(self.num_epochs):
                d_loss = 0
                g_loss = 0

                k = self.adpt_l * prev_diff_loss
                kd, kg = np.maximum([1, 1], [k, -k]).astype(np.int32)

                for i in range(kd):
                    batch_x, batch_y = next(batch)
                    Z = self.sample_Z(n=batch_x.shape[0])

                    s, _, ld = sess.run(
                        [self.merged, self.D_solver, self.D_only_loss],
                        feed_dict={
                            self.X: batch_x,
                            self.Y: batch_y,
                            self.Z: Z,
                            self.keep_prob: 0.5
                        }
                    )

                    writer.add_summary(s, count)
                    count += 1

                    d_loss += ld

                for i in range(kg):
                    batch_x, batch_y = next(batch)
                    Z = self.sample_Z(n=batch_x.shape[0])

                    s, _, lg = sess.run(
                        [self.merged, self.G_solver, self.G_loss],
                        feed_dict={
                            self.X: batch_x,
                            self.Z: Z,
                            self.Y: batch_y,
                            self.keep_prob: 0.5
                        }
                    )

                    writer.add_summary(s, count)
                    count += 1

                    g_loss += lg

                prev_diff_loss = ld - lg

                if epoch % self.display_step == 0:
                    display_str = (
                        'Epoch {0:04} with D_loss={1:7.5f}||G_loss={2:.5f}'
                    )
                    display_str += '\nkd={3}, kg={4}'
                    display_str = display_str.format(
                        epoch+1,
                        d_loss/kd,
                        g_loss/kg,
                        kd, kg
                    )
                    self.print(display_str)

            # assign normalization values
            if self.normalize == 'rescaling':
                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            self.print('Optimization Finished')

            # save model
            save_path = self.saver.save(sess, './model.ckpt')
            self.print('Model saved in file: {}'.format(save_path))

    def test(self, X, Y):
        """Tests classifier

        Args:
            X (np.ndarray): Features with shape
                (num_samples * time_steps, features).
            Y (np.array): Labels.

        Returns:
            dict: Dictionary containing the following fields:
        """

        with tf.compat.v1.Session(config=self.config) as sess:
            self.saver.restore(sess, './model.ckpt')

            # normalize data
            if self.normalize == 'rescaling':
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()
                X = ds.rescale(X, _min, _max, -1, 1)

            elif self.normalize == 'vector_norm':
                X = ds.vector_norm(X, -1, 1)

            labels, acc, mat, d_loss, g_loss = sess.run(
                [self.scores, self.accuracy, self.confusion_matrix,
                 self.D_loss, self.G_loss],
                feed_dict={
                    self.X: X,
                    self.Y: Y,
                    self.Z: self.sample_Z(n=X.shape[0]),
                    self.keep_prob: 1.0
                }
            )

            avg_benign      = []
            avg_malicious   = []
            for i, label in enumerate(labels):
                if Y[i] == 1:
                    avg_benign.append(label)
                else:
                    avg_malicious.append(label)

            data = {
                'benign': {
                    'mean': np.mean(avg_benign, axis=0).tolist(),
                    'stddev': np.std(avg_benign, axis=0).tolist()
                },
                'malicious': {
                    'mean': np.mean(avg_malicious, axis=0).tolist(),
                    'stddev': np.std(avg_malicious, axis=0).tolist()
                }
            }

            data['confusion_matrix'] = mat.tolist()
            data['accuracy'] = acc * 100
            data['d_loss'] = float(d_loss)
            data['g_loss'] = float(g_loss)

            self.print(json.dumps(data, indent=4))

            # Embedddings
            Z = self.sample_Z(n=X.shape[0])
            embeddings = sess.run(self.embedding_ops, feed_dict={
                self.X: X,
                self.Y: Y,
                self.Z: Z,
                self.keep_prob: 1.0
            })

            for i, embedding in enumerate(embeddings):
                name = self.embedding_ops[i].name.split(':')[0]
                name = name.replace('/', '_')

                with open('graph/{}'.format(name), 'w') as f:
                    csv.writer(f).writerows(embedding)

            return data

    def sample_Z(self, n):
        return np.random.normal(size=(n, self.latent_vector_size))

    def add_noise(self, x):
        """Adds noise to an input tensor.

        Args:
            x (np.array): Input tensor (2d array).

        Returns:
            np.array: x + noise
        """

        std = np.std(x, axis=0)

        return x + np.random.normal(scale=std, size=x.shape)

    def print(self, val):
        if self.debug:
            print(val)
