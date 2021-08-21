"""Base class for encoders"""

import tensorflow as tf
import tensorflow_probability as tfp
from enum import Enum
import math
from abc import ABC, abstractmethod

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers


class TaskTypes(Enum):
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MUTLICLASS_CLASSIFICATION = 3


class BaseGLMMTargetEncoder(tf.keras.Model):
    def __init__(self, num_levels, task_type, seed=None):
        super(BaseGLMMTargetEncoder, self).__init__()
        self.num_levels = num_levels
        self.task_type = task_type
        if task_type in [TaskTypes.BINARY_CLASSIFICATION, TaskTypes.REGRESSION]:
            self.surrogate_posterior = self.make_surrogate_posterior()
        self.seed = seed

    def make_joint_distribution_coroutine(self, feature_vals):
        def model():
            level_scale = yield tfd.Uniform(low=0., high=2., name='scale_prior')
            intercept = yield tfd.Normal(loc=0., scale=1, name='intercept')
            level_prior = yield tfd.Normal(loc=tf.zeros(self.num_levels),
                                           scale=level_scale,
                                           name='level_prior')
            random_effect = tf.gather(level_prior, feature_vals, axis=-1)
            fixed_effect = intercept
            response = fixed_effect + random_effect
            if self.task_type in [TaskTypes.BINARY_CLASSIFICATION, TaskTypes.MUTLICLASS_CLASSIFICATION]:
                yield tfd.Bernoulli(logits=response, name='likelihood')
            elif self.task_type == TaskTypes.REGRESSION:
                yield tfd.Normal(loc=response, scale=1., name='likelihood')
            else:
                raise ValueError("Invalid TaskType."
                                 "Must be one of [TaskTypes.REGRESSION, TaskTypes.BINARY_CLASSIFICATION")

        return tfd.JointDistributionCoroutineAutoBatched(model)

    def make_surrogate_posterior(self):
        _init_loc = lambda shape=(): tf.Variable(
            tf.random.uniform(shape, minval=-2., maxval=2.))
        _init_scale = lambda shape=(): tfp.util.TransformedVariable(
            initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
            bijector=tfb.Softplus())
        return tfd.JointDistributionSequentialAutoBatched([
            tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),  # scale_prior
            tfd.Normal(_init_loc(), _init_scale()),  # intercept
            tfd.Normal(_init_loc([self.num_levels]), _init_scale([self.num_levels]))])  # level_prior

    def call(self, feature_vals, training=None, mask=None):
        model = self.surrogate_posterior.model
        intercept_estimate = model[1].mode()
        random_effect_estimate = model[2].mode()
        if training:
            return tf.gather(random_effect_estimate, feature_vals, axis=-1) + intercept_estimate
        else:
            # In order to accommodate new feature levels (not in train set) during prediction, we create
            # a new level (self.num_levels) and assign all unseen levels to that value. We also add a
            # corresponding 0 entry to the random_effect_estimate vector for that level.
            # This ensures that unseen levels are assigned 0 + intercept_estimate in the output
            random_effect_estimate_with_missing = tf.concat([random_effect_estimate, tf.zeros([1])], axis=-1)
            feature_vals_with_missing = tf.where(
                tf.math.logical_and(feature_vals < self.num_levels, feature_vals >= 0),
                x=feature_vals,
                y=self.num_levels
            )
        return tf.gather(random_effect_estimate_with_missing, feature_vals_with_missing, axis=-1) + intercept_estimate

    def print_posterior_estimates(self):
        model = self.surrogate_posterior.model
        intercept_estimate = model[1].mode()
        print(f"Intercept estimate = {intercept_estimate}")
        def softplus(x): return math.log1p(math.exp(-abs(x))) + max(x, 0)
        print(f"Random effect variance estimate = {softplus(model[0].distribution.mean().numpy())}")

    def train_step(self, data):
        x, y = data
        joint = self.make_joint_distribution_coroutine(x)

        def target_log_prob_fn(*args):
            return joint.log_prob(*args, likelihood=y)

        loss = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            self.surrogate_posterior,
            optimizer=self.optimizer,
            num_steps=1,
            seed=self.seed,
            sample_size=5
        )

        return {"loss": loss}


class GLMMRegressionTargetEncoder(BaseGLMMTargetEncoder):
    def __init__(self, num_levels):
        super().__init__(num_levels, TaskTypes.REGRESSION)


class GLMMBinaryTargetEncoder(BaseGLMMTargetEncoder):
    def __init__(self, num_levels):
        super().__init__(num_levels, TaskTypes.BINARY_CLASSIFICATION)