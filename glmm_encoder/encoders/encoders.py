"""Base class for encoders"""

import tensorflow as tf
import tensorflow_probability as tfp
from enum import Enum
import math

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers


class TaskTypes(Enum):
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2


class BaseGLMMTargetEncoder():
    def __init__(self, num_levels, task_type):
        super().__init__()
        self.num_levels = num_levels
        self.task_type = task_type

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
            if self.task_type == TaskTypes.BINARY_CLASSIFICATION:
                yield tfd.Bernoulli(logits=response, name='likelihood')
            elif self.task_type == TaskTypes.REGRESSION:
                yield tfd.Normal(loc=response, scale=1., name='likelihood')
            else:
                raise ValueError("Invalid TaskType."
                                 "Must be one of [TaskTypes.REGRESSION, TaskTypes.BINARY_CLASSIFICATION")

        return tfd.JointDistributionCoroutineAutoBatched(model)

    def make_surrogate_posterior(self, joint):
        _init_loc = lambda shape=(): tf.Variable(
            tf.random.uniform(shape, minval=-2., maxval=2.))
        _init_scale = lambda shape=(): tfp.util.TransformedVariable(
            initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
            bijector=tfb.Softplus())
        return tfd.JointDistributionSequentialAutoBatched([
            tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),  # scale_prior
            tfd.Normal(_init_loc(), _init_scale()),  # intercept
            tfd.Normal(_init_loc([self.num_levels]), _init_scale([self.num_levels]))])  # level_prior

    def fit(self, feature_vals, target_vals):
        joint = self.make_joint_distribution_coroutine(feature_vals)
        self.surrogate_posterior = self.make_surrogate_posterior(joint)

        def target_log_prob_fn(*args):
            return joint.log_prob(*args, likelihood=target_vals)

        optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        return tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            self.surrogate_posterior,
            optimizer=optimizer,
            num_steps=10000,
            seed=42,
            sample_size=5)

    def predict(self, feature_vals):
        # N.b. this method does not account for feature levels that don't appear in the training set
        # For those features, this method needs to be modified to return the global intercept
        model = self.surrogate_posterior.model
        intercept_estimate = model[1].mode()
        random_effect_estimate = model[2].sample(feature_vals.shape)
        print(f"Intercept estimate = {intercept_estimate}")
        def softplus(x): return math.log1p(math.exp(-abs(x))) + max(x, 0)
        print(f"Random effect stddev estimate = {softplus(model[0].distribution.mean())}")
        return tf.gather(random_effect_estimate, feature_vals, axis=-1) + intercept_estimate


class GLMMRegressionTargetEncoder(BaseGLMMTargetEncoder):
    def __init__(self, num_levels):
        super().__init__(num_levels, TaskTypes.REGRESSION)


class GLMMClassificationTargetEncoder(BaseGLMMTargetEncoder):
    def __init__(self, num_levels):
        super().__init__(num_levels, TaskTypes.BINARY_CLASSIFICATION)
