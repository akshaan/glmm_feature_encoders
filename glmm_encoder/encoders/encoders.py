"""Base class for encoders"""
from typing import Any, Dict, List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from enum import Enum
import math

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tfp.layers

class TaskTypes(Enum):
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MUTLICLASS_CLASSIFICATION = 3


class BaseGLMMSingleTargetEncoder(tf.keras.Model):
    def __init__(self, num_levels: int, task_type: TaskTypes, seed: float = None):
        super().__init__()
        self.num_levels = num_levels
        self.__task_type = task_type
        self.surrogate_posterior = self.make_surrogate_posterior()
        self.seed = seed

    def make_joint_distribution_coroutine(self, feature_vals: tf.Tensor) -> tfd.Distribution:
        def model():
            level_scale = yield tfd.Uniform(low=0., high=2., name='scale_prior')
            intercept = yield tfd.Normal(loc=0., scale=1, name='intercept')
            level_prior = yield tfd.Normal(loc=tf.zeros(self.num_levels),
                                           scale=level_scale,
                                           name='level_prior')
            random_effect = tf.gather(level_prior, feature_vals, axis=-1)
            fixed_effect = intercept
            response = fixed_effect + random_effect
            if self.__task_type in [TaskTypes.BINARY_CLASSIFICATION, TaskTypes.MUTLICLASS_CLASSIFICATION]:
                yield tfd.Bernoulli(logits=response, name='likelihood')
            elif self.__task_type == TaskTypes.REGRESSION:
                yield tfd.Normal(loc=response, scale=1., name='likelihood')
            else:
                raise ValueError("Invalid TaskType."
                                 "Must be one of [TaskTypes.REGRESSION, TaskTypes.BINARY_CLASSIFICATION")

        return tfd.JointDistributionCoroutineAutoBatched(model)

    def make_surrogate_posterior(self) -> tfd.Distribution:
        _init_loc = lambda shape=(): tf.Variable(
            tf.random.uniform(shape, minval=-2., maxval=2.))
        _init_scale = lambda shape=(): tfp.util.TransformedVariable(
            initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
            bijector=tfb.Softplus())
        return tfd.JointDistributionSequentialAutoBatched([
            tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),  # scale_prior
            tfd.Normal(_init_loc(), _init_scale()),  # intercept
            tfd.Normal(_init_loc([self.num_levels]), _init_scale([self.num_levels]))])  # level_prior

    def call(self, feature_vals: tf.Tensor, training: bool = None, mask: Any = None) -> tf.Tensor:
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

    def print_posterior_estimates(self) -> None:
        model = self.surrogate_posterior.model
        intercept_estimate = model[1].mode()
        print(f"Intercept estimate = {intercept_estimate}")

        def softplus(x): return math.log1p(math.exp(-abs(x))) + max(x, 0)

        print(f"Random effect variance estimate = {softplus(model[0].distribution.mean().numpy())}")

    def train_step(self, data) -> Dict[str, tf.Tensor]:
        x, y = data
        joint = self.make_joint_distribution_coroutine(x)

        def target_log_prob_fn(*args):
            return joint.log_prob(*args, likelihood=y)

        loss = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            self.surrogate_posterior,
            optimizer=self.optimizer,
            num_steps=1,
            seed=42,
            sample_size=5
        )

        return {"loss": loss}


class GLMMRegressionTargetEncoder(BaseGLMMSingleTargetEncoder):
    def __init__(self, num_levels: int, seed: float = None):
        super().__init__(num_levels, task_type=TaskTypes.REGRESSION, seed=seed)


class GLMMBinaryTargetEncoder(BaseGLMMSingleTargetEncoder):
    def __init__(self, num_levels: int, seed: float = None):
        super().__init__(num_levels, task_type=TaskTypes.BINARY_CLASSIFICATION, seed=seed)


class GLMMMulticlassTargetEncoder(tf.keras.Model):
    def __init__(self, num_levels: int, num_classes: int, seed: float = None):
        super().__init__()
        self.num_levels = num_levels
        self.__task_type = TaskTypes.MUTLICLASS_CLASSIFICATION
        self.num_classes = num_classes
        self.seed = seed
        self.models = [GLMMBinaryTargetEncoder(num_levels) for _ in range(num_classes)]

    def call(self, feature_vals: tf.Tensor, training: bool = None, mask: Any = None):
        return tf.stack(
            [m(feature_vals) for m in self.models],
            axis=-1
        )

    def compile(self, optimizer: Any = 'rmsprop', loss: Any = None, metrics: Any = None, loss_weights: Any = None,
                weighted_metrics: Any = None, run_eagerly: Any = None, steps_per_execution: Any = None, **kwargs
                ):
        for model in self.models:
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                **kwargs
            )
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs
        )

    @staticmethod
    def __merge_metrics(metrics_dict_list: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        merged_metrics = {}
        key_to_merge_func = {"loss": tf.reduce_sum}
        for key, merge_func in key_to_merge_func.items():
            metric = tf.concat([x[key] for x in metrics_dict_list], axis=-1)
            merged_metrics[key] = merge_func(metric)
        return merged_metrics

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        x, y = data
        per_class_metrics_dicts = []
        for c in range(self.num_classes):
            y_class = tf.where(y == c, x=1, y=0)
            per_class_metrics_dicts.append(self.models[c].train_step((x, y_class)))

        return self.__merge_metrics(per_class_metrics_dicts)

    def print_posterior_estimates(self) -> None:
        for i, class_model in enumerate(self.models):
            print(f"Sub-model for class {i}")
            model = class_model.surrogate_posterior.model
            intercept_estimate = model[1].mode()
            print(f"\tIntercept estimate = {intercept_estimate}")

            def softplus(x): return math.log1p(math.exp(-abs(x))) + max(x, 0)

            print(f"\tRandom effect variance estimate = {softplus(model[0].distribution.mean().numpy())}")
