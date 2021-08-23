"""GLMM Target Encoders"""

from typing import Any, Dict, List, Tuple, Iterator, Union, Optional
from enum import Enum

import tensorflow as tf
import tensorflow_probability as tfp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class _TaskTypes(Enum):
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MUTLICLASS_CLASSIFICATION = 3


class BaseGLMMSingleTargetFeatureEncoder(tf.keras.Model):
    """Base class for single-target (regression / bin-classification) encoders"""

    def __init__(self, num_levels: int, task_type: _TaskTypes, seed: float = None):
        """Constructor

        Args:
            num_levels (int): Number of (training) levels in the feature being encoded
            task_type (_TaskTypes): Task type for model being built. Must be in
                                    [_TaskTypes.REGRESSION, _TaskTypes.BINARY_CLASSIFICATION]
            seed (float): Float seed to use when fitting posterior
        """
        super().__init__()
        self.num_levels = num_levels
        self.__task_type = task_type
        self.surrogate_posterior = self.make_surrogate_posterior()
        self.seed = seed

    def make_prior(self, feature_vals: tf.Tensor) -> tfp.distributions.Distribution:
        """Make prior distribution object

        Args:
            feature_vals (tf.Tensor): Tensor with feature values to be encoded

        Returns: Auto-batched prior distribution

        """
        def model() -> Iterator[tfp.distributions.Distribution]:
            level_scale = yield tfp.distributions.Uniform(low=0., high=2., name='scale_prior')
            intercept = yield tfp.distributions.Normal(loc=0., scale=1, name='intercept')
            level_prior = yield tfp.distributions.Normal(loc=tf.zeros(self.num_levels),
                                                         scale=level_scale,
                                                         name='level_prior')
            random_effect = tf.gather(level_prior, feature_vals, axis=-1)
            fixed_effect = intercept
            response = fixed_effect + random_effect
            if self.__task_type in [_TaskTypes.BINARY_CLASSIFICATION, _TaskTypes.MUTLICLASS_CLASSIFICATION]:
                yield tfp.distributions.Bernoulli(logits=response, name='likelihood')
            elif self.__task_type == _TaskTypes.REGRESSION:
                yield tfp.distributions.Normal(loc=response, scale=1., name='likelihood')
            else:
                raise ValueError("Invalid TaskType."
                                 "Must be one of [TaskTypes.REGRESSION, TaskTypes.BINARY_CLASSIFICATION")

        return tfp.distributions.JointDistributionCoroutineAutoBatched(model)

    def make_surrogate_posterior(self) -> tfp.distributions.Distribution:
        """Make surrogate (mean field) posterior distribution object for variational inference

        Returns: Auto-batched surrogate posterior distribution

        """
        _init_loc = lambda shape=(): tf.Variable(
            tf.random.uniform(shape, -2., 2.))
        _init_scale = lambda shape=(): tfp.util.TransformedVariable(
            initial_value=tf.random.uniform(shape, 0.01, 1.),
            bijector=tfp.bijectors.Softplus())
        return tfp.distributions.JointDistributionSequentialAutoBatched([
            tfp.bijectors.Softplus()(tfp.distributions.Normal(_init_loc(), _init_scale())),  # scale_prior
            tfp.distributions.Normal(_init_loc(), _init_scale()),  # intercept
            tfp.distributions.Normal(_init_loc([self.num_levels]), _init_scale([self.num_levels]))])  # level_prior

    def call(
            self,
            feature_vals: tf.Tensor,
            training: bool = None, mask: Any = None) -> tf.Tensor:  # pylint: disable=unused-argument
        """tf.keras.Model.call implementation. Applied the model to inputs and returns a tensor of outputs.

        Args:
            feature_vals: tf.Tensor: A tensor with input values to the model
            training (bool): Flag indicating training vs. prediction mode
            mask (tf.Tensor): Mask for inputs

        Returns: Tensor with result from applying the model to feature_vals

        """
        model = self.surrogate_posterior.model
        intercept_estimate = model[1].mean()
        random_effect_estimate = model[2].mean()

        if training:
            return tf.gather(random_effect_estimate, feature_vals, axis=-1) + intercept_estimate

        # In order to accommodate new feature levels (not in train set) during prediction, we create
        # a new level (self.num_levels) and assign all unseen levels to that value. We also add a
        # corresponding 0 entry to the random_effect_estimate vector for that level.
        # This ensures that unseen levels are assigned 0 + intercept_estimate in the output
        random_effect_estimate_with_missing = tf.concat([random_effect_estimate, tf.zeros([1])], -1)
        feature_vals_with_missing = tf.where(
            tf.math.logical_and(feature_vals < self.num_levels, feature_vals >= 0),
            x=feature_vals,
            y=self.num_levels
        )
        return tf.gather(random_effect_estimate_with_missing, feature_vals_with_missing, axis=-1) + intercept_estimate

    def print_posterior_estimates(self) -> None:
        """Debug method to print parameter posterior estimates

        """
        (scale_prior_estimate,
         intercept_estimate,
         _), _ = self.surrogate_posterior.sample_distributions()
        print(f"Intercept estimate = {intercept_estimate.mean()}")
        print(f"Random effect variance estimate = {tf.reduce_mean(scale_prior_estimate.sample(10000))}")

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """A single train step for the model

        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): Tuple of train feature tensor and train label tensor for current batch

        Returns: Dict mapping metric names to metric value tensors for the current training step

        """
        x, y = data
        joint = self.make_prior(x)

        def target_log_prob_fn(*args: tf.Tensor) -> tf.Tensor:
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


class GLMMRegressionFeatureEncoder(BaseGLMMSingleTargetFeatureEncoder):
    """GLMM feature encoder for regression tasks"""

    def __init__(self, num_levels: int, seed: float = None):
        """Constructor

        Args:
            num_levels (int): Number of (training) levels in the feature being encoded
            seed (float): Float seed to use when fitting posterior
        """
        super().__init__(num_levels, task_type=_TaskTypes.REGRESSION, seed=seed)


class GLMMBinClassifierFeatureEncoder(BaseGLMMSingleTargetFeatureEncoder):
    """GLMM feature encoder for binary classification tasks"""

    def __init__(self, num_levels: int, seed: float = None):
        """Constructor

        Args:
            num_levels (int): Number of (training) levels in the feature being encoded
            seed (float): Float seed to use when fitting posterior
        """
        super().__init__(num_levels, task_type=_TaskTypes.BINARY_CLASSIFICATION, seed=seed)


class GLMMMultiClassifierFeatureEncoder(tf.keras.Model):
    """GLMM feature encoder for multiclass classification tasks.

    Trains a GLMMBinClassifierFeatureEncoder for each class in the output and returns one encoded feature per class.
    """

    def __init__(self, num_levels: int, num_classes: int, seed: float = None):
        """Constructor

        Args:
            num_levels (int): Number of (training) levels in the feature being encoded
            num_classes (int): Number of possible output classes in training set
            seed (float): Float seed to use when fitting posterior
        """
        super().__init__()
        self.num_levels = num_levels
        self.__task_type = _TaskTypes.MUTLICLASS_CLASSIFICATION
        self.num_classes = num_classes
        self.seed = seed
        self.models = [GLMMBinClassifierFeatureEncoder(num_levels) for _ in range(num_classes)]

    def call(
            self,
            feature_vals: tf.Tensor,
            training: bool = None, mask: Any = None) -> tf.Tensor:  # pylint: disable=unused-argument
        """tf.keras.Model.call implementation. Applied the model to inputs and returns a tensor of outputs.

        Args:
            feature_vals: tf.Tensor: A tensor with input values to the model
            training (bool): Flag indicating training vs. prediction mode
            mask (tf.Tensor): Mask for inputs

        Returns: Tensor with result from applying the model to feature_vals

        """
        return tf.stack(
            [m(feature_vals) for m in self.models],
            axis=-1
        )

    def compile(
            self,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'rmsprop',
            loss: Optional[Union[str, tf.keras.losses.Loss]] = None,
            metrics: Optional[Union[str, tf.keras.metrics.Metric]] = None,
            loss_weights: Optional[Union[List[float], Dict[str, float]]] = None,
            weighted_metrics: Optional[Union[str, tf.keras.metrics.Metric]] = None,
            run_eagerly: bool = None,
            steps_per_execution: int = None,
            **kwargs: Any) -> None:
        """tf.keras.Model.compile implementation.

        Compiles each sub-model (i.e. per-class GLMMBinClassifierFeatureEncoder models)

        Args:
            optimizer: see tf.keras.Model.compile
            loss: see tf.keras.Model.compile
            metrics: see tf.keras.Model.compile
            loss_weights: see tf.keras.Model.compile
            weighted_metrics: see tf.keras.Model.compile
            run_eagerly: see tf.keras.Model.compile
            steps_per_execution: see tf.keras.Model.compile
            **kwargs: see tf.keras.Model.compile

        """
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
            metric = tf.concat([x[key] for x in metrics_dict_list], 1)
            merged_metrics[key] = merge_func(metric)
        return merged_metrics

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """A single train step for the model

        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): Tuple of train feature tensor and train label tensor for current batch

        Returns: Dict mapping metric names to metric value tensors for the current training step

        """
        x, y = data
        per_class_metrics_dicts = []
        for cls in range(self.num_classes):
            y_class = tf.where(y == cls, x=1, y=0)
            per_class_metrics_dicts.append(self.models[cls].train_step((x, y_class)))

        return self.__merge_metrics(per_class_metrics_dicts)

    def print_posterior_estimates(self) -> None:
        """Debug method to print parameter posterior estimates

        """
        for i, class_model in enumerate(self.models):
            print(f"Sub-model for class {i}")
            class_model.print_posterior_estimates()
