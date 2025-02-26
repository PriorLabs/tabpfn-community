"""TODO: Don't copy data but keep indices instead."""

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import numpy as np
import time
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
)

from .SklearnBasedDecisionTreeTabPFN import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RF-PFN")


def softmax_numpy(logits):
    exp_logits = np.exp(logits)  # Apply exponential to each logit
    sum_exp_logits = np.sum(
        exp_logits,
        axis=-1,
        keepdims=True,
    )  # Sum of exponentials across classes
    return exp_logits / sum_exp_logits  # Normalize to get probabilities


class RandomForestTabPFNBase:
    """Base Class for common functionalities."""

    def get_n_estimators(self, X):
        return self.n_estimators

    def fit(self, X, y, sample_weight=None):
        """Fits RandomForestTabPFN
        :param X: Feature training data
        :param y: Label training data
        :param sample_weight: Weights of each sample
        :return: None.
        """
        # Add explicit sparse matrix check
        X = check_array(X, accept_sparse=False, force_all_finite=False)

        if sample_weight is not None:
            logger.warning(
                "sample_weight is only used for the random forest part of the model, "
                "not for TabPFN. This might lead to unexpected behavior."
            )

        self.estimator = self.init_base_estimator()
        self.estimator.set_categorical_features(self.categorical_features)

        self.n_estimators = self.get_n_estimators(X)

        time.time()
        if self.max_depth == 0:
            self.tabpfn.fit(X, y)
        else:
            try:
                if torch.is_tensor(X):
                    X = X.numpy()
                if torch.is_tensor(y):
                    y = y.numpy()
                super().fit(X, y, sample_weight=sample_weight)
            except TypeError as e:
                print("Error in fit with data", X, y)
                raise e

        return self


class RandomForestTabPFNClassifier(RandomForestTabPFNBase, RandomForestClassifier):
    """RandomForestTabPFNClassifier."""

    task_type = "multiclass"

    def __init__(
        self,
        tabpfn=None,
        n_jobs=1,
        categorical_features=None,
        show_progress=False,
        verbose=0,
        adaptive_tree=True,
        fit_nodes=True,
        adaptive_tree_overwrite_metric="log_loss",
        adaptive_tree_test_size=0.2,
        adaptive_tree_min_train_samples=100,
        adaptive_tree_max_train_samples=5000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        preprocess_X_once=False,
        max_predict_time=60,
        rf_average_logits=True,
        dt_average_logits=True,
        adaptive_tree_skip_class_missing=True,
        # Added to make cloneable.
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        min_samples_split=1000,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """:param N_ensemble_configurations: Parameter of TabPFNClassifier
        :param n_estimators: Number of DT-TabPFN models
        :param min_samples_split: min_samples_split param of DTTabPFN
        :param max_features: sklearn DT param
        :param bootstrap: Whether use bootstrapping or not
        :param n_jobs: parallel processing of submodels n_jobs will be processed simultaniously
        :param random_state: random state of sklearn RF model
        """
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            verbose=verbose,
            n_jobs=n_jobs,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        if class_weight is not None:
            logger.warning(
                "class_weight is only used for the random forest part of the model, "
                "not for TabPFN. This might lead to unexpected behavior."
            )

        self.tabpfn = tabpfn

        self.categorical_features = categorical_features
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.adaptive_tree = adaptive_tree
        self.fit_nodes = fit_nodes
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = (
            adaptive_tree_min_valid_samples_fraction_of_train
        )
        self.preprocess_X_once = preprocess_X_once
        self.max_predict_time = max_predict_time
        self.rf_average_logits = rf_average_logits
        self.dt_average_logits = dt_average_logits
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing
        self.n_estimators = n_estimators

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        tags.estimator_type = "classifier"
        return tags

    def init_base_estimator(self):
        return DecisionTreeTabPFNClassifier(
            tabpfn=self.tabpfn,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            max_depth=self.max_depth,
            show_progress=self.show_progress,
            adaptive_tree=self.adaptive_tree,
            fit_nodes=self.fit_nodes,
            verbose=self.verbose,
            adaptive_tree_test_size=self.adaptive_tree_test_size,
            adaptive_tree_overwrite_metric=self.adaptive_tree_overwrite_metric,
            adaptive_tree_min_train_samples=self.adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=self.adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=self.adaptive_tree_min_valid_samples_fraction_of_train,
            average_logits=self.dt_average_logits,
            adaptive_tree_skip_class_missing=self.adaptive_tree_skip_class_missing,
        )

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns:
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        n_samples = proba[0].shape[0]
        # all dtypes should be the same, so just take the first
        class_type = self.classes_[0].dtype
        predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

        for k in range(self.n_outputs_):
            predictions[:, k] = self.classes_[k].take(
                np.argmax(proba[k], axis=1),
                axis=0,
            )

        return predictions

    def _final_proba(self, all_proba, evaluated_estimators):
        if evaluated_estimators == 0:
            raise ValueError("No estimators were evaluated. Check estimator initialization.")
        
        for proba in all_proba:
            proba /= evaluated_estimators

        if self.rf_average_logits:
            # convert all_proba logits for the multiclass output to probabilities per class
            all_proba = softmax_numpy(all_proba)

        if len(all_proba) == 1:
            return all_proba[0]
        return all_proba

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns:
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]

        start_time = time.time()
        evaluated_estimators = 0
        for e in self.estimators_:
            _accumulate_prediction(
                e.predict_proba,
                X,
                all_proba,
                accumulate_logits=self.rf_average_logits,
            )

            time_elapsed = time.time() - start_time
            evaluated_estimators += 1
            if time_elapsed > self.max_predict_time and self.max_predict_time > 0:
                break

        return self._final_proba(all_proba, evaluated_estimators)


class RandomForestTabPFNRegressor(RandomForestTabPFNBase, RandomForestRegressor):
    """RandomForestTabPFNRegressor."""

    task_type = "regression"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        tags.estimator_type = "regressor"
        return tags

    def __init__(
        self,
        tabpfn=None,
        n_jobs=1,
        categorical_features=None,
        show_progress=False,
        verbose=0,
        adaptive_tree=True,
        fit_nodes=True,
        adaptive_tree_overwrite_metric="rmse",
        adaptive_tree_test_size=0.2,
        adaptive_tree_min_train_samples=100,
        adaptive_tree_max_train_samples=5000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        preprocess_X_once=False,
        max_predict_time=-1,
        rf_average_logits=False,
        # Added to make cloneable.
        n_estimators=16,
        criterion="friedman_mse",
        max_depth=5,
        min_samples_split=300,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """:param N_ensemble_configurations: Parameter of TabPFNClassifier
        :param n_estimators: Number of DT-TabPFN models
        :param min_samples_split: min_samples_split param of DTTabPFN
        :param max_features: sklearn DT param
        :param bootstrap: Whether use bootstrapping or not
        :param n_jobs: parallel processing of submodels n_jobs will be processed simultaniously
        :param random_state: random state of sklearn RF model
        """
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        self.tabpfn = tabpfn

        self.categorical_features = categorical_features
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.adaptive_tree = adaptive_tree
        self.fit_nodes = fit_nodes
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = (
            adaptive_tree_min_valid_samples_fraction_of_train
        )
        self.preprocess_X_once = preprocess_X_once
        self.max_predict_time = max_predict_time
        self.rf_average_logits = rf_average_logits

    def init_base_estimator(self):
        return DecisionTreeTabPFNRegressor(
            tabpfn=self.tabpfn,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            categorical_features=self.categorical_features,
            max_depth=self.max_depth,
            show_progress=self.show_progress,
            adaptive_tree=self.adaptive_tree,
            fit_nodes=self.fit_nodes,
            verbose=self.verbose,
            adaptive_tree_test_size=self.adaptive_tree_test_size,
            adaptive_tree_overwrite_metric=self.adaptive_tree_overwrite_metric,
            adaptive_tree_min_train_samples=self.adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=self.adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=self.adaptive_tree_min_valid_samples_fraction_of_train,
        )

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns:
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        y_hat = (
            np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
            if self.n_outputs_ > 1
            else np.zeros(X.shape[0], dtype=np.float64)
        )

        start_time = time.time()
        evaluated_estimators = 0

        for e in self.estimators_:
            _accumulate_prediction(e.predict, X, [y_hat])
            time_elapsed = time.time() - start_time
            evaluated_estimators += 1
            if time_elapsed > self.max_predict_time and self.max_predict_time > 0:
                break

        y_hat /= evaluated_estimators

        return y_hat


def _accumulate_prediction(predict, X, out, accumulate_logits=False):
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)

    if accumulate_logits:
        # convert multiclass probabilities to logits
        prediction = np.log(prediction + 1e-10)  # Add small value to avoid log(0)

    if len(out) == 1:
        out[0] += prediction
    else:
        for i in range(len(out)):
            out[i] += prediction[i]
