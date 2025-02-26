#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import random
import torch
import warnings
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier


class DecisionTreeTabPFNBase(BaseDecisionTree):
    """Class that implements a DT-TabPFN model based on sklearn package"""

    task_type = None

    def __init__(
        self,
        *,
        # Decision Tree Arguments
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=1000,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        # TabPFN Arguments
        tabpfn: TabPFNRegressor | TabPFNClassifier = None,
        categorical_features=None,
        verbose=False,
        show_progress=False,
        fit_nodes=True,
        tree_seed=None,
        adaptive_tree=True,
        adaptive_tree_min_train_samples=50,
        adaptive_tree_max_train_samples=2000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_test_size: float = 0.2,
        average_logits=True,
        adaptive_tree_skip_class_missing=True,
    ):
        """:param criterion: DecisionTree parameter, see Sklearn documentation
        :param splitter: DecisionTree parameter, see Sklearn documentation
        :param max_depth: DecisionTree parameter, see Sklearn documentation
        :param min_samples_split: DecisionTree parameter, see Sklearn documentation
        :param min_samples_leaf: DecisionTree parameter, see Sklearn documentation
        :param min_weight_fraction_leaf: DecisionTree parameter, see Sklearn documentation
        :param max_features: DecisionTree parameter, see Sklearn documentation
        :param random_state: DecisionTree parameter, see Sklearn documentation
        :param max_leaf_nodes: DecisionTree parameter, see Sklearn documentation
        :param min_impurity_decrease: DecisionTree parameter, see Sklearn documentation
        :param class_weight: DecisionTree parameter, see Sklearn documentation
        :param ccp_alpha: DecisionTree parameter, see Sklearn documentation
        :param monotonic_cst: DecisionTree parameter, see Sklearn documentation
        :param tabpfn: TabPFNBase Model
        :param categorical_features:
        :param verbose:
        :param show_progress:
        :param fit_nodes: Wheather to fit the nodes of the decision tree or only the leafs
        :param tree_seed:
        :param adaptive_tree: Wheather to use adaptive tree, if true, the tree will be pruned based on the performance
            in the out of bag split
        :param adaptive_tree_min_train_samples: TODO: Could we merge this with min_samples_leaf
        :param adaptive_tree_max_train_samples:
        :param adaptive_tree_min_valid_samples_fraction_of_train:
        :param adaptive_tree_overwrite_metric:
        :param adaptive_tree_test_size: Whats the hold out split fraction
        :param average_logits:
        :param adaptive_tree_skip_class_missing:
        """
        # Optional args are arguments that may not be present in BaseDecisionTree, depending on the sklearn version
        # This code helps keep compatability across versions
        optional_args = {"monotonic_cst": monotonic_cst}
        optional_args_filtered = {}
        for key, value in optional_args.items():
            if BaseDecisionTree.__init__.__code__.co_varnames.__contains__(key):
                optional_args_filtered[key] = value

        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            splitter=splitter,
            **optional_args_filtered,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.tabpfn = tabpfn
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree_seed = tree_seed
        self.categorical_features = categorical_features
        self.verbose = verbose
        self.show_progress = show_progress
        self.splitter = splitter
        self.fit_nodes = fit_nodes
        self.adaptive_tree = adaptive_tree
        self.adaptive_tree_min_train_samples = adaptive_tree_min_train_samples
        self.adaptive_tree_min_valid_samples_fraction_of_train = adaptive_tree_min_valid_samples_fraction_of_train
        self.adaptive_tree_max_train_samples = adaptive_tree_max_train_samples
        self.adaptive_tree_overwrite_metric = adaptive_tree_overwrite_metric
        self.adaptive_tree_test_size = adaptive_tree_test_size
        self.average_logits = average_logits
        self.adaptive_tree_skip_class_missing = adaptive_tree_skip_class_missing

    def _more_tags(self):
        return {"multilabel": True, "allow_nan": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags

    def set_categorical_features(self, categorical_features):
        """Sets categorical features
        :param categorical_features: Categorical features
        :return: None
        """
        self.categorical_features = categorical_features

    def fit(self, X, y, sample_weight=None, check_input=True):
        #TODO(Leo): I don't think this is used
        # Add return self to enable method chaining
        self._fit(X, y, sample_weight, check_input=check_input)
        return self

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        """Method that fits the DecisionTree-TabPFN model
        :param X: Feature training data
        :param y: Label training data
        :param sample_weight: NOT IMPLEMENTED
        :param check_input:
        :return: None
        """
        # Initialize attributes that should only be set in fit
        self.n_classes_ = 0
        self.classes_ = []
        self._decision_tree = None
        self._leaf_nodes = []
        self._leaf_train_data = {}
        self.label_encoder_ = LabelEncoder()

        if self.tree_seed is None:
            self.tree_seed_ = random.randint(1, 10000)
        else:
            self.tree_seed_ = self.tree_seed

        if check_input:
            pass
            # X = check_array(X)

        # Handle X and y
        X_preprocessed = self.preprocess_data_for_tree(X)

        sample_weight = (
            np.ones((X_preprocessed.shape[0],))
            if sample_weight is None
            else sample_weight
        )

        y = y.copy()
        self.classes_ = np.unique(y)
        y = pd.DataFrame(y).reset_index().drop("index", axis=1)
        self.n_classes_ = int(y.max().iloc[0] + 1)
        y_train = pd.DataFrame()
        y_train["class"] = y

        # Split data into train and validation set
        if (self.task_type == "multiclass") and (
            np.unique(y, return_counts=True)[1].min() == 1
        ):
            # If there is a class with only one sample, we cannot split the data properly.
            self.adaptive_tree = False

        if self.adaptive_tree:
            stratify = y if self.task_type == "multiclass" else None
            (
                train_X,
                valid_X,
                train_X_preprocessed,
                valid_X_preprocessed,
                train_y,
                valid_y,
                train_sample_weight,
                valid_sample_weight,
            ) = train_test_split(
                X,
                X_preprocessed,
                y,
                sample_weight,
                test_size=self.adaptive_tree_test_size,
                random_state=self.random_state,
                stratify=stratify,
            )

            if self.task_type == "multiclass" and len(np.unique(train_y)) != len(
                np.unique(valid_y),
            ):
                self.adaptive_tree = False
        else:
            train_X, train_X_preprocessed, train_y, train_sample_weight = (
                X,
                X_preprocessed,
                y,
                sample_weight,
            )

        # Create DecisionTree and get leaf nodes of each datapoint
        self._tree = self.init_decision_tree()

        self._tree.fit(
            train_X_preprocessed,
            train_y,
            sample_weight=train_sample_weight,
        )

        if self.verbose:
            self.post_fit()

        (
            self._X,
            self._y,
            self._train_X,
            self._train_X_preprocessed,
            self._train_y,
            self._train_sample_weight,
        ) = (
            X,
            y,
            train_X,
            train_X_preprocessed,
            train_y,
            train_sample_weight,
        )
        if self.adaptive_tree:
            (
                self._valid_X,
                self._valid_X_preprocessed,
                self._valid_y,
                self._valid_sample_weight,
            ) = (
                valid_X,
                valid_X_preprocessed,
                valid_y,
                valid_sample_weight,
            )

        self._todo_post_fit = True

    def fit_leafs(self, train_X, train_y):
        self._leaf_train_data = {}

        # Get leaf nodes of each datapoint in bootstrap sample
        self._leaf_nodes, bootstrap_X, bootstrap_y = self.apply_tree_train(
            train_X,
            train_y,
        )
        if self.verbose:
            print(
                f"Estimators: {self._leaf_nodes.shape[2]}, Nodes per estimator: {self._leaf_nodes.shape[1]}",
            )

        # Store train data point for each leaf
        for estimator_id in range(self._leaf_nodes.shape[2]):
            self._leaf_train_data[estimator_id] = {}
            for leaf_id in range(self._leaf_nodes.shape[1]):
                indices = np.argwhere(self._leaf_nodes[:, leaf_id, estimator_id]).ravel()
                X_train_samples = np.take(train_X, indices, axis=0)
                y_train_samples = np.array(np.take(train_y, indices, axis=0)).ravel()

                if self.verbose and not self.adaptive_tree:
                    print(
                        f"Leaf: {leaf_id} Shape: {X_train_samples.shape} / {train_X.shape}",
                    )

                self._leaf_train_data[estimator_id][leaf_id] = (
                    X_train_samples,
                    y_train_samples,
                )

    def get_tree(self):
        return self._tree

    def apply_tree(self, X):
        """Apply tree for different kinds of tree types.
        TODO: This function could also be overwritten in each type of tree

        (N_samples, N_nodes, N_estimators)
        :param bootstrap_X:
        :return:
        """
        decision_path = self.get_tree().decision_path(X)
        # If trees is a single tree, apply it and expand the missing dimension
        return np.expand_dims(decision_path.todense(), 2)

    def apply_tree_train(self, X, y):
        return self.apply_tree(X), X, y

    def init_decision_tree(self):
        raise NotImplementedError("init_decision_tree must be implemented")

    def post_fit(self):
        pass

    def preprocess_data_for_tree(self, X):
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        X_copy = np.array(X, copy=True)  # Create a writeable copy
        X_copy[np.isnan(X_copy)] = 0.0
        return X_copy

    def predict_(self, X, y=None, check_input=True):
        """Predicts X
        :param X: Data that should be evaluated
        :param y: True labels of holdout data used for adaptive tree.
            - If not None: Prunes nodes based on the performance of the holdout data y
            - If None: Predicts the data based on the previous hold out performances
        :param check_input:
        :return: Probabilities of each class
        """
        check_is_fitted(self)

        # TODO: This is a bit of a hack, but it is the only way to stop the fitting at a time budget
        #    We should probably refactor this to a more elegant solution where we move more logic to fit
        if self._todo_post_fit:
            self._todo_post_fit = False
            if self.adaptive_tree:
                if self.verbose:
                    print(
                        f"Starting tree leaf validation train {self._train_X.shape} valid {self._valid_X.shape}",
                    )
                # Pre-fits the leafs of the tree
                self.fit_leafs(self._train_X, self._train_y)
                self.predict_(
                    self._valid_X,
                    self._valid_y.values.ravel(),
                    check_input=False,
                )

            self.fit_leafs(self._X, self._y)

        #### END OF FITTING #####

        X_preprocessed = self.preprocess_data_for_tree(X)
        self.tabpfn.categorical_features_indices = self.categorical_features

        # Get leaf nodes of each datapoint in X
        # Matrix of shape (N_samples, N_nodes, N_estimators) with a one if the sample went trough that decision node
        X_test_leaf_nodes = self.apply_tree(X_preprocessed)
        N_samples, N_nodes, N_estimators = X_test_leaf_nodes.shape

        # We store the probabilities and metrics after evaluating each node or leaf
        y_prob, y_metric = {}, {}
        for estimator_id in range(N_estimators):
            y_prob[estimator_id], y_metric[estimator_id] = {}, {}

        prune_leafs_and_nodes = y is not None and self.adaptive_tree
        if prune_leafs_and_nodes:
            self.node_prediction_type = (
                {}
            )  # Node prediction type stores if a node was pruned or not
            for estimator_id in range(N_estimators):
                self.node_prediction_type[estimator_id] = {}

        if self.show_progress:
            import tqdm.auto

        # Make prediction via TabPFN on each leaf node
        for estimator_id in range(N_estimators):
            loop = (
                tqdm.auto.tqdm(range(N_nodes)) if self.show_progress else range(N_nodes)
            )
            for leaf_id in loop:
                # Initializes the predictions of the node, uses previous predictions
                self._pruning_init_node_predictions(
                    leaf_id,
                    estimator_id,
                    y_prob,
                    y_metric,
                    N_nodes,
                    N_samples,
                )

                # if first leaf of new tree, use last predictions of previous tree, since this is always the same
                if estimator_id > 0 and leaf_id == 0:
                    # print("Skipping leaf 0", estimator_id, leaf_id)
                    continue

                # Fetch the test sample indices that are in the node
                test_sample_indices = np.argwhere(
                    X_test_leaf_nodes[:, leaf_id, estimator_id],
                ).ravel()

                # Fetch the train samples that are in the node
                X_train_samples, y_train_samples = self._leaf_train_data[estimator_id][
                    leaf_id
                ]

                if X_train_samples.shape[0] == 0 or len(test_sample_indices) == 0:
                    if prune_leafs_and_nodes:
                        self.node_prediction_type[estimator_id][leaf_id] = "previous"
                    continue

                is_leaf = (
                    X_test_leaf_nodes[
                        test_sample_indices,
                        leaf_id + 1 :,
                        estimator_id,
                    ].sum()
                    == 0.0
                )  # if larger than 0, then it is not a leaf

                if (
                    not is_leaf
                    and not self.fit_nodes
                    and not (leaf_id == 0 and self.adaptive_tree)
                ):
                    if prune_leafs_and_nodes:
                        self.node_prediction_type[estimator_id][leaf_id] = "previous"
                    continue

                if self.adaptive_tree and leaf_id != 0:
                    # if we are running predictions and the node was skipped
                    if (
                        y is None
                        and self.node_prediction_type[estimator_id][leaf_id]
                        == "previous"
                    ):
                        continue

                    if (
                        self.task_type == "multiclass"
                        and len(np.unique(y_train_samples)) < self.n_classes_
                        and self.adaptive_tree_skip_class_missing
                    ):
                        self.node_prediction_type[estimator_id][leaf_id] = "previous"
                        continue

                    # if not enough train or test datapoints in a node, skip it
                    if (
                        (
                            X_train_samples.shape[0]
                            < self.adaptive_tree_min_train_samples
                        )
                        or (
                            len(test_sample_indices)
                            < self.adaptive_tree_min_valid_samples_fraction_of_train
                            * self.adaptive_tree_min_train_samples
                        )
                        or (
                            X_train_samples.shape[0]
                            > self.adaptive_tree_max_train_samples
                            and not is_leaf
                        )
                    ):
                        if prune_leafs_and_nodes:
                            if self.verbose:
                                print(
                                    f"Leaf: {leaf_id} Tree {estimator_id} No train ({X_train_samples.shape[0]}) / test data {len(test_sample_indices)}",
                                )
                            self.node_prediction_type[estimator_id][
                                leaf_id
                            ] = "previous"

                        continue

                if self.verbose:
                    if y is not None:
                        print(
                            f"Leaf: {leaf_id} Shape: {X_train_samples.shape} Shape test: {len(test_sample_indices)} / {X_preprocessed.shape}"
                            f" N classes (Train Test) {len(np.unique(y_train_samples))} {len(np.unique(y[test_sample_indices]))} /  {self.n_classes_}",
                        )
                    else:
                        print(
                            f"Leaf: {leaf_id} Shape: {X_train_samples.shape} Shape test: {len(test_sample_indices)} / {X_preprocessed.shape}"
                            f" N classes (Train Test) {len(np.unique(y_train_samples))} /  {self.n_classes_}",
                        )

                leaf_prediction = self.predict_leaf_(
                    X_train_samples,
                    y_train_samples,
                    leaf_id,
                    X,
                    test_sample_indices,
                )

                (
                    y_prob_averaging,
                    y_prob_replacement,
                ) = self._pruning_get_prediction_type_results(
                    y_prob,
                    leaf_prediction,
                    test_sample_indices,
                    estimator_id,
                    leaf_id,
                )

                if self.adaptive_tree:
                    if y is not None:
                        self._pruning_set_node_prediction_type(
                            y,
                            y_prob_averaging,
                            y_prob_replacement,
                            y_metric,
                            estimator_id,
                            leaf_id,
                        )
                    self._pruning_set_predictions(
                        y_prob,
                        y_prob_averaging,
                        y_prob_replacement,
                        estimator_id,
                        leaf_id,
                    )
                    if y is not None:
                        y_metric[estimator_id][leaf_id] = self._score(
                            y,
                            y_prob[estimator_id][leaf_id],
                        )
                        if self.verbose:
                            print(
                                f"Leaf: {leaf_id} Score: {y_metric[estimator_id][leaf_id]}",
                            )
                else:
                    y_prob[estimator_id][leaf_id] = y_prob_replacement

        return y_prob[N_estimators - 1][N_nodes - 1]

    def get_optimize_metric(self):
        if self.adaptive_tree_overwrite_metric is not None:
            return self.adaptive_tree_overwrite_metric
        return "roc" if self.task_type == "multiclass" else "rmse"

    def _score(self, y_true, y_pred):
        from ..scoring import scoring_utils

        if self.task_type == "multiclass":
            return scoring_utils.score_classification(
                self.get_optimize_metric(),
                y_true,
                y_pred,
            )
        if self.task_type == "regression":
            return scoring_utils.score_regression(
                self.get_optimize_metric(),
                y_true,
                y_pred,
            )
        return None

    def _pruning_init_node_predictions(
        self,
        leaf_id,
        estimator_id,
        y_prob,
        y_metric,
        N_nodes,
        N_samples,
    ):
        if estimator_id == 0 and leaf_id == 0:
            y_prob[0][0] = self.init_eval_prob(N_samples, to_zero=True)
            y_metric[0][0] = 0
        elif (
            leaf_id == 0 and estimator_id > 0
        ):  # if first leaf of new tree, use last predictions of previous tree
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id - 1][N_nodes - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id - 1][N_nodes - 1]
        else:  # else use last predictions of previous leaf
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]
            y_metric[estimator_id][leaf_id] = y_metric[estimator_id][leaf_id - 1]

    @staticmethod
    def softmax(logits):
        exp_logits = np.exp(logits)  # Apply exponential to each logit
        sum_exp_logits = np.sum(
            exp_logits,
            axis=-1,
            keepdims=True,
        )  # Sum of exponentials across classes
        return exp_logits / sum_exp_logits  # Normalize to get probabilities

    def _pruning_get_prediction_type_results(
        self,
        y_eval_prob,
        leaf_prediction,
        test_sample_indices,
        estimator_id,
        leaf_id,
    ):
        y_prob_replacement = np.copy(y_eval_prob[estimator_id][leaf_id])
        y_prob_replacement[test_sample_indices] = leaf_prediction[test_sample_indices]
        if self.task_type == "multiclass":
            y_prob_replacement /= y_prob_replacement.sum(axis=1)[:, None]

        y_prob_averaging = np.copy(y_eval_prob[estimator_id][leaf_id])
        if self.task_type == "multiclass":
            if self.average_logits:
                y_prob_averaging[test_sample_indices] = np.log(
                    y_prob_averaging[test_sample_indices] + 1e-6,
                )
                leaf_prediction[test_sample_indices] = np.log(
                    leaf_prediction[test_sample_indices] + 1e-6,
                )
                # y_prob_averaging[test_sample_indices] += leaf_prediction[test_sample_indices]

                # convert all_proba logits for the multiclass output to probabilities per class
                y_prob_averaging[test_sample_indices] = DecisionTreeTabPFNBase.softmax(
                    y_prob_averaging[test_sample_indices],
                )
            else:
                y_prob_averaging[test_sample_indices] += leaf_prediction[
                    test_sample_indices
                ]
                y_prob_averaging[test_sample_indices] /= y_prob_averaging[
                    test_sample_indices
                ].sum(axis=1)[:, None]
        else:
            y_prob_averaging[test_sample_indices] += leaf_prediction[
                test_sample_indices
            ]
            y_prob_averaging[test_sample_indices] /= 2

        return y_prob_averaging, y_prob_replacement

    def _pruning_set_predictions(
        self,
        y_prob,
        y_prob_averaging,
        y_prob_replacement,
        estimator_id,
        leaf_id,
    ):
        if self.node_prediction_type[estimator_id][leaf_id] == "averaging":
            y_prob[estimator_id][leaf_id] = y_prob_averaging
        elif self.node_prediction_type[estimator_id][leaf_id] == "replacement":
            y_prob[estimator_id][leaf_id] = y_prob_replacement
        else:
            y_prob[estimator_id][leaf_id] = y_prob[estimator_id][leaf_id - 1]

    def _pruning_set_node_prediction_type(
        self,
        y,
        y_prob_averaging,
        y_prob_replacement,
        y_metric,
        estimator_id,
        leaf_id,
    ):
        averaging_score = self._score(y, y_prob_averaging)
        replacement_score = self._score(y, y_prob_replacement)

        if (
            leaf_id == 0
            or max(averaging_score, replacement_score)
            > y_metric[estimator_id][leaf_id - 1]
        ):
            if replacement_score > averaging_score:
                prediction_type = "replacement"
            else:
                prediction_type = "averaging"
        else:
            prediction_type = "previous"

        self.node_prediction_type[estimator_id][leaf_id] = prediction_type

        if self.verbose:
            print(
                f"Leaf: {leaf_id} Averaging: {averaging_score} Replacement: {replacement_score}",
            )

    def predict_leaf_(self, X_train_samples, y_train_samples, leaf_id, X, indices):
        raise NotImplementedError("predict_leaf_ must be implemented")

    def init_eval_prob(self, n_samples, to_zero=False):
        if self.task_type == "multiclass":
            y_eval_prob = np.ones((n_samples, self.n_classes_)) / self.n_classes_
            if to_zero:
                y_eval_prob *= 0
        elif self.task_type == "regression":
            y_eval_prob = np.zeros((n_samples,))
        else:
            raise NotImplementedError("Task type not implemented")
        return y_eval_prob


class DecisionTreeTabPFNClassifier(ClassifierMixin, DecisionTreeTabPFNBase):
    """Class that implements a DT-TabPFN model based on sklearn package"""

    task_type = "multiclass"

    def predict_leaf_(self, X_train_samples, y_train_samples, leaf_id, X, indices):
        y_eval_prob = self.init_eval_prob(X.shape[0], to_zero=True)
        classes = np.array(np.unique(list(map(int, y_train_samples))))

        # Make actual prediction
        if len(classes) > 1:
            original_random_state = self.tabpfn.random_state
            try:
                self.tabpfn.random_state = leaf_id + self.tree_seed_
                self.tabpfn.fit(X_train_samples, y_train_samples)
                y_eval_prob[indices[:, None], classes] = self.tabpfn.predict_proba(
                    X[indices],
                )
            except ValueError as e:
                if (len(e.args) == 0) or e.args[
                    0
                ] != "All features are constant and would have been removed! Unable to predict using TabPFN.":
                    raise e

                warnings.warn(
                    "One leaf node has bad data to predict with TabPFN (e.g., only constant features). Returning ratio as predictions.",
                    stacklevel=2,
                )
                y_eval_prob[indices[:, None], np.array(classes)] = np.unique(
                    y_train_samples,
                    return_counts=True,
                )[1] / len(y_train_samples)
            finally:
                self.tabpfn.random_state = original_random_state
        else:
            y_eval_prob[indices[:, None], np.array(classes)] = 1

        return y_eval_prob

    def predict(self, X, check_input=True):
        """Predicts X_test
        :param X: Data that should be evaluated
        :param check_input:
        :return: Labels of the predictions
        """
        return np.array(
            list(map(np.argmax, self.predict_proba(X, check_input=check_input))),
        )

    def predict_proba(self, X, check_input=True):
        """Predicts X_test
        :param X: Data that should be evaluated
        :param check_input:
        :return: Probabilities of each class
        """
        return self.predict_(X, check_input=check_input)

    def post_fit(self):
        pass

    def init_decision_tree(self):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            splitter=self.splitter,
        )


class DecisionTreeTabPFNRegressor(RegressorMixin, DecisionTreeTabPFNBase):
    """Class that implements a DT-TabPFN model based on sklearn package"""

    task_type = "regression"

    def __init__(
        self,
        *,
        # Decision Tree Arguments
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=1000,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        # TabPFN Arguments
        tabpfn=None,
        categorical_features=None,
        verbose=False,
        show_progress=False,
        fit_nodes=True,
        tree_seed=0,
        adaptive_tree=True,
        adaptive_tree_min_train_samples=50,
        adaptive_tree_max_train_samples=2000,
        adaptive_tree_min_valid_samples_fraction_of_train=0.2,
        adaptive_tree_overwrite_metric=None,
        adaptive_tree_test_size=0.2,
        average_logits=True,
        adaptive_tree_skip_class_missing=True,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
            tabpfn=tabpfn,
            categorical_features=categorical_features,
            verbose=verbose,
            show_progress=show_progress,
            fit_nodes=fit_nodes,
            tree_seed=tree_seed,
            adaptive_tree=adaptive_tree,
            adaptive_tree_min_train_samples=adaptive_tree_min_train_samples,
            adaptive_tree_max_train_samples=adaptive_tree_max_train_samples,
            adaptive_tree_min_valid_samples_fraction_of_train=adaptive_tree_min_valid_samples_fraction_of_train,
            adaptive_tree_overwrite_metric=adaptive_tree_overwrite_metric,
            adaptive_tree_test_size=adaptive_tree_test_size,
            average_logits=average_logits,
            adaptive_tree_skip_class_missing=adaptive_tree_skip_class_missing,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags

    def init_decision_tree(self):
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            splitter=self.splitter,
        )

    def predict_leaf_(self, X_train_samples, y_train_samples, leaf_id, X, indices):
        original_random_state = self.tabpfn.random_state
        try:
            self.tabpfn.random_state = leaf_id + self.tree_seed_
            self.tabpfn.fit(X_train_samples, y_train_samples)
            y_eval_prob = np.zeros((X.shape[0],))
            y_eval_prob[indices] = self.tabpfn.predict(X[indices])
        finally:
            self.tabpfn.random_state = original_random_state

        return y_eval_prob

    def predict(self, X, check_input=True):
        """Predicts X_test
        :param X: Data that should be evaluated
        :param check_input:
        :return: Labels of the predictions
        """
        return self.predict_(X, check_input=check_input)

    def predict_full(self, X):
        """Predicts X
        :param X: Data that should be evaluated
        :param check_input:
        :return: Labels of the predictions
        """
        return self.predict_(X, check_input=False)

    def post_fit(self):
        pass
        # print(f"Decision Tree fitted with N nodes: {self._tree.tree_.node_count}")
