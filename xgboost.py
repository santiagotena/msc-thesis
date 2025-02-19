# Colab installs
!pip install seaborn
!pip install torch-geometric
!pip install networkx
!pip install ucimlrepo
!pip install xgboost

# Core imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

# XGBoost
import xgboost as xgb

# UCI ML repository import
from ucimlrepo import fetch_ucirepo

# Itertools
import itertools

# Cudf
import cudf

class DataLoader():
    def __init__(self, parameters, dataset):
        self.parameters = parameters
        self.dataset = dataset
        self.loaded_dataset = fetch_ucirepo(id=dataset['id'])


class DataProcessor():
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']
        self.loaded_dataset = pipeline_registry[dataset_name]['data_loader'].loaded_dataset
        self.X = self.loaded_dataset.data.features
        self.X_numerical_features, self.X_categorical_features = self.split_feature_types()

        if self.X_numerical_features.empty:
            self.X_numeric_scaled = pd.DataFrame()
        else:
            self.X_numeric_scaled = self.scale_numeric()

        if self.X_categorical_features.empty:
            self.X_categorical_encoded = pd.DataFrame()
        else:
            self.X_categorical_encoded = pd.get_dummies(self.X_categorical_features)

        self.X_prepared = pd.concat([self.X_numeric_scaled, self.X_categorical_encoded], axis=1)
        self.X_prepared = self.X_prepared.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        self.x_tensor = torch.tensor(self.X_prepared.values.astype(np.float32), dtype=torch.float).to(self.device)

        self.y = self.loaded_dataset.data.targets
        self.y_encoded = self.encode_target()
        self.num_classes = len(self.y_encoded['target'].unique())
        self.y_tensor = torch.tensor(self.y_encoded.values.ravel(), dtype=torch.long).to(self.device)

    def split_feature_types(self):
        numerical_features = self.X.select_dtypes(include=[np.number])
        categorical_features = self.X.select_dtypes(exclude=[np.number])
        return numerical_features, categorical_features

    def scale_numeric(self):
        scaler = StandardScaler()
        X_numeric_scaled = pd.DataFrame(scaler.fit_transform(self.X_numerical_features),
                                        columns=self.X_numerical_features.columns)
        return X_numeric_scaled

    def encode_target(self):
        encoder = LabelEncoder()
        y_encoded = pd.DataFrame(encoder.fit_transform(self.y.values.ravel()), columns=['target'])
        return y_encoded


class DataSplitter():
    def __init__(self, parameters):
        self.random_seed = parameters['random_seed']
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_seed)

    def split(self, X, y):
        return self.kfold.split(X, y)

    def train_test_split(self, X, y, test_size=0.1):
        return train_test_split(X, y, test_size=test_size, random_state=self.random_seed)

class XGBoostModel():
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']
        self.X = pipeline_registry[dataset_name]['data_processor'].X_prepared
        self.y = pipeline_registry[dataset_name]['data_processor'].y_encoded['target']
        self.num_classes = len(np.unique(self.y.values))
        self.data_splitter = pipeline_registry[dataset_name]['data_splitter']
        self.xgb_params = parameters['xgboost_model']

        self.results = {
            'f1_scores': [],
            'accuracy_scores': [],
            'best_hyperparameters': []
        }

        self.run_model()

    def run_model(self):
        final_f1_scores = []
        final_accuracy_scores = []
        final_hyperparameters = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(self.data_splitter.split(self.X, self.y)):
            print(f"\nRunning fold {fold_idx + 1}/10...")
            X_train_val, X_test = self.X.iloc[train_val_idx], self.X.iloc[test_idx]
            y_train_val, y_test = self.y.iloc[train_val_idx], self.y.iloc[test_idx]
            X_train, X_val, y_train, y_val = self.data_splitter.train_test_split(
                X_train_val, y_train_val, test_size=0.1
            )

            X_train = self.move_to_device(X_train)
            X_val = self.move_to_device(X_val)
            X_test = self.move_to_device(X_test)
            y_train = self.move_to_device(y_train)
            y_val = self.move_to_device(y_val)
            y_test = self.move_to_device(y_test)

            unique_classes = set(np.unique(y_train))
            full_classes = set(range(self.num_classes))
            missing_classes = full_classes - unique_classes

            print(f"Fold {fold_idx + 1}: Classes in training = {unique_classes}, Missing classes = {missing_classes}")

            if missing_classes:
                dummy_samples = pd.DataFrame(
                    np.zeros((len(missing_classes), X_train.shape[1])),
                    columns=X_train.columns
                )
                dummy_labels = pd.Series(list(missing_classes), name=y_train.name)

                X_train = pd.concat([X_train, dummy_samples], ignore_index=True)
                y_train = pd.concat([y_train, dummy_labels], ignore_index=True)

                sample_weights = np.ones(len(y_train))
                sample_weights[-len(missing_classes):] = 0
            else:
                sample_weights = np.ones(len(y_train))

            best_model = None
            best_score = -float('inf')
            best_params = None
            param_grid = self.get_param_grid()

            for params in param_grid:
                model = xgb.XGBClassifier(
                    objective='multi:softmax',
                    eval_metric='mlogloss',
                    num_class=self.num_classes,
                    tree_method='gpu_hist' if self.device == 'cuda' else 'hist',
                    random_state=self.parameters['random_seed'],
                    **params
                )

                model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)

                val_preds = model.predict(X_val)
                val_score = f1_score(y_val, val_preds, average='weighted')

                if val_score > best_score:
                    best_score = val_score
                    best_model = model
                    best_params = params

            print(f"Best validation F1 score for fold {fold_idx + 1}: {best_score:.4f}")
            print(f"Best hyperparameters for fold {fold_idx + 1}: {best_params}")

            fold_f1_scores = []
            fold_accuracy_scores = []

            for retrain_run in range(3):
                print(f"Retraining best model (run {retrain_run + 1}/3)...")

                unique_classes = np.unique(y_train_val)
                full_classes = np.arange(self.num_classes)
                missing_classes = set(full_classes) - set(unique_classes)

                if missing_classes:
                    dummy_samples = pd.DataFrame(
                        np.zeros((len(missing_classes), X_train_val.shape[1])),
                        columns=X_train_val.columns
                    )
                    dummy_labels = pd.Series(list(missing_classes), name=y_train_val.name)

                    X_train_val = pd.concat([X_train_val, dummy_samples], ignore_index=True)
                    y_train_val = pd.concat([y_train_val, dummy_labels], ignore_index=True)

                    sample_weights = np.ones(len(y_train_val))
                    sample_weights[-len(missing_classes):] = 0
                else:
                    sample_weights = np.ones(len(y_train_val))

                X_train_val = self.move_to_device(X_train_val)
                y_train_val = self.move_to_device(y_train_val)

                best_model.fit(X_train_val, y_train_val, sample_weight=sample_weights)

                test_preds = best_model.predict(X_test)
                test_f1 = f1_score(y_test, test_preds, average='weighted')
                test_accuracy = accuracy_score(y_test, test_preds)

                fold_f1_scores.append(test_f1)
                fold_accuracy_scores.append(test_accuracy)

            avg_f1 = np.mean(fold_f1_scores)
            avg_accuracy = np.mean(fold_accuracy_scores)
            print(f"Fold {fold_idx + 1} - Avg F1: {avg_f1:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

            final_f1_scores.append(avg_f1)
            final_accuracy_scores.append(avg_accuracy)
            final_hyperparameters.append(best_params)

        self.results['f1_scores'] = final_f1_scores
        self.results['accuracy_scores'] = final_accuracy_scores
        self.results['best_hyperparameters'] = final_hyperparameters

        print("\nFinal Results:")
        print(f"F1 Score - Mean: {np.mean(final_f1_scores):.4f}, Std: {np.std(final_f1_scores):.4f}")
        print(f"Accuracy - Mean: {np.mean(final_accuracy_scores):.4f}, Std: {np.std(final_accuracy_scores):.4f}")

        print("\nBest Hyperparameters for Each Fold:")
        for i, params in enumerate(final_hyperparameters, start=1):
            print(f"Fold {i}: {params}")

        most_common_params = max(set(tuple(d.items()) for d in final_hyperparameters),
                                key=lambda x: final_hyperparameters.count(dict(x)))
        print(f"\nMost Frequently Selected Hyperparameters: {dict(most_common_params)}")

    def move_to_device(self, data):
        if self.device == 'cuda':
            if isinstance(data, pd.DataFrame):
                return cudf.DataFrame.from_pandas(data)
            elif isinstance(data, pd.Series):
                return cudf.Series(data)
        return data

    def get_param_grid(self):
        param_keys = self.xgb_params.keys()
        param_values = self.xgb_params.values()

        param_combinations = itertools.product(*param_values)
        param_grid = [dict(zip(param_keys, combination)) for combination in param_combinations]

        return param_grid

# Main
def build_parameters():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_seed = 42

    datasets = [
        {'name': 'abalone',
         'id': 1, },
        {'name': 'dry_bean',
         'id': 602, },
        {'name': 'isolet',
         'id': 54, },
        {'name': 'musk_v2',
         'id': 75, },
    ]

    xgboost_model = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
    }

    return {
        'device': device,
        'random_seed': random_seed,
        'datasets': datasets,
        'xgboost_model': xgboost_model,
    }


def build_pipeline_registry(dataset_names):
    pipeline_registry = {}
    for _, dataset_name in enumerate(dataset_names):
        pipeline_registry.setdefault(dataset_name, {})
    return pipeline_registry


def main():
    parameters = build_parameters()
    torch.manual_seed(parameters['random_seed'])
    np.random.seed(parameters['random_seed'])
    dataset_names = [dataset['name'] for dataset in parameters['datasets']]
    pipeline_registry = build_pipeline_registry(dataset_names)

    for dataset in parameters['datasets']:
        torch.cuda.empty_cache()
        dataset_name = dataset['name']
        print("--------------------------------")
        print(f"Loading dataset: {dataset_name}")
        print("--------------------------------")
        pipeline_registry[dataset_name]['data_loader'] = DataLoader(parameters=parameters, dataset=dataset)
        pipeline_registry[dataset_name]['data_processor'] = DataProcessor(parameters=parameters,
                                                                          pipeline_registry=pipeline_registry,
                                                                          dataset_name=dataset_name)
        pipeline_registry[dataset_name]['data_splitter'] = DataSplitter(parameters=parameters)
        pipeline_registry[dataset_name]['xgboost_model'] = XGBoostModel(parameters=parameters,
                                                                        pipeline_registry=pipeline_registry,
                                                                        dataset_name=dataset_name)

main()