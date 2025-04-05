import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd

# np.random.seed(12)
# random.seed(12)
# torch.manual_seed(12)
# torch.random.manual_seed(12)


class MoleculeACEDataset(Dataset):
    """
    Custom PyTorch Dataset for datasets published by Van Tilborg et al.  (2022)

    Attributes:
        samples (torch.Tensor): A tensor of ECFPs with shape (num_samples, num_features).
        targets (torch.Tensor): A tensor of target labels.
        similar_molecules (list or None): A list of indices indicating similar molecules for each sample.

    Methods:
        __getitem__(index):
            Returns the sample, target value and optionally positive and negative examples for the given index.

        __len__():
            Returns the total number of samples in the dataset.
    """

    def __init__(self, samples, targets, similar_molecules=None):
        super().__init__()
        samples_stacked = np.vstack(samples.values)
        self.samples = torch.tensor(samples_stacked, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)
        self.similar_molecules = list(
            similar_molecules) if similar_molecules is not None else similar_molecules

    def __getitem__(self, index):
        if self.similar_molecules:
            anchor = self.samples[index]
            target_anchor = self.targets[index]

            similar_samples_to_anchor = self.samples[self.similar_molecules[index]]
            targets_similar_samples = self.targets[self.similar_molecules[index]]

            indices_pos_similar_samples = np.where(
                targets_similar_samples == target_anchor)[0]
            indices_neg_similar_samples = np.where(
                targets_similar_samples != target_anchor)[0]

            # Handle the cases when there are no similar positive and/or negative examples
            if len(indices_pos_similar_samples) == 0:
                # set all positive examples as similar examples, when no real similar example is available
                indices_pos_samples = np.where(
                    self.targets == target_anchor)[0]

                # delete index of anchor from similar examples
                temp_idx = np.where(indices_pos_samples == index)[0]
                indices_pos_samples = np.delete(indices_pos_samples, temp_idx)

                pos_idx = np.random.choice(indices_pos_samples)
                pos = self.samples[pos_idx]

            else:
                # Sample pos out of similar examples
                pos_idx = np.random.choice(indices_pos_similar_samples)
                pos = similar_samples_to_anchor[pos_idx]

            if len(indices_neg_similar_samples) == 0:
                # set all negative examples as similar examples, when no real similar example is available
                indices_neg_samples = np.where(
                    self.targets != target_anchor)[0]

                neg_idx = np.random.choice(indices_neg_samples)
                neg = self.samples[neg_idx]

            else:
                # Sample neg out of similar examples
                neg_idx = np.random.choice(indices_neg_similar_samples)
                neg = similar_samples_to_anchor[neg_idx]

            return anchor, target_anchor, pos, neg

        else:
            sample = self.samples[index]
            target = self.targets[index]

            return sample, target

    def __len__(self):
        return len(self.samples)


class MLP(nn.Module):
    """
    Multi-layer Perceptron architecture.

    Attributes:
        hidden_layers (nn.Sequential): A sequence of hidden layers.
        output_layer (nn.Linear): The final linear layer that outputs a single value (e.g., for regression).

    Methods:
        forward(x: torch.Tensor): Performs a forward pass through the network.
    """

    def __init__(self, n_input_features: int, n_hidden_layers: int, n_hidden_units: int, activation_function: callable,
                 input_dropout: int = 0.25, dropout: int = 0.5, alpha_dropout: bool = False):
        super().__init__()

        hidden_layers = []

        dropout_fcn = nn.AlphaDropout if alpha_dropout else nn.Dropout
        hidden_layers.append(dropout_fcn(input_dropout))

        for _ in range(n_hidden_layers):
            layer = nn.Linear(in_features=n_input_features,
                              out_features=n_hidden_units)
            torch.nn.init.normal_(layer.weight, mean=0,
                                  std=1 / np.sqrt(layer.in_features))
            hidden_layers.append(layer)
            hidden_layers.append(activation_function())
            hidden_layers.append(dropout_fcn(dropout))
            n_input_features = n_hidden_units

        hidden_layers.append(nn.LayerNorm(
            n_hidden_units, elementwise_affine=False))

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(n_hidden_units, 1)

    def forward(self, x: torch.Tensor):
        hidden_features = self.hidden_layers(x)
        output = self.output_layer(hidden_features)
        return output, hidden_features


def concatenate_dataloader(loader):
    """
    Concatenates all samples returned from a dataloader

    Parameters:
        loader (torch.utils.data.DataLoader): DataLoader to concatenate.

    Returns:
        tuple:
            A tuple containing:
                - np.ndarray: Array of all samples.
                - np.ndarray: Array of all labels.
    """

    all_X = []
    all_y = []

    for batch_X, batch_y in loader:
        all_X.append(batch_X.numpy())
        all_y.append(batch_y.numpy())

    X_np = np.concatenate(all_X, axis=0)
    y_np = np.concatenate(all_y, axis=0)

    return X_np, y_np


def compute_rf_metrics(rf, X_np, y_np):
    """
    Computes metrics for a given Random Forest.

    Parameters:
        rf (sklearn.ensemble.RandomForestClassifier): Random Forest to be evaluated.
        X_np (np.ndarray): Features of the samples.
        X_np (np.ndarray): Labels of the samples.

    Returns:
        various metrics (float)
    """
    preds = rf.predict(X_np)
    preds_proba = rf.predict_proba(X_np)[:, 1]

    loss = log_loss(y_np, preds_proba)
    accuracy = accuracy_score(y_np, preds)
    precision = precision_score(y_np, preds)
    recall = recall_score(y_np, preds)
    f1 = f1_score(y_np, preds)
    balanced_acc = balanced_accuracy_score(y_np, preds)
    roc_auc = roc_auc_score(y_np, preds_proba)

    return loss, accuracy, precision, recall, f1, roc_auc, balanced_acc


def train_rf(train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs):
    """
    Trains and evaluates a Random Forest with 100 trees.

    Parameters:
        train_loader (torch.utils.data.DataLoader): DataLoader for Training.
        val_loader (torch.utils.data.DataLoader): DataLoader for Validation.
        test_loader (torch.utils.data.DataLoader): DataLoader for Testing.
        train_loader_cliffs (torch.utils.data.DataLoader): DataLoader for Training, containing only cliff molecules.
        val_loader_cliffs (torch.utils.data.DataLoader): DataLoader for Validation, containing only cliff molecules.
        test_loader_cliffs (torch.utils.data.DataLoader): DataLoader for Testing, containing only cliff molecules.

    Returns:
        various metrics (float)
    """

    X_np, y_np = concatenate_dataloader(train_loader)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_np, y_np)

    val_X_np, val_y_np = concatenate_dataloader(val_loader)
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_balanced_acc = compute_rf_metrics(
        rf, val_X_np, val_y_np)

    val_X_np_cliffs, val_y_np_cliffs = concatenate_dataloader(
        val_loader_cliffs)
    val_loss_cliffs, val_accuracy_cliffs, val_precision_cliffs, val_recall_cliffs, val_f1_cliffs, val_roc_auc_cliffs, val_balanced_acc_cliffs = \
        compute_rf_metrics(rf, val_X_np_cliffs, val_y_np_cliffs)
    
    val_X_np_non_cliffs, val_y_np_non_cliffs = concatenate_dataloader(
        val_loader_non_cliffs)
    val_loss_non_cliffs, val_accuracy_non_cliffs, val_precision_non_cliffs, val_recall_non_cliffs, val_f1_non_cliffs, val_roc_auc_non_cliffs, val_balanced_acc_non_cliffs = \
        compute_rf_metrics(rf, val_X_np_non_cliffs, val_y_np_non_cliffs)

    test_X_np, test_y_np = concatenate_dataloader(test_loader)
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc = compute_rf_metrics(
        rf, test_X_np, test_y_np)

    test_X_np_cliffs, test_y_np_cliffs = concatenate_dataloader(
        test_loader_cliffs)
    test_loss_cliffs, test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs = compute_rf_metrics(
        rf, test_X_np_cliffs, test_y_np_cliffs)
    
    test_X_np_non_cliffs, test_y_np_non_cliffs = concatenate_dataloader(
        test_loader_non_cliffs)
    test_loss_non_cliffs, test_accuracy_non_cliffs, test_precision_non_cliffs, test_recall_non_cliffs, test_f1_non_cliffs, test_roc_auc_non_cliffs, test_balanced_acc_non_cliffs = compute_rf_metrics(
        rf, test_X_np_non_cliffs, test_y_np_non_cliffs)
    

    val_results = {
            "Loss": val_loss,
            "Accuracy": val_accuracy,
            "Precision": val_precision,
            "Recall": val_recall,
            "F1-Score": val_f1,
            "ROC-AUC": val_roc_auc,
            "Balanced Accuracy": val_balanced_acc
        }
    
    val_cliffs_results = {
            "Loss": val_loss_cliffs,
            "Accuracy": val_accuracy_cliffs,
            "Precision": val_precision_cliffs,
            "Recall": val_recall_cliffs,
            "F1-Score": val_f1_cliffs,
            "ROC-AUC": val_roc_auc_cliffs,
            "Balanced Accuracy": val_balanced_acc_cliffs
        }
    
    val_non_cliffs_results = {
            "Loss": val_loss_non_cliffs,
            "Accuracy": val_accuracy_non_cliffs,
            "Precision": val_precision_non_cliffs,
            "Recall": val_recall_non_cliffs,
            "F1-Score": val_f1_non_cliffs,
            "ROC-AUC": val_roc_auc_non_cliffs,
            "Balanced Accuracy": val_balanced_acc_non_cliffs
        }
    
    test_results = {
            "Loss": test_loss,
            "Accuracy": test_accuracy,
            "Precision": test_precision,
            "Recall": test_recall,
            "F1-Score": test_f1,
            "ROC-AUC": test_roc_auc,
            "Balanced Accuracy": test_balanced_acc
        }
    
    test_cliffs_results = {
            "Loss": test_loss_cliffs,
            "Accuracy": test_accuracy_cliffs,
            "Precision": test_precision_cliffs,
            "Recall": test_recall_cliffs,
            "F1-Score": test_f1_cliffs,
            "ROC-AUC": test_roc_auc_cliffs,
            "Balanced Accuracy": test_balanced_acc_cliffs
        }
    
    test_non_cliffs_results = {
            "Loss": test_loss_non_cliffs,
            "Accuracy": test_accuracy_non_cliffs,
            "Precision": test_precision_non_cliffs,
            "Recall": test_recall_non_cliffs,
            "F1-Score": test_f1_non_cliffs,
            "ROC-AUC": test_roc_auc_non_cliffs,
            "Balanced Accuracy": test_balanced_acc_non_cliffs
        }

    return rf, pd.DataFrame([val_results]), pd.DataFrame([val_cliffs_results]), pd.DataFrame([val_non_cliffs_results]), pd.DataFrame([test_results]), pd.DataFrame([test_cliffs_results]), pd.DataFrame([test_non_cliffs_results])

    # return val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_balanced_acc, \
    #     val_loss_cliffs, val_accuracy_cliffs, val_precision_cliffs, val_recall_cliffs, val_f1_cliffs, val_roc_auc_cliffs, val_balanced_acc_cliffs, \
    #     val_loss_non_cliffs, val_accuracy_non_cliffs, val_precision_non_cliffs, val_recall_non_cliffs, val_f1_non_cliffs, val_roc_auc_non_cliffs, val_balanced_acc_non_cliffs, \
    #     test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc, \
    #     test_loss_cliffs, test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs, \
    #     test_loss_non_cliffs, test_accuracy_non_cliffs, test_precision_non_cliffs, test_recall_non_cliffs, test_f1_non_cliffs, test_roc_auc_non_cliffs, test_balanced_acc_non_cliffs
