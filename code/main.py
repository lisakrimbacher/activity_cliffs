import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv
import wandb

from model import MoleculeACEDataset, MLP, train_rf
import preprocessing


def build_dataset(batch_size, use_contrastive_learning=False):
    """
    Builds DataLoaders for the training, validation and test sets for all molecules and just for activity cliff molecules.

    Parameters:
        batch_size (int): Batch size for all DataLoaders.
        use_contrastive_learning (bool): Whether contrastive learning is used (DataLoaders include positive and negative 
            examples in that case).

    Returns:
        tuple:
            A tuple containing:
                - torch.utils.data.DataLoader: DataLoader for training set (all molecules).
                - torch.utils.data.DataLoader: DataLoader for validation set (all molecules).
                - torch.utils.data.DataLoader: DataLoader for test set (all molecules).
                - torch.utils.data.DataLoader: DataLoader for training set (cliff molecules).
                - torch.utils.data.DataLoader: DataLoader for validation set (cliff molecules).
                - torch.utils.data.DataLoader: DataLoader for test set (cliff molecules).
    """

    cliff_df_train = df_train[df_train['cliff_mol_binary'] == 1]
    cliff_df_val = df_val[df_val['cliff_mol_binary'] == 1]
    cliff_df_test = df_test[df_test['cliff_mol_binary'] == 1]

    non_cliff_df_train = df_train[df_train['cliff_mol_binary'] == 0]
    non_cliff_df_val = df_val[df_val['cliff_mol_binary'] == 0]
    non_cliff_df_test = df_test[df_test['cliff_mol_binary'] == 0]

    # no similar molecules extracted, since these datasets with just cliffs are only used for validation,
    # not training with triplet loss!
    train_set_cliffs = MoleculeACEDataset(
        cliff_df_train['ecfp'], cliff_df_train['active'])
    val_set_cliffs = MoleculeACEDataset(
        cliff_df_val['ecfp'], cliff_df_val['active'])
    test_set_cliffs = MoleculeACEDataset(
        cliff_df_test['ecfp'], cliff_df_test['active'])

    train_loader_cliffs = DataLoader(
        train_set_cliffs, shuffle=True, batch_size=batch_size)
    val_loader_cliffs = DataLoader(
        val_set_cliffs, shuffle=False, batch_size=batch_size)
    test_loader_cliffs = DataLoader(
        test_set_cliffs, shuffle=False, batch_size=batch_size)

    train_set_non_cliffs = MoleculeACEDataset(
        non_cliff_df_train['ecfp'], non_cliff_df_train['active'])
    val_set_non_cliffs = MoleculeACEDataset(
        non_cliff_df_val['ecfp'], non_cliff_df_val['active'])
    test_set_non_cliffs = MoleculeACEDataset(
        non_cliff_df_test['ecfp'], non_cliff_df_test['active'])

    train_loader_non_cliffs = DataLoader(
        train_set_non_cliffs, shuffle=True, batch_size=batch_size)
    val_loader_non_cliffs = DataLoader(
        val_set_non_cliffs, shuffle=False, batch_size=batch_size)
    test_loader_non_cliffs = DataLoader(
        test_set_non_cliffs, shuffle=False, batch_size=batch_size)

    if use_contrastive_learning:
        train_set = MoleculeACEDataset(
            df_train['ecfp'], df_train['active'], df_train['similar_molecules'])
    else:
        train_set = MoleculeACEDataset(
            df_train['ecfp'], df_train['active'])
    val_set = MoleculeACEDataset(
        df_val['ecfp'], df_val['active'])
    test_set = MoleculeACEDataset(
        df_test['ecfp'], df_test['active'])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs


def build_network(n_hidden_layers=2, n_hidden_units=1024, activation_function=nn.ReLU, input_dropout=0.25,
                  dropout=0.5, alpha_dropout=False, seed=12):
    """
    Builds a Multi-layer Perceptron with given hyperparameters.

    Parameters:
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_units (int): Number of hidden units in each hidden layer.
        activation_function (Callable): Activation function to be used for each layer.
        input_dropout (float): Dropout rate for the inputs (applied before first layer).
        dropout (float): Dropout rate for all hidden units (applied after each layer).

    Returns:
        MLP: Built Multi-layer Perceptron.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)

    network = MLP(n_input_features=2048, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units,
                  activation_function=activation_function, input_dropout=input_dropout,
                  dropout=dropout, alpha_dropout=alpha_dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    return network


def build_optimizer(network, optimizer, learning_rate):
    """
    Builds an optimizer for a given MLP.

    Parameters:
        network (MLP): Instance of Multi-layer Perceptron class.
        optimizer (string): Optimizer to be used (either 'adam' or 'sgd').
        learning_rate (float): Learning rate to be used for the updates.

    Returns:
        torch.optim.Adam or torch.optim.SGD: Built optimizer.
    """
    if optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=learning_rate)

    return optimizer


def train(config, use_contrastive_learning=False, use_cosine_sim=False, seed=12):
    """
    Trains a Multi-layer Perceptron with given hyperparameters.

    Parameters:
        config (dict): Dictionary of hyperparameters.
        use_contrastive_learning (bool): Whether contrastive learning is used.
        use_cosine_sim (bool): Whether Cosine similarity should be used as a distance metric in the Triplet Loss. 
            Has no effect if use_contrastive_learning=False.

    Returns:
        MLP: Trained Multi-layer Perceptron.
    """
    alpha_dropout = True if config['activation_function'] == "selu" else False

    if config['activation_function'] == "selu":
        act_fcn = nn.SELU
    elif config['activation_function'] == "relu":
        act_fcn = nn.ReLU
    elif config['activation_function'] == "leaky_relu":
        act_fcn = nn.LeakyReLU
    else:
        raise Exception("Invalid activation function.")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)

    loaders = build_dataset(
        config['batch_size'], use_contrastive_learning=use_contrastive_learning)

    network = build_network(n_hidden_layers=config['n_hidden_layers'], n_hidden_units=config['n_hidden_units'],
                            activation_function=act_fcn,
                            input_dropout=config['input_dropout'], dropout=config['dropout'], alpha_dropout=alpha_dropout, seed=seed)
    optimizer = build_optimizer(
        network, config['optimizer'], config['learning_rate'])

    patience = config.patience if 'patience' in config else 20
    best_val_loss = float('inf')
    epoch_best_val_loss = -1
    early_stop_counter = 0

    for epoch in tqdm(range(config['epochs'])):
        if use_contrastive_learning:
            train_results, val_results, val_cliffs_results, val_non_cliffs_results = \
                train_epoch(epoch, network, loaders, optimizer,
                            alpha=config['alpha'], use_contrastive_learning=use_contrastive_learning, use_cosine_sim=use_cosine_sim, seed=seed)
        else:
            train_results, val_results, val_cliffs_results, val_non_cliffs_results = \
                train_epoch(epoch, network, loaders, optimizer,
                            use_contrastive_learning=use_contrastive_learning, seed=seed)

        val_loss = val_results["Loss"][0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_best_val_loss = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(
        f"\nBest val-loss ({best_val_loss}) in epoch {epoch_best_val_loss}\n")

    return network


def train_wandb(config=None):
    # TODO: add docstring

    # TODO: check use_contrastive_learning, use_cosine_sim, current_seed

    #config = wandb.config

    alpha_dropout = True if config.activation_function == "selu" else False

    if config.activation_function == "selu":
        act_fcn = nn.SELU
    elif config.activation_function == "relu":
        act_fcn = nn.ReLU
    elif config.activation_function == "leaky_relu":
        act_fcn = nn.LeakyReLU
    else:
        raise Exception("Invalid activation function.")

    np.random.seed(wandb.config.current_seed)
    random.seed(config.current_seed)
    torch.manual_seed(config.current_seed)
    torch.random.manual_seed(config.current_seed)

    loaders = build_dataset(
        config.batch_size, use_contrastive_learning=use_contrastive_learning)

    network = build_network(n_hidden_layers=config.n_hidden_layers, n_hidden_units=config.n_hidden_units,
                            activation_function=act_fcn,
                            input_dropout=config.input_dropout, dropout=config.dropout, alpha_dropout=alpha_dropout, seed=config.current_seed)
    optimizer = build_optimizer(
        network, config.optimizer, config.learning_rate)

    patience = config.patience if 'patience' in config else 20
    best_val_loss = float('inf')
    epoch_best_val_loss = -1
    early_stop_counter = 0

    for epoch in tqdm(range(config['epochs'])):
        if use_contrastive_learning:
            train_results, val_results, val_cliffs_results, val_non_cliffs_results = \
                train_epoch(epoch, network, loaders, optimizer,
                            alpha=config['alpha'], use_contrastive_learning=use_contrastive_learning, use_cosine_sim=use_cosine_sim, seed=config.current_seed)
        else:
            train_results, val_results, val_cliffs_results, val_non_cliffs_results = \
                train_epoch(epoch, network, loaders, optimizer,
                            use_contrastive_learning=use_contrastive_learning, seed=config.current_seed)

        wandb.log({
            "train_loss_" + str(config.current_seed): train_results["Loss"].item(),
            "train_accuracy_" + str(config.current_seed): train_results["Accuracy"].item(),
            "train_precision_" + str(config.current_seed): train_results["Precision"].item(),
            "train_recall_" + str(config.current_seed): train_results["Recall"].item(),
            "train_f1-score_" + str(config.current_seed): train_results["F1-Score"].item(),
            "train_roc-auc_" + str(config.current_seed): train_results["ROC-AUC"].item(),
            "train_balanced-accuracy_" + str(config.current_seed): train_results["Balanced Accuracy"].item(),

            "val_loss_" + str(config.current_seed): val_results["Loss"].item(),
            "val_accuracy_" + str(config.current_seed): val_results["Accuracy"].item(),
            "val_precision_" + str(config.current_seed): val_results["Precision"].item(),
            "val_recall_" + str(config.current_seed): val_results["Recall"].item(),
            "val_f1-score_" + str(config.current_seed): val_results["F1-Score"].item(),
            "val_roc-auc_" + str(config.current_seed): val_results["ROC-AUC"].item(),
            "val_balanced-accuracy_" + str(config.current_seed): val_results["Balanced Accuracy"].item(),

            "val_cliffs_loss_" + str(config.current_seed): val_cliffs_results["Loss"].item(),
            "val_cliffs_accuracy_" + str(config.current_seed): val_cliffs_results["Accuracy"].item(),
            "val_cliffs_precision_" + str(config.current_seed): val_cliffs_results["Precision"].item(),
            "val_cliffs_recall_" + str(config.current_seed): val_cliffs_results["Recall"].item(),
            "val_cliffs_f1-score_" + str(config.current_seed): val_cliffs_results["F1-Score"].item(),
            "val_cliffs_roc-auc_" + str(config.current_seed): val_cliffs_results["ROC-AUC"].item(),
            "val_cliffs_balanced-accuracy_" + str(config.current_seed): val_cliffs_results["Balanced Accuracy"].item(),

            "val_non_cliffs_loss_" + str(config.current_seed): val_non_cliffs_results["Loss"].item(),
            "val_non_cliffs_accuracy_" + str(config.current_seed): val_non_cliffs_results["Accuracy"].item(),
            "val_non_cliffs_precision_" + str(config.current_seed): val_non_cliffs_results["Precision"].item(),
            "val_non_cliffs_recall_" + str(config.current_seed): val_non_cliffs_results["Recall"].item(),
            "val_non_cliffs_f1-score_" + str(config.current_seed): val_non_cliffs_results["F1-Score"].item(),
            "val_non_cliffs_roc-auc_" + str(config.current_seed): val_non_cliffs_results["ROC-AUC"].item(),
            "val_non_cliffs_balanced-accuracy_" + str(config.current_seed): val_non_cliffs_results["Balanced Accuracy"].item()
        })

        val_loss = val_results["Loss"][0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_best_val_loss = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(
        f"\nBest val-loss ({best_val_loss}) in epoch {epoch_best_val_loss}\n")

    # network_path = f"wandb/{dataset_folder}_{run.id}_{config.current_seed}.pt"
    # torch.save(network, network_path)

    return network, *loaders #train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs


def train_epoch(epoch_id, network, loaders, optimizer, alpha=0.1, margin=1, loss_function=nn.BCEWithLogitsLoss(),
                use_contrastive_learning=False, use_cosine_sim=False, seed=12):
    """
    Trains a Multi-layer Perceptron for one epoch.

    Parameters:
        epoch_id (int): Dictionary of hyperparameters.
        network (MLP): Instance of Multi-layer Perceptron class.
        loaders (tuple): Tuple of all DataLoaders.
        optimizer (torch.optim.Optimizer): Optimizer to be used.
        alpha (float): Weighting factor for Triplet loss. Has no effect if use_contrastive_learning=False.
        margin (int): Margin of for the Triplet loss. Has no effect if use_contrastive_learning=False.
        loss_function (Callable): Loss function to be used for all Multi-layer Perceptron, no matter if 
            use_contrastive_learning=True or use_contrastive_learning=False.
        use_contrastive_learning (bool): Whether contrastive learning is used.
        use_cosine_sim (bool): Whether Cosine similarity should be used as a distance metric in the Triplet Loss. 
            Has no effect if use_contrastive_learning=False.

    Returns:
        various metrics (float)
    """
    # TODO: check reproducibility
    # np.random.seed(seed + epoch_id)
    # random.seed(seed + epoch_id)
    # torch.manual_seed(seed + epoch_id)
    # torch.random.manual_seed(seed + epoch_id)

    train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs = loaders

    network.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplet_loss_function = nn.TripletMarginLoss(margin=margin, p=1)

    for batch in train_loader:
        if use_contrastive_learning:
            anchors, targets_anchors, pos, neg = batch
            anchors = anchors.to(device)
            targets_anchors = targets_anchors.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchors_output, anchors_emb = network(anchors)
            anchors_output = anchors_output.squeeze(dim=1)

            pos_output, pos_emb = network(pos)
            pos_output = pos_output.squeeze(dim=1)

            neg_output, neg_emb = network(neg)
            neg_output = neg_output.squeeze(dim=1)

            basic_loss = loss_function(anchors_output, targets_anchors)

            if use_cosine_sim:
                # cosine sim now between 0 and 1
                dist_pos = (nn.CosineSimilarity()(
                    anchors_emb, pos_emb) + 1) / 2
                dist_neg = (nn.CosineSimilarity()(anchors_emb, neg_emb)) / 2
                triplet_loss_by_samples = torch.abs(
                    dist_pos - dist_neg + margin)

                triplet_loss = torch.sum(
                    triplet_loss_by_samples) / len(triplet_loss_by_samples)

            else:
                triplet_loss = triplet_loss_function(
                    anchors_emb, pos_emb, neg_emb)

            loss = basic_loss + alpha * triplet_loss

            if tune_wandb:
                wandb.log({
                    "bce_loss_batch": basic_loss,
                    "triplet_loss_batch": triplet_loss
                })
            else:
                train_triplet_losses.append(
                    triplet_loss.cpu().detach().numpy())
                train_basic_losses.append(basic_loss.cpu().detach().numpy())

        else:
            samples, targets = batch
            samples = samples.to(device)
            targets = targets.to(device)

            output, emb = network(samples)
            output = output.squeeze(dim=1)
            loss = loss_function(output, targets)

        if tune_wandb:
            wandb.log({
                "total_loss_batch": loss,
            })
        else:
            train_losses_total.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_results = compute_metrics(
        train_loader, network, train_loader=True)
    val_results = compute_metrics(
        val_loader, network)
    val_cliffs_results = \
        compute_metrics(val_loader_cliffs, network)
    val_non_cliffs_results = \
        compute_metrics(val_loader_non_cliffs, network)

    if not tune_wandb:
        val_losses.append(val_results["Loss"][0])

    return train_results, val_results, val_cliffs_results, val_non_cliffs_results


# def create_sweep_config():
#     # TODO: create docstring


#     # https://docs.wandb.ai/guides/sweeps/sweep-config-keys

#     sweep_config = {
#         'method': 'bayes'
#         }

#     metric = {
#         'name': 'val_loss',
#         'goal': 'minimize'
#         }

#     sweep_config['metric'] = metric

#     parameters_dict = {
#         'optimizer': {
#             'values': ['adam']
#             },
#         'epochs': {
#             'values': [20, 50, 70, 100]
#             },
#         'learning_rate': {
#             'min': 1e-3,
#             'max': 1e0
#             },
#         'batch_size': {
#             'values': [56, 128]
#             },
#         'n_hidden_layers': {
#             'values': [2, 3, 4, 6]
#             },
#         'n_hidden_units': {
#             'values': [1024, 768, 512, 256, 128]
#             },
#         'activation_function': {
#             'values': ["relu", "selu", "leaky_relu"]
#             },
#         'input_dropout': {
#             'values': [0., 0.1, 0.2, 0.5, 0.6, 0.7]
#             },
#         'dropout': {
#             'values': [0., 0.1, 0.2, 0.3, 0.5]
#             },
#         'alpha': {
#             'min': 0.1,
#             'max': 2.0
#             },
#         'margin': {
#             'values': [1.]
#             },
#         }

#     sweep_config['parameters'] = parameters_dict

#     return sweep_config

def run_sweep(sweep_id, fcn, count):
    # TODO: create docstring
    wandb.agent(sweep_id, function=fcn, count=count)


def log_avg_results(set_name, results, results_cliffs, results_non_cliffs):

    for results_array, label in zip([results, results_non_cliffs, results_cliffs], ["total", "non_cliffs", "cliffs"]):
        mean = np.mean(results_array, axis=0)
        std = np.std(results_array, axis=0)

        wandb.log({
            set_name + 'loss_' + label + '_mean': mean['Loss'].item(),
            set_name + 'loss_' + label + '_std': std['Loss'].item(),

            set_name + 'roc-auc_' + label + '_mean': mean['ROC-AUC'].item(),
            set_name + 'roc-auc_' + label + '_std': std['ROC-AUC'].item(),

            set_name + 'accuracy_' + label + '_mean': mean['Accuracy'].item(),
            set_name + 'accuracy_' + label + '_std': std['Accuracy'].item(),

            set_name + 'precision_' + label + '_mean': mean['Precision'].item(),
            set_name + 'precision_' + label + '_std': std['Precision'].item(),

            set_name + 'recall_' + label + '_mean': mean['Recall'].item(),
            set_name + 'recall_' + label + '_std': std['Recall'].item(),

            set_name + 'f1-score_' + label + '_mean': mean['F1-Score'].item(),
            set_name + 'f1-score_' + label + '_std': std['F1-Score'].item(),

            set_name + 'balanced-accuracy_' + label + '_mean': mean['Balanced Accuracy'].item(),
            set_name + 'balanced-accuracy_' + label + '_std': std['Balanced Accuracy'].item(),
        })


def print_save_results(set_name, results, results_cliffs, results_non_cliffs, save_to_csv, model_name=""):

    if save_to_csv:
        header = [
            "Molecules",
            "Loss_mean", "Loss_std",
            "ROC-AUC_mean", "ROC-AUC_std",
            "Accuracy_mean", "Accuracy_std",
            "Precision_mean", "Precision_std",
            "Recall_mean", "Recall_std",
            "F1-Score_mean", "F1-Score_std",
            "Balanced_Accuracy_mean", "Balanced_Accuracy_std",
        ]

        data = []

    for results_array, label in zip([results, results_non_cliffs, results_cliffs], ["all molecules", "non-cliff molecules", "cliff molecules"]):
        mean = np.mean(results_array, axis=0)
        std = np.std(results_array, axis=0)

        print(f"Performance on {set_name} set ({label}):")
        print(
            f"- {set_name}-Loss: mean={mean['Loss']:.4f} std={std['Loss']:.4f}")
        print(
            f"- ROC AUC: mean={mean['ROC-AUC']:.4f} std={std['ROC-AUC']:.4f}")
        print(
            f"- Accuracy: mean={mean['Accuracy']:.4f} std={std['Accuracy']:.4f}")
        print(
            f"- Precision: mean={mean['Precision']:.4f} std={std['Precision']:.4f}")
        print(f"- Recall: mean={mean['Recall']:.4f} std={std['Recall']:.4f}")
        print(
            f"- F1-Score: mean={mean['F1-Score']:.4f} std={std['F1-Score']:.4f}")
        print(
            f"- Balanced Accuracy: mean={mean['Balanced Accuracy']:.4f} std={std['Balanced Accuracy']:.4f}")

        print()

        if save_to_csv:
            data.append([
                label,
                mean['Loss'], std['Loss'],
                mean['ROC-AUC'], std['ROC-AUC'],
                mean['Accuracy'], std['Accuracy'],
                mean['Precision'], std['Precision'],
                mean['Recall'], std['Recall'],
                mean['F1-Score'], std['F1-Score'],
                mean['Balanced Accuracy'], std['Balanced Accuracy'],
            ])

        if save_to_csv:
            with open(f"results/Results_{set_name}_{model_name}.csv", mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)  # Write the header
                writer.writerows(data)


def save_results_test_cliff_groups(cliff_group_results, model_name):
    data = []

    for i, results in cliff_group_results.items():
        df_combined = pd.concat(results, axis=0)

        mean = df_combined.mean(axis=0)
        std = df_combined.std(axis=0)

        data.append([
            i,
            mean['Loss'], std['Loss'],
            mean['ROC-AUC'], std['ROC-AUC'],
            mean['Accuracy'], std['Accuracy'],
            mean['Precision'], std['Precision'],
            mean['Recall'], std['Recall'],
            mean['F1-Score'], std['F1-Score'],
            mean['Balanced Accuracy'], std['Balanced Accuracy'],
        ])

    with open(f"results/Results_Cliff_Groups_Test_{model_name}.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [
            "Cliff_Group",
            "Loss_mean", "Loss_std",
            "ROC-AUC_mean", "ROC-AUC_std",
            "Accuracy_mean", "Accuracy_std",
            "Precision_mean", "Precision_std",
            "Recall_mean", "Recall_std",
            "F1-Score_mean", "F1-Score_std",
            "Balanced_Accuracy_mean", "Balanced_Accuracy_std",
        ]
        writer.writerow(header)
        writer.writerows(data)


def compute_metrics(loader, network, loss_function=nn.BCEWithLogitsLoss(), train_loader=False):
    """
    Computes various performance metrics of a network.

    Parameters:
        loader (torch.utils.data.DataLoader): DataLoader with data to evaluate network on.
        network (torch.nn.Module): Network to be evaluated.
        loss_function (Callable): Loss function to be used for loss computation.
        train_loader (bool): Whether given loader is a Train loader.

    Returns:
        various metrics (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.eval()

    with torch.no_grad():
        outputs_total = []
        targets_total = []
        for batch in tqdm(loader):
            if not train_loader or not use_contrastive_learning:
                samples, targets = batch
            else:
                samples, targets, _, _ = batch

            samples = samples.to(device)
            targets = targets.to(device)

            outputs, emb = network(samples)
            outputs = outputs.squeeze(dim=1)
            outputs_total.append(outputs.cpu())
            targets_total.append(targets.cpu())

        outputs_total = torch.cat(outputs_total, dim=0)
        targets_total = torch.cat(targets_total, dim=0)

        loss = loss_function(outputs_total, targets_total)

        probabilities = torch.sigmoid(outputs_total)
        predictions = (probabilities >= 0.5).float()

        predictions_np = predictions.numpy()
        targets_np = targets_total.numpy()
        probabilities_np = probabilities.numpy()

        accuracy = accuracy_score(targets_np, predictions_np)
        precision = precision_score(targets_np, predictions_np)
        recall = recall_score(targets_np, predictions_np)
        f1 = f1_score(targets_np, predictions_np)
        roc_auc = roc_auc_score(targets_np, probabilities_np)
        balanced_acc = balanced_accuracy_score(targets_np, predictions_np)

        results = {
            "Loss": loss.item(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Balanced Accuracy": balanced_acc
        }

        return pd.DataFrame([results])


def create_sweep_config():
    # TODO: create docstring

    # https://docs.wandb.ai/guides/sweeps/sweep-config-keys

    sweep_config = {
        'method': 'bayes'
    }

    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'optimizer': {
            'values': ['adam']
        },
        'epochs': {
            'values': [20, 50, 70, 100]
        },
        'learning_rate': {
            'min': 1e-3,
            'max': 1e0
        },
        'batch_size': {
            'values': [56, 128]
        },
        'n_hidden_layers': {
            'values': [2, 3, 4, 6]
        },
        'n_hidden_units': {
            'values': [1024, 768, 512, 256, 128]
        },
        'activation_function': {
            'values': ["relu", "selu", "leaky_relu"]
        },
        'input_dropout': {
            'values': [0., 0.1, 0.2, 0.5, 0.6, 0.7]
        },
        'dropout': {
            'values': [0., 0.1, 0.2, 0.3, 0.5]
        },
        'alpha': {
            'min': 0.1,
            'max': 2.0
        },
        'margin': {
            'values': [1.]
        },
    }

    sweep_config['parameters'] = parameters_dict

    return sweep_config


def run_sweep(sweep_id, fcn, count):
    # TODO: create docstring
    wandb.agent(sweep_id, function=fcn, count=count)


def run_sweep_multiple_seeds(config=None):

    if use_contrastive_learning:
        if use_cosine_sim:
            contrastive_tag = "Cosine"
        else:
            contrastive_tag = "Manhattan"
    else:
        contrastive_tag = "BCE"

    run = wandb.init(config=config, tags=[dataset_folder, contrastive_tag])

    val_results_list = []
    val_cliffs_results_list = []
    val_non_cliffs_results_list = []
    test_results_list = []
    test_cliffs_results_list = []
    test_non_cliffs_results_list = []

    for current_seed in [12, 68, 94, 39, 7]:

        # sweep_config["parameters"]["current_seed"] = {"value": current_seed}
        wandb.config.update({"current_seed": current_seed},
                            allow_val_change=True)
        network, train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs = train_wandb(
            wandb.config)

        # train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs = build_dataset(
        #     wandb.config.batch_size, use_contrastive_learning=False)

        val_results = compute_metrics(val_loader, network)
        val_cliffs_results = compute_metrics(val_loader_cliffs, network)
        val_non_cliffs_results = compute_metrics(
            val_loader_non_cliffs, network)

        test_results = compute_metrics(test_loader, network)
        test_cliffs_results = compute_metrics(test_loader_cliffs, network)
        test_non_cliffs_results = compute_metrics(
            test_loader_non_cliffs, network)

        val_results_list.append(val_results)
        val_cliffs_results_list.append(val_cliffs_results)
        val_non_cliffs_results_list.append(val_non_cliffs_results)

        test_results_list.append(test_results)
        test_cliffs_results_list.append(test_cliffs_results)
        test_non_cliffs_results_list.append(test_non_cliffs_results)

    cumulated_val_results = pd.concat(val_results_list, ignore_index=True)
    cumulated_val_cliffs_results = pd.concat(
        val_cliffs_results_list, ignore_index=True)
    cumulated_val_non_cliffs_results = pd.concat(
        val_non_cliffs_results_list, ignore_index=True)

    cumulated_test_results = pd.concat(test_results_list, ignore_index=True)
    cumulated_test_cliffs_results = pd.concat(
        test_cliffs_results_list, ignore_index=True)
    cumulated_test_non_cliffs_results = pd.concat(
        test_non_cliffs_results_list, ignore_index=True)

    log_avg_results("val", cumulated_val_results,
                    cumulated_val_cliffs_results, cumulated_val_non_cliffs_results)
    log_avg_results("test", cumulated_test_results,
                    cumulated_test_cliffs_results, cumulated_test_non_cliffs_results)

    run.finish()


perform_add_preprocessing = False

# CHEMBL214_Ki
# CHEMBL233_Ki
# CHEMBL234_Ki
# CHEMBL244_Ki
# CHEMBL264_Ki
dataset_folder = "CHEMBL234_Ki"

df, df_train, df_val, df_test = preprocessing.preprocess_data(
    perform_add_preprocessing, dataset_folder=dataset_folder)

tune_wandb = True

train_eval_rf = False

# choose from: 'MLP', 'MLP Triplet Manhattan', 'MLP Triplet Cosine', None
load_model = None  # from seed-run 12

use_contrastive_learning = False
use_cosine_sim = False

train_losses_total = []
train_triplet_losses = []
train_basic_losses = []
val_losses = []


if __name__ == "__main__":

    if tune_wandb:
        wandb.login()
        sweep_config = create_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project="ActivityCliffs")

        run_sweep(sweep_id, run_sweep_multiple_seeds, 1)

    else:

        configs = [
            {  # MLP BCE
                'optimizer': 'adam',
                'epochs': 16,
                'learning_rate': 0.0041,
                'batch_size': 32,
                'n_hidden_layers': 2,
                'n_hidden_units': 1000,
                'activation_function': 'leaky_relu',
                'input_dropout': 0.5,
                'dropout': 0.25,
                'alpha': 0.0
            },
            {  # MLP Manhattan
                'optimizer': 'adam',
                'epochs': 34,
                'learning_rate': 0.0228,
                'batch_size': 128,
                'n_hidden_layers': 6,
                'n_hidden_units': 128,
                'activation_function': 'relu',
                'input_dropout': 0.2,
                'dropout': 0.1,
                'alpha': 0.5405
            },
            {  # MLP Cosine
                'optimizer': 'adam',
                'epochs': 26,
                'learning_rate': 0.1192,
                'batch_size': 128,
                'n_hidden_layers': 4,
                'n_hidden_units': 128,
                'activation_function': 'leaky_relu',
                'input_dropout': 0.7,
                'dropout': 0.1,
                'alpha': 1.4944
            }
        ]

        if (not load_model):
            if (use_contrastive_learning and use_cosine_sim):
                model_name = "MLP_Cosine"
                config_dict = configs[2]
            elif (use_contrastive_learning and not use_cosine_sim):
                model_name = "MLP_Manhattan"
                config_dict = configs[1]
            else:
                model_name = "MLP_BCE"
                config_dict = configs[0]

        val_results_list = []
        val_cliffs_results_list = []
        val_non_cliffs_results_list = []
        test_results_list = []
        test_cliffs_results_list = []
        test_non_cliffs_results_list = []

        # extract cliff groups of test set
        group_dict = preprocessing.get_cliff_groups_test(
            "data/" + dataset_folder + "/df_test.csv")

        # add cliff group as column to dataframe
        group_map = {idx: key for key, indices in group_dict.items()
                     for idx in indices}
        df_test_groups = df_test.copy()
        df_test_groups['cliff_group'] = df_test_groups.index.map(group_map)
        # NaN / missing values are non-cliffs
        df_test_groups = df_test_groups.dropna(subset=['cliff_group'])
        df_test_groups['cliff_group'] = df_test_groups['cliff_group'].astype(
            int)
        cliff_group_results = dict()

        for current_seed in [12, 68, 94, 39, 7]:

            train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs = build_dataset(
                config_dict['batch_size'], use_contrastive_learning=False)
            if train_eval_rf:
                model_name = "RF"
                val_results, val_cliffs_results, val_non_cliffs_results, test_results, test_cliffs_results, test_non_cliffs_results = train_rf(train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs,
                                                                                                                                               test_loader_cliffs, train_loader_non_cliffs, val_loader_non_cliffs, test_loader_non_cliffs)
            else:
                if load_model is None:
                    network = train(
                        config_dict, use_contrastive_learning=use_contrastive_learning, use_cosine_sim=use_cosine_sim, seed=current_seed)
                elif load_model == 'MLP':
                    network = torch.load(
                        'models/' + dataset_folder + '/baseline_mlp.pt', weights_only=False)
                elif load_model == 'MLP Triplet Manhattan':
                    network = torch.load(
                        'models/' + dataset_folder + '/mlp_triplet_manhattan.pt', weights_only=False)
                elif load_model == 'MLP Triplet Cosine':
                    network = torch.load(
                        'models/' + dataset_folder + '/mlp_triplet_cosine.pt', weights_only=False)
                else:
                    raise Exception("invalid flag combination")

                val_results = compute_metrics(val_loader, network)
                val_cliffs_results = compute_metrics(
                    val_loader_cliffs, network)
                val_non_cliffs_results = compute_metrics(
                    val_loader_non_cliffs, network)

                test_results = compute_metrics(test_loader, network)
                test_cliffs_results = compute_metrics(
                    test_loader_cliffs, network)
                test_non_cliffs_results = compute_metrics(
                    test_loader_non_cliffs, network)

                torch.save(network, 'models/' + dataset_folder + '/' +
                           model_name + "_seed" + str(current_seed) + ".pt")

            if not train_eval_rf:
                for i in range(min(df_test_groups['cliff_group']), max(df_test_groups['cliff_group']) + 1):
                    if (i not in cliff_group_results):
                        cliff_group_results[i] = []

                    filtered_df = df_test_groups[df_test_groups['cliff_group'] == i]
                    dataset = MoleculeACEDataset(
                        filtered_df['ecfp'], filtered_df['active'])
                    loader = DataLoader(dataset, shuffle=True,
                                        batch_size=config_dict['batch_size'])
                    results = compute_metrics(loader, network)
                    cliff_group_results[i].append(results)

            val_results_list.append(val_results)
            val_cliffs_results_list.append(val_cliffs_results)
            val_non_cliffs_results_list.append(val_non_cliffs_results)

            test_results_list.append(test_results)
            test_cliffs_results_list.append(test_cliffs_results)
            test_non_cliffs_results_list.append(test_non_cliffs_results)

        cumulated_val_results = pd.concat(val_results_list, ignore_index=True)
        cumulated_val_cliffs_results = pd.concat(
            val_cliffs_results_list, ignore_index=True)
        cumulated_val_non_cliffs_results = pd.concat(
            val_non_cliffs_results_list, ignore_index=True)

        cumulated_test_results = pd.concat(
            test_results_list, ignore_index=True)
        cumulated_test_cliffs_results = pd.concat(
            test_cliffs_results_list, ignore_index=True)
        cumulated_test_non_cliffs_results = pd.concat(
            test_non_cliffs_results_list, ignore_index=True)

        print_save_results("Validation", cumulated_val_results,
                           cumulated_val_cliffs_results, cumulated_val_non_cliffs_results, save_to_csv=True, model_name=model_name)
        print()
        print_save_results("Test", cumulated_test_results,
                           cumulated_test_cliffs_results, cumulated_test_non_cliffs_results, save_to_csv=True, model_name=model_name)

        if not train_eval_rf:
            save_results_test_cliff_groups(
                cliff_group_results, model_name=model_name)

        if load_model is None and not train_eval_rf:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))

            axes[0, 0].plot(train_losses_total)
            axes[0, 0].set_title('Total Train Losses')

            axes[0, 1].plot(val_losses)
            axes[0, 1].set_title('Validation Losses (BCE)')

            axes[1, 0].plot(train_basic_losses)
            axes[1, 0].set_title('BCE Losses Train')

            axes[1, 1].plot(train_triplet_losses)
            axes[1, 1].set_title('Triplet Losses Train')

            plt.tight_layout()
            plt.show()
