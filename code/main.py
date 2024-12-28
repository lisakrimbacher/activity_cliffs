import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import random

from model import MoleculeACEDataset, MLP, train_rf
import preprocessing

# np.random.seed(12)
# random.seed(12)
# torch.manual_seed(12)
# torch.random.manual_seed(12)


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

    filtered_df_train = df_train[df_train['cliff_mol_binary'] == 1]
    filtered_df_val = df_val[df_val['cliff_mol_binary'] == 1]
    filtered_df_test = df_test[df_test['cliff_mol_binary'] == 1]

    # no similar molecules extracted, since this dataset with just cliffs is only used for validation,
    # not training with triplet loss!
    train_set_cliffs = MoleculeACEDataset(
        filtered_df_train['ecfp'], filtered_df_train['active'])
    val_set_cliffs = MoleculeACEDataset(
        filtered_df_val['ecfp'], filtered_df_val['active'])
    test_set_cliffs = MoleculeACEDataset(
        filtered_df_test['ecfp'], filtered_df_test['active'])

    train_loader_cliffs = DataLoader(
        train_set_cliffs, shuffle=True, batch_size=batch_size)
    val_loader_cliffs = DataLoader(
        val_set_cliffs, shuffle=False, batch_size=batch_size)
    test_loader_cliffs = DataLoader(
        test_set_cliffs, shuffle=False, batch_size=batch_size)

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

    return train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs


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
            train_loss, val_loss, accuracy, precision, recall, f1, roc_auc, balanced_acc, \
                val_loss_cliffs, accuracy_cliffs, precision_cliffs, recall_cliffs, f1_cliffs, \
                roc_auc_cliffs, balanced_acc_cliffs = \
                train_epoch(epoch, network, loaders, optimizer,
                            alpha=config['alpha'], use_contrastive_learning=use_contrastive_learning, use_cosine_sim=use_cosine_sim, seed=seed)
        else:
            train_loss, val_loss, accuracy, precision, recall, f1, roc_auc, balanced_acc, \
                val_loss_cliffs, accuracy_cliffs, precision_cliffs, recall_cliffs, f1_cliffs, \
                roc_auc_cliffs, balanced_acc_cliffs = \
                train_epoch(epoch, network, loaders, optimizer,
                            use_contrastive_learning=use_contrastive_learning, seed=seed)

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
    np.random.seed(seed + epoch_id)
    random.seed(seed + epoch_id)
    torch.manual_seed(seed + epoch_id)
    torch.random.manual_seed(seed + epoch_id)

    train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs = loaders

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

            train_triplet_losses.append(triplet_loss.cpu().detach().numpy())
            train_basic_losses.append(basic_loss.cpu().detach().numpy())

        else:
            samples, targets = batch
            samples = samples.to(device)
            targets = targets.to(device)

            output, emb = network(samples)
            output = output.squeeze(dim=1)
            loss = loss_function(output, targets)

        train_losses_total.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss, accuracy, precision, recall, f1, roc_auc, balanced_acc = compute_metrics(
        val_loader, network)
    val_loss_cliffs, accuracy_cliffs, precision_cliffs, recall_cliffs, f1_cliffs, roc_auc_cliffs, balanced_acc_cliffs = \
        compute_metrics(val_loader_cliffs, network)
    train_loss, _, _, _, _, _, _ = compute_metrics(
        train_loader, network, train_loader=True)

    val_losses.append(val_loss)

    return train_loss, val_loss, accuracy, precision, recall, f1, roc_auc, balanced_acc, val_loss_cliffs, accuracy_cliffs, precision_cliffs, recall_cliffs, f1_cliffs, roc_auc_cliffs, balanced_acc_cliffs


def print_results(set_name, loss, accuracy, precision, recall, f1, roc_auc, balanced_acc, loss_cliffs, accuracy_cliffs,
                  precision_cliffs, recall_cliffs, f1_cliffs, roc_auc_cliffs, balanced_acc_cliffs):
    """
    Prints given metrics.

    Parameters:
        set_name (string): Name of the evaluated dataset, e.g. 'Validation' or 'Test'.
        various metrics (float)
    """
    print()

    print(f"Performance on {set_name} set (all molecules): ")
    print(f"- {set_name}-Loss: ", loss)
    print("- ROC AUC: ", roc_auc)
    print("- Accuracy: ", accuracy)
    print("- Precision: ", precision)
    print("- Recall: ", recall)
    print("- F1-Score: ", f1)
    print("- Balanced Accuracy: ", balanced_acc)
    print()

    print(f"Performance on {set_name} set (cliff molecules): ")
    print(f"- {set_name}-Loss: ", loss_cliffs)
    print("- ROC AUC: ", roc_auc_cliffs)
    print("- Accuracy: ", accuracy_cliffs)
    print("- Precision: ", precision_cliffs)
    print("- Recall: ", recall_cliffs)
    print("- F1-Score: ", f1_cliffs)
    print("- Balanced Accuracy: ", balanced_acc_cliffs)
    print()

def print_results_array(set_name, results):

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    # test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc, test_loss_cliffs,
    #             test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs

    print(f"Performance on {set_name} set (all molecules): ")
    print(f"- {set_name}-Loss: mean={mean[0]:.4f} std={std[0]:.4f}")
    print(f"- ROC AUC: mean={mean[5]:.4f} std={std[5]:.4f}")
    print(f"- Accuracy: mean={mean[1]:.4f} std={std[1]:.4f}")
    print(f"- Precision: mean={mean[2]:.4f} std={std[2]:.4f}")
    print(f"- Recall: mean={mean[3]:.4f} std={std[3]:.4f}")
    print(f"- F1-Score: mean={mean[4]:.4f} std={std[4]:.4f}")
    print(f"- Balanced Accuracy: mean={mean[6]:.4f} std={std[6]:.4f}")
    print()

    print(f"Performance on {set_name} set (cliff molecules): ")
    print(f"- {set_name}-Loss: mean={mean[7]:.4f} std={std[7]:.4f}")
    print(f"- ROC AUC: mean={mean[12]:.4f} std={std[12]:.4f}")
    print(f"- Accuracy: mean={mean[8]:.4f} std={std[8]:.4f}")
    print(f"- Precision: mean={mean[9]:.4f} std={std[9]:.4f}")
    print(f"- Recall: mean={mean[10]:.4f} std={std[10]:.4f}")
    print(f"- F1-Score: mean={mean[11]:.4f} std={std[11]:.4f}")
    print(f"- Balanced Accuracy: mean={mean[13]:.4f} std={std[13]:.4f}")
    print()

    print(f"Difference between all molecules and activity cliffs: ")
    print(f"- {set_name}-Loss: {(mean[7] - mean[0]):.4f}")
    print(f"- ROC AUC: {(mean[5] - mean[12]):.4f}")
    print(f"- Accuracy: {(mean[1] - mean[8]):.4f}")
    print(f"- Precision: {(mean[2] - mean[9]):.4f}")
    print(f"- Recall: {(mean[3] - mean[10]):.4f}")
    print(f"- F1-Score: {(mean[4] - mean[11]):.4f}")
    print(f"- Balanced Accuracy: {(mean[6] - mean[13]):.4f}")


    print()
    print(results)

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

        val_loss = loss_function(outputs_total, targets_total)

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

        return val_loss.item(), accuracy, precision, recall, f1, roc_auc, balanced_acc


perform_add_preprocessing = False
df, df_train, df_val, df_test = preprocessing.preprocess_data(
    perform_add_preprocessing, path="data/CHEMBL234_Ki.csv")

train_eval_rf = False
load_model = None  # choose from: 'MLP', 'MLP Triplet Manhattan', 'MLP Triplet Cosine', None
use_contrastive_learning = False
use_cosine_sim = False

train_losses_total = []
train_triplet_losses = []
train_basic_losses = []
val_losses = []


if __name__ == "__main__":

    configs = [
        {
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
        {
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
        {
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

    config_dict = configs[0]
    results = []

    for current_seed in [12]:#, 68, 94, 39, 7]:

        # config_dict = {
        #     'optimizer': 'adam',
        #     'epochs': 16,
        #     'learning_rate': 0.0041,
        #     'batch_size': 32,
        #     'n_hidden_layers': 2,
        #     'n_hidden_units': 1000,
        #     'activation_function': 'leaky_relu',
        #     'input_dropout': 0.5,
        #     'dropout': 0.25,
        #     'alpha': 0.0
        # }

        train_loader, val_loader, test_loader, train_loader_cliffs, val_loader_cliffs, test_loader_cliffs = build_dataset(
            config_dict['batch_size'], use_contrastive_learning=False)
        if train_eval_rf:
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_balanced_acc, \
                val_loss_cliffs, val_accuracy_cliffs, val_precision_cliffs, val_recall_cliffs, val_f1_cliffs, val_roc_auc_cliffs, val_balanced_acc_cliffs, \
                test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc, \
                test_loss_cliffs, test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs = \
                train_rf(train_loader, val_loader, test_loader,
                        train_loader_cliffs, val_loader_cliffs, test_loader_cliffs)
        else:
            if load_model is None:
                network = train(
                    config_dict, use_contrastive_learning=use_contrastive_learning, use_cosine_sim=use_cosine_sim, seed=current_seed)
            elif load_model == 'MLP':
                network = torch.load('models/baseline_mlp.pt', weights_only=False)
            elif load_model == 'MLP Triplet Manhattan':
                network = torch.load(
                    'models/mlp_triplet_manhattan.pt', weights_only=False)
            elif load_model == 'MLP Triplet Cosine':
                network = torch.load(
                    'models/mlp_triplet_cosine.pt', weights_only=False)
            else:
                raise Exception("invalid flag combination")

            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_balanced_acc = \
                compute_metrics(val_loader, network)
            val_loss_cliffs, val_accuracy_cliffs, val_precision_cliffs, val_recall_cliffs, val_f1_cliffs, val_roc_auc_cliffs, val_balanced_acc_cliffs = \
                compute_metrics(val_loader_cliffs, network)
            test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc = \
                compute_metrics(test_loader, network)
            test_loss_cliffs, test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs = \
                compute_metrics(test_loader_cliffs, network)
            
        results.append(np.array([test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc, test_loss_cliffs,
                test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs]))

    # print_results("Validation", val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_balanced_acc, val_loss_cliffs,
    #             val_accuracy_cliffs, val_precision_cliffs, val_recall_cliffs, val_f1_cliffs, val_roc_auc_cliffs, val_balanced_acc_cliffs)
    # print_results("Test", test_loss, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_balanced_acc, test_loss_cliffs,
    #             test_accuracy_cliffs, test_precision_cliffs, test_recall_cliffs, test_f1_cliffs, test_roc_auc_cliffs, test_balanced_acc_cliffs)
    
    print_results_array("Test", np.array(results))

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
