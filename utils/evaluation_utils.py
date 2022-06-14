from statistics import mean, stdev

import matplotlib.pyplot as plt
import torch
from numpy.random import randint
from sklearn import metrics as skl
from torch.nn.functional import binary_cross_entropy


def bootstrap_statistic(predictions, interaction_labels):
    # TODO: how to handle the overrepresentative class of 0 = no interaction?
    # by just randomly sampling from the interaction_labels and predictions, it can happen
    # that all the interactions in the bootstrap set are 0
    # if the model just always predicts 0, this means, that it would predict many correct things
    # maybe leading to a false assumption on the error

    # And there are other problems, like the precision zero-division
    # or invalid value encoutered in true_divide ....

    stacked_samples = torch.stack((predictions, interaction_labels), dim=0)
    num_predictions = len(predictions)

    stats = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'aucpr': [], 'loss': []}  # , 'aucroc': []}
    # does not really make sense for loss but ...
    for i in range(10000):
        bootstrap_set = stacked_samples[:, randint(num_predictions, size=num_predictions)]
        bootstrap_predictions = bootstrap_set[0, :]
        bootstrap_labels = bootstrap_set[1, :]
        boot_statistic = statistics(bootstrap_predictions, bootstrap_labels)
        stats = {key: stats[key] + [boot_statistic[key]] for key in stats}

    return {key: (mean(stat), stdev(stat), 1.96 * stdev(stat)) for key, stat in stats.items()}


def statistics(predictions, interaction_labels):
    with torch.no_grad():
        loss = binary_cross_entropy(predictions, interaction_labels).item()

        predictions_boundary = .5
        encountered_pairs = len(predictions)
        binary_predictions = (predictions_boundary * torch.ones(encountered_pairs) < predictions).float()
        tn, fp, fn, tp = skl.confusion_matrix(interaction_labels, binary_predictions, labels=[0, 1]).ravel()

        acc = skl.accuracy_score(interaction_labels, binary_predictions)
        precision, recall, f1, _ = skl.precision_recall_fscore_support(interaction_labels, binary_predictions,
                                                                       average='binary', zero_division=0)

        aucpr = skl.auc_pr(interaction_labels, predictions)
        # aucroc = auc_roc(interaction_labels, predictions)

    return {'loss': loss / encountered_pairs, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'acc': acc,
            'precision': precision, 'recall': recall, 'f1': f1, 'aucpr': aucpr}  # , 'aucroc': aucroc}


def plot_all_statistics(predictions, interaction_labels, path):
    precision, recall, aucpr = plot_pr(predictions, interaction_labels, path)
    fpr, tpr, aucroc = plot_roc(predictions, interaction_labels, path)
    plot_label_dist(predictions, interaction_labels, path)

    # with open(f'{path}output.txt', 'w+') as f:
    #    output_format = 'Precision={:<8.6}, Recall={:<8.6}, AUPR={:<8.6}, AUROC={:8.6}'
    #    f.write(output_format.format(*[precision, recall, aucpr, aucroc]))


def plot_label_dist(predictions, interaction_labels, path):
    pred_contact = predictions[interaction_labels == 1]
    pred_no_contact = predictions[interaction_labels == 0]
    real_contact = interaction_labels[interaction_labels == 1]
    real_no_contact = interaction_labels[interaction_labels == 0]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle("Distribution of Predictions/Labels")
    ax1.hist(pred_contact)
    ax1.set_xlim(0, 1)
    ax1.set_title("Contact")
    ax1.set_xlabel("predicted")

    ax2.hist(pred_no_contact)
    ax2.set_xlim(0, 1)
    ax2.set_title("No Contact")
    ax2.set_xlabel("predicted")

    ax3.hist(real_contact)
    ax3.set_xlim(0, 1)
    ax3.set_title("Contact")
    ax3.set_xlabel("real")

    ax4.hist(real_no_contact)
    ax4.set_xlim(0, 1)
    ax4.set_title("No Contact")
    ax4.set_xlabel("real")

    plt.savefig(path + "dist.png")
    plt.close()


def plot_pr(predictions, interaction_labels, path):
    precision, recall, _ = skl.precision_recall_curve(interaction_labels, predictions)
    aucpr = skl.auc_pr(interaction_labels, predictions)

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"PR (AUCPR: {aucpr:.3})")
    plt.savefig(path + "aucpr.png")
    plt.close()

    return precision, recall, aucpr


def plot_roc(predictions, interaction_labels, path):
    fpr, tpr, _ = skl.roc_curve(interaction_labels, predictions)
    aucroc = skl.auc_roc(interaction_labels, predictions)

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"ROC (AUCROC: {aucroc:.3})")
    plt.savefig(path + "aucroc.png")
    plt.close()

    return fpr, tpr, aucroc

# def for_later_use():
#     boot = bootstrap_statistic(*eva)
#     self.boot_epoch_report_fmt = '# Finished Epoch {:<2} / {:<2}: Loss={:<10}, Acc={:<14%}, ' \
#                                  'Precision={:<16}, Recall={:<16}, F1={:<16}, AUPR={:<16}'  # , AUROC={:16}'
#
#     def helper_formatter(stat_name):
#         return f"{statstics[stat_name]:<5.3} ({boot[stat_name][0]:<4.2} +- {boot[stat_name][1]:<3.2})"
#
#     print(self.epoch_report_fmt.format(*[epoch, self.num_epochs, helper_formatter('loss'), helper_formatter('acc'),
#                                          helper_formatter('precision'), helper_formatter('recall'),
#                                          helper_formatter('f1'),
#                                          helper_formatter('aucpr')]))  # , helper_formatter('aucroc')]))
