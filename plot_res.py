import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from matplotlib import cm


def plot_gnn_heads(data, args):
    iter_num = len(data)
    plt.figure(dpi=200, figsize=(12, 9))

    for i in range(iter_num):
        epochs, train_losses, test_f1s = data.iloc[i][["epochs", "train_losses", "test_f1s"]]
        window, gnn_nheads, hidden_size, seqcontext_nlayer = data.iloc[i][["wp", "gnn_nheads", "hidden_size", "seqcontext_nlayer"]]
        best_test_f1 = max(test_f1s)
        best_test_f1_epoch = test_f1s.index(best_test_f1) + 1
        epochs_range = [i for i in range(1, epochs + 1)]
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label=f"w = {window}, heads = {gnn_nheads}, h_size = {hidden_size}, seqcontext = {seqcontext_nlayer}")
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, test_f1s, label=f"w = {window}, heads = {gnn_nheads}, h_size = {hidden_size}, seqcontext = {seqcontext_nlayer}")
        plt.plot(best_test_f1_epoch, best_test_f1, "r*", markersize=8)
        plt.text(best_test_f1_epoch - 1, best_test_f1 + 0.007, str(f"{best_test_f1:.3f}"), fontsize=5)

    plt.subplot(1, 2, 1)    
    plt.xlabel("epochs")
    plt.ylabel("losses")

    plt.subplot(1, 2, 2)
    plt.xlabel("epochs")
    plt.ylabel("F1")
    
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right")
    plt.savefig(f"results/img/gnn_heads/plot_res_{args.dataset}_{args.gnn_nheads}heads.png")


def plot_windows(data):
    def set_ticks_size(font_size):
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

    iter_num, font_size, marker_size = len(data), 20, 2
    plt.figure(dpi=200, figsize=(20, 20))
    grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    colors = cm.Blues(np.linspace(0, 1, iter_num))

    for i in range(iter_num):
        epochs, window, train_losses, test_f1s = data.iloc[i][["epochs", "wp", "train_losses", "test_f1s"]]
        epochs_range = [i for i in range(1, epochs + 1)]
        if window > 0:
            line_color = colors[i]
        elif window == 0:
            line_color = "red"
        else:
            line_color = "orange"
        
        plt.subplot(grid[0, 0])
        plt.title("loss curve", fontsize=font_size)
        plt.plot(epochs_range, train_losses, "o-", label=f"window = {window}", linewidth=1, color=line_color, markersize=marker_size)
        
        plt.subplot(grid[0, 1])
        plt.title("learning curve", fontsize=font_size)
        plt.plot(epochs_range, test_f1s, "o-", label=f"window = {window}", linewidth=1, color=line_color, markersize=marker_size)

        plt.subplot(grid[1, 0])
        plt.title("zoom in", fontsize=font_size)
        plt.plot(epochs_range, train_losses, "o-", label=f"window = {window}", linewidth=1, color=line_color, markersize=marker_size)
        plt.xlim((40, 55))
        plt.ylim((0.3, 0.75))

        plt.subplot(grid[1, 1])
        plt.title("zoom in", fontsize=font_size)
        plt.plot(epochs_range, test_f1s, "o-", label=f"window = {window}", linewidth=1, color=line_color, markersize=marker_size)
        plt.xlim((40, 55))
        plt.ylim((0.62, 0.68))

    for j in range(2):
        for k in range(2):
            if k == 0:
                plt.subplot(grid[j, k])
                plt.xlabel("epochs", fontsize=font_size)
                plt.ylabel("losses", fontsize=font_size)
            else:
                plt.subplot(grid[j, k])
                plt.xlabel("epochs", fontsize=font_size)
                plt.ylabel("F1", fontsize=font_size)
            set_ticks_size(font_size)

    plt.legend(loc=2, bbox_to_anchor=(1.02, 1.5), prop={"size": 12}) 
    plt.savefig("results/img/windows/plot_res_windows.png")


def plot_label_metrics(data, args):
    labels = {
        "hap": "happy",
        "sad": "sad",
        "neu": "neutral",
        "ang": "angry",
        "exc": "excited",
        "fru": "frustrated"
    }
    metrics = defaultdict(list)
    for dic in data:
        for emotion_label in dic.keys():
            metrics[emotion_label].append(dic[emotion_label]["f1-score"])

    plt.figure(dpi=200, figsize=(8, 6))
    epoch = len(metrics["hap"]) + 1
    epochs = np.arange(1, epoch)
    for emotion_label, series in metrics.items():
        series = np.array(series)
        plt.plot(epochs, series, label=labels[emotion_label])
        plt.fill_between(epochs, series - np.std(series), series + np.std(series), alpha=0.2)

    plt.xlabel("epochs")
    plt.ylabel("F1")
    plt.title(f"dataset: {args.dataset}, modalities: {args.modalities}")
    plt.legend()
    plt.savefig(f"results/img/label_metrics/label_metrics_{args.dataset}_{args.modalities}.png")


def main(args):
    res = pd.read_csv("results/train_res.csv")
    res.loc[:, "train_losses"] = res["train_losses"].apply(literal_eval)
    res.loc[:, "test_f1s"] = res["test_f1s"].apply(literal_eval)
    
    if args.plot == "gnn_heads":
        res_heads = res[(res["gnn_nheads"] == args.gnn_nheads) & (res["dataset"] == args.dataset) & (res["experiment"] == 1)]
        plot_gnn_heads(res_heads, args)

    if args.plot == "windows":
        res_windows = res[(res["gnn_nheads"] == 6) & (res["dataset"] == "iemocap") & (res["experiment"] == 2)]
        plot_windows(res_windows)

    if args.plot == "label_metrics":
        with open(f"results/label_metrics_pkl/label_metrics_{args.dataset}_{args.modalities}.pkl", "rb") as file:
            res_label_metrics = pickle.load(file)
        plot_label_metrics(res_label_metrics, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_res.py")
    parser.add_argument(
        "--plot", type=str, help="Types of figures", choices=["gnn_heads", "windows", "label_metrics"], default="gnn_heads"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset that was used", default="iemocap_4"
    )
    parser.add_argument(
        "--gnn_nheads", type=int, help="hyperparameter GNN heads", default=5
    )
    parser.add_argument(
        "--modalities", type=str, help="used modalities", choices=["a", "t", "v", "at", "av", "tv", "atv"]
    )

    args = parser.parse_args()
    main(args)
