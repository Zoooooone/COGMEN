import argparse
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval


def plot(data, args):
    iter_num = len(data)
    plt.figure(dpi=200, figsize=(12, 9))

    for i in range(iter_num):
        epochs, train_losses, test_f1s = data.iloc[i]["epochs"], data.iloc[i]["train_losses"], data.iloc[i]["test_f1s"]
        window, gnn_nheads, hidden_size, seqcontext_nlayer = data.iloc[i]["wp"], data.iloc[i]["gnn_nheads"], data.iloc[i]["hidden_size"], data.iloc[i]["seqcontext_nlayer"]
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
    plt.savefig(f"results/img/plot_res_{args.dataset}_{args.gnn_nheads}heads.png")


def main(args):
    res = pd.read_csv("results/train_res.csv")
    res_heads = res[(res["gnn_nheads"] == args.gnn_nheads) & (res["dataset"] == args.dataset)]
    res_heads.loc[:, "train_losses"] = res_heads["train_losses"].apply(literal_eval)
    res_heads.loc[:, "test_f1s"] = res_heads["test_f1s"].apply(literal_eval)
    plot(res_heads, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot_res.py")
    parser.add_argument(
        "--dataset", type=str, help="Dataset that was used", default="iemocap_4"
    )
    parser.add_argument(
        "--gnn_nheads", type=int, help="hyperparameter GNN heads", default=5
    )

    args = parser.parse_args()
    main(args)
