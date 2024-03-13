import data
import models
import utils
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mtick
import matplotlib.patheffects as PathEffects
import ast
import oapackage
import random

def plot_cifar_distribution():
    
    loaders, testloaders = data.cifar_data()
    loaders = loaders[0:10]
    testloaders = testloaders[0:10]

    sns.set_theme(style="white", font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 6), sharey=True, gridspec_kw={"wspace": 0.03})
    cifar_classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    classes_counts = pd.DataFrame([pd.Series(np.concatenate([batch[1].numpy() for batch in loader])).value_counts() for loader in loaders]).fillna(0).reset_index(drop=True)
    classes_counts = classes_counts.reindex(sorted(classes_counts.columns), axis=1)
    classes_counts["client"] = [f"Client {value}" for value in reversed(range(1, 11))]
    classes_counts.columns = cifar_classes + ["client"]
    classes_counts.plot(x="client", y=cifar_classes, kind="barh", stacked=True, width=0.8, ax=axes[0], colormap="tab10")

    classes_test = pd.DataFrame([pd.Series(np.concatenate([batch[1].numpy() for batch in test])).value_counts() for test in testloaders]).fillna(0).reset_index(drop=True)
    classes_test = classes_test.reindex(sorted(classes_test.columns), axis=1)
    classes_test["client"] = [f"Client {value}" for value in reversed(range(1, 11))]
    classes_test.columns = cifar_classes + ["client"]
    classes_test.plot(x="client", y=cifar_classes, kind="barh", stacked=True, width=0.8, ax=axes[1], colormap="tab10")

    axes[0].legend(title="Class", loc=(2.05, 0.04), handlelength=1.5, handleheight=2, handletextpad=0.3)
    axes[1].legend().remove()

    axes[0].set_xlabel("Number of Datapoints") 
    axes[1].set_xlabel("Number of Datapoints") 
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[0].set_title(f"Distribution of CIFAR-10 Classes\nin the Training Dataset on 10 Example Clients", fontsize=16)
    axes[1].set_title(f"Distribution of CIFAR-10 Classes\nin the Test Dataset on 10 Example Clients", fontsize=16)
    axes[0].grid(alpha=0.4, axis="x")
    axes[1].grid(alpha=0.4, axis="x")

    plt.savefig(f"figures/cifar_class_distribution.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_femnist_distribution():
    loaders, _ = data.femnist_data(path_to_data_folder="E:/Folders/femnist_data")
    loaders = loaders[11:21]

    classes_counts = pd.DataFrame([pd.Series(np.concatenate([batch[1].numpy() for batch in loader])).value_counts() for loader in loaders]).fillna(0).reset_index(drop=True)
    classes_counts = classes_counts.reindex(sorted(classes_counts.columns), axis=1)
    classes_counts["client"] = [f"Client {value}" for value in reversed(range(1, 11))]

    sns.set_theme(style="white", font_scale=1.25)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    mpn65 = ['#ff0029', '#377eb8', '#66a61e', '#984ea3', '#00d2d5', '#ff7f00', '#af8d00',
            '#7f80cd', '#b3e900', '#c42e60', '#a65628', '#f781bf', '#8dd3c7', '#bebada',
            '#fb8072', '#80b1d3', '#fdb462', '#fccde5', '#bc80bd', '#ffed6f', '#c4eaff',
            '#cf8c00', '#1b9e77', '#d95f02', '#e7298a', '#e6ab02', '#a6761d', '#0097ff',
            '#00d067', '#000000', '#252525', '#525252', '#737373', '#969696', '#bdbdbd',
            '#f43600', '#4ba93b', '#5779bb', '#927acc', '#97ee3f', '#bf3947', '#9f5b00',
            '#f48758', '#8caed6', '#f2b94f', '#5e5e4a', '#e43872', '#d9b100', '#9d7a00',
            '#698cff', '#d9d9d9', '#00d27e', '#d06800', '#009f82', '#c49200', '#cbe8ff',
            '#fecddf', '#c27eb6', '#8cd2ce', '#c4b8d9', '#f883b0', '#a49100']
    random.seed(7)
    random.shuffle(mpn65)

    classes_counts.plot(x="client", y=list(range(62)), kind="barh", stacked=True, width=0.8, color=mpn65, ax=ax)
    ax.get_legend().remove()

    ax.set_xlabel("Number of Datapoints") 
    ax.set_ylabel("")
    ax.set_title(f"Distribution of FEMNIST Classes on 10 Example Clients", fontsize=16)
    ax.grid(alpha=0.4, axis="x")
    
    plt.savefig(f"figures/femnist_class_distribution.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_download_size():
    
    downloads = pd.read_csv("results/cifar_downloads.csv")
    sns.set_theme(style='white', font_scale=1.25)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    sns.barplot(data=downloads, 
                x="Method", 
                y="Public dataset size", 
                ax=ax, 
                hatch="/", 
                edgecolor="black", 
                color="white", 
                alpha=0.3)
    sns.barplot(data=downloads, 
                x="Method", 
                y="Download size (bytes)", 
                hue="Method",
                palette=["#000000", "#e34a33", "#7a0177", "#c51b8a"],
                ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Client Download Size (Bytes)");
    ax.text(x=0.75, y=0.8, s="* Public\ndataset\nsize", alpha=0.6, horizontalalignment="right", fontsize=17, transform=ax.transAxes);
    ax.set_title("Download Size for each Client on CIFAR-10", fontsize=16);
    ax.set_xticklabels(["FedAvg", "Sparsification", "One-shot\n(MA-Echo)", "Federated\nDistillation"]);

    plt.savefig("figures/download_size_cifar.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_sparsification_pareto():
    
    all_results = pd.read_csv("results/results.csv")
    all_results = all_results[all_results["keep_first_last"]==False]
    extra_topk = pd.read_csv("results/threshold_vs_topk.csv").iloc[:,:-1]
    fedavg = pd.read_csv("results/fedavg_results.csv")
    fedavg = fedavg.rename(columns={"client_bytes":"bytes_size"})
    fedavg["spars_label"] = "FedAvg"
    all_results = pd.concat([all_results, extra_topk, fedavg]).drop_duplicates().reset_index(drop=True)
    all_results["max_accuracy"] = all_results["accs"].apply(lambda row: max([value[1] for value in ast.literal_eval(row)]))

    palette = ["#000000", '#2b8cbe', '#e34a33', '#31a354']
    palette_dict = {"FedAvg":"#000000", "Top-k":'#b30000', "Random":'#045a8d', "Threshold":'#006837'}

    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, 
                            nrows=2, 
                            figsize=(14, 12),
                            gridspec_kw={"wspace": 0.15, "hspace":0.2})

    for i, dataset in enumerate([("FEMNIST", 30000), ("CIFAR-10", 70000)]):

        data = all_results[all_results["dataset"]==dataset[0]].reset_index(drop=True)

        pareto=oapackage.ParetoDoubleLong()
        for index in range(0, data.shape[0]):
            solution=oapackage.doubleVector((-data.loc[index, "bytes_size"], data.loc[index, "max_accuracy"]))
            pareto.addvalue(solution, index)
        optimal_solutions=data.loc[pareto.allindices(),:]

        sns.lineplot(data=optimal_solutions,
                     x="bytes_size", 
                     y="max_accuracy",
                     color="black",
                     lw=2.5,
                     alpha=0.8,
                     zorder=-100,
                     ax=axes[i, 0])

        sns.scatterplot(data=data, 
                     x="bytes_size", 
                     y="max_accuracy", 
                     hue="approach", 
                     hue_order=["FedAvg", "Random", "Top-k", "Threshold"],
                     palette=palette,
                     s=200,
                     alpha=1,
                     ax=axes[i, 0])

        sns.scatterplot(data=optimal_solutions, 
             x="bytes_size", 
             y="max_accuracy", 
             hue="approach", 
             hue_order=["Top-k", "Threshold"],
             palette=palette[2:],
             s=200,
             alpha=1,
             ax=axes[i, 1])

        sns.lineplot(data=optimal_solutions,
                 x="bytes_size", 
                 y="max_accuracy",
                 color="black",
                 lw=2.5,
                 alpha=0.8,
                 zorder=-100,
                 ax=axes[i, 1])

        for _, row in optimal_solutions.iterrows():
            colour = palette_dict[row["approach"]]
            if row["spars_label"]=="0.001 (~29.4%)":
                xshift=-200000
                yshift=-1.5
            elif row["spars_label"]=="50%" and dataset[0]=="CIFAR-10":
                xshift=-100000
                yshift=-1.5
            elif row["spars_label"]=="3%" and dataset[0]=="CIFAR-10":
                xshift=-150000
                yshift=0
            else:
                xshift=dataset[1]
                yshift=-0.3
            txt = axes[i, 1].text(x=row["bytes_size"]+xshift, 
                         y=row["max_accuracy"]+yshift, 
                         s=row["spars_label"], 
                         c=colour, 
                         weight='bold', 
                         horizontalalignment="left", 
                         size=13, 
                         color='black', 
                         zorder=100)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

        axes[i, 0].set_ylabel("Highest Accuracy Achieved")
        axes[i, 1].set_ylabel("")

        for x in range(2):
            axes[i, x].grid(alpha=0.4)
            axes[i, x].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i, x].legend(title="Sparsification\nApproach", facecolor = 'white', framealpha=1.0, loc="lower right");
            axes[1, x].set_xlabel("Bytes Uploaded by a Client")
            axes[0, x].set_xlabel("")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            axes[x, 0].legend([handles[0]] + ["Sparsification", "Approach"] + handles[1:4],
                      [labels[0]] + ["", ""] + labels[1:4],
                       handler_map={str: LegendTitle({'fontsize': 15})},
                       facecolor="white",
                       framealpha=1,
                       loc="lower right")

    axes[0, 0].set_title("Bytes Uploaded vs Accuracy\nfor Sparsification Approaches on FEMNIST")
    axes[0, 1].set_title("Pareto Optimal Sparsification Approaches on FEMNIST")
    axes[1, 0].set_title("Bytes Uploaded vs Accuracy\nfor Sparsification Approaches on CIFAR-10")
    axes[1, 1].set_title("Pareto Optimal Sparsification Approaches on CIFAR-10")

    plt.savefig(f"figures/pareto_front_sparsification.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white");
    
def plot_combined_pareto():
    
    combined_results = pd.read_csv("results/combined_results.csv")
    combined_results["Total Bytes Uploaded"] = combined_results["Total Bytes Uploaded"].astype(np.float64)
    combined_results["Client Bytes Uploaded"] = combined_results["Client Bytes Uploaded"].astype(np.float64)
    
    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, 
                            nrows=2, 
                            figsize=(16, 12),
                            gridspec_kw={"wspace": 0.12, "hspace":0.2})

    for i, dataset in enumerate(["FEMNIST", "CIFAR-10"]):
        data = combined_results[combined_results["Dataset"]==dataset].reset_index(drop=True)
        for x, variable in enumerate(["Client Bytes Uploaded", "Total Bytes Uploaded"]):

            pareto=oapackage.ParetoDoubleLong()
            for index in range(0, data.shape[0]):
                solution=oapackage.doubleVector((-data.loc[index, variable], data.loc[index, "Accuracy"]))
                pareto.addvalue(solution, index)
            optimal_solutions=data.loc[pareto.allindices(),:]

            sns.lineplot(data=optimal_solutions,
                 x=variable, 
                 y="Accuracy",
                 color="black",
                 lw=2.5,
                 alpha=0.8,
                 zorder=-100,
                 ax=axes[i, x])

            sns.scatterplot(data=data[data["Type"]=="Baseline"], 
                 x=variable, 
                 y="Accuracy", 
                 hue="Method", 
                 marker="X",
                 color="black",
                 palette=["#000000"],
                 s=200,
                 alpha=1,
                 ax=axes[i, x])

            sns.scatterplot(data=data[data["Type"]=="Sparsification"], 
                 x=variable, 
                 y="Accuracy", 
                 hue="Method", 
                 marker="o",
                 palette=["#e34a33", "#fc8d59", "#fdbb84"],
                 s=200,
                 alpha=1,
                 ax=axes[i, x])
                #['#b30000', '#e34a33', "#fc8d59", '#fdbb84', '#fdd49e', '#ebdcc3']
            sns.scatterplot(data=data[data["Type"]=="Distillation"], 
                 x=variable, 
                 y="Accuracy", 
                 hue="Method", 
                 palette=["#c51b8a", "#fa9fb5"],
                 marker="s",
                 s=200,
                 alpha=1,
                 ax=axes[i, x])

            sns.scatterplot(data=data[data["Type"]=="One-shot"], 
                 x=variable, 
                 y="Accuracy", 
                 hue="Method", 
                 palette=["#08519c", "#6baed6", "#bdd7e7"],
                 marker="^",
                 s=250,
                 alpha=1,
                 ax=axes[i, x])

            axes[i, x].grid(alpha=0.4)
            axes[i, x].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i, x].set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend([handles[0]] + ["Sparsification"] + handles[1:4] + ["Federated distillation"] + handles[4:6] + ["One-shot"] + handles[6:10],
                 [labels[0]] + [""] + labels[1:4] + [""] + ["Distillation (10 (CIFAR) \n or 4 (FEMNIST) rounds)", "Distillation (2 rounds)"] + [""] + labels[6:10],
                   handler_map={str: LegendTitle({'fontsize': 15})},
                   facecolor="white",
                   framealpha=1,
                   borderpad=0.8,
                   loc=(0.3, 0.03))

    axes[0, 0].legend().remove()
    axes[0, 1].legend().remove()
    axes[1, 0].legend().remove()
    axes[0, 0].set_xlabel("Client Bytes Uploaded (FEMNIST)")
    axes[0, 1].set_xlabel("Total Bytes Uploaded (FEMNIST)")
    axes[1, 0].set_xlabel("Client Bytes Uploaded (CIFAR-10)")
    axes[1, 1].set_xlabel("Total Bytes Uploaded (CIFAR-10)")
    axes[0, 0].set_ylabel("Accuracy on FEMNIST")
    axes[1, 0].set_ylabel("Accuracy on CIFAR-10")

    axes[0, 0].set_title("Comparison of Methods on Accuracy vs\nBytes Uploaded by a Client", fontsize=16)
    axes[0, 1].set_title("Comparison of Methods on Accuracy vs\nTotal Bytes Uploaded by all Clients across all Rounds", fontsize=16)        

    plt.savefig(f"figures/pareto_front_combined.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white");

def plot_keep_first_last_comparison():

    results = pd.read_csv("results/results.csv")
    results = results[results["dataset"]=="FEMNIST"]
    results["accs"] = results["accs"].apply(lambda row: ast.literal_eval(row))
    results = results.explode("accs").reset_index(drop=True)
    results["round"] = results["accs"].apply(lambda row: row[0])
    results["accuracy"] = results["accs"].apply(lambda row: row[1])

    baseline_acc = 80
    ylim = [0, 85]
    linewidth = 4

    sns.set_theme(style='white', font_scale=1.25)
    fig, ax = plt.subplots(figsize=(8, 6))

    topk_results = results[(results["sparsify_by"]==0.3) & (results["approach"]=="Top-k")]
    sns.lineplot(data=topk_results, 
                 x="round", 
                 y="accuracy", 
                 color="#e34a33",
                 style="keep_first_last", 
                 dashes=["", (2, 1)],
                 linewidth=4,
                 ax=ax)

    threshold_results = results[(results["sparsify_by"]==0.003) & (results["approach"]=="Threshold")]
    sns.lineplot(data=threshold_results, 
                 x="round", 
                 y="accuracy", 
                 color="#31a354",
                 style="keep_first_last",
                 dashes=["", (2, 1)],
                 linewidth=4,
                 ax=ax)

    random_results = results[(results["sparsify_by"]==0.05) & (results["approach"]=="Random")]
    sns.lineplot(data=random_results, 
                 x="round", 
                 y="accuracy", 
                 color="#2b8cbe",
                 style="keep_first_last",
                 dashes=["", (2, 1)],
                 linewidth=4,
                 ax=ax)

    ax.grid(alpha=0.4)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(["Top-k (30%)"] + handles[0:2] + ["Threshold (0.003)"] + handles[2:4] + ["Random (5%)"] + handles[4:6],
                 [""] + ["False", "True          "] + [""] + labels[2:4] + [""] + labels[4:6],
                handler_map={str: LegendTitle({'fontsize': 14})},
       facecolor="white",
       framealpha=1,
       borderpad=0.5,
       loc=(0.005, 0.435))
    ax.set_title("Results when forcing (or not) the selection\nof the first and last layers of the model", fontsize=16);
    ax.set_xlabel("Federated Learning Round")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"figures/keep_first_last_comparison.png", 
            dpi=300,
            bbox_inches="tight",
            facecolor="white")
    
def plot_all_accuracy_rounds():
    
    results = pd.read_csv("results/results.csv")
    results["accs"] = results["accs"].apply(lambda row: ast.literal_eval(row))
    results = results.explode("accs").reset_index(drop=True)
    results["round"] = results["accs"].apply(lambda row: row[0])
    results["accuracy"] = results["accs"].apply(lambda row: row[1])
    results = results[results["keep_first_last"]==False]

    fedavg = pd.read_csv("results/fedavg_results.csv")
    fedavg["accs"] = fedavg["accs"].apply(lambda row: ast.literal_eval(row))
    fedavg_femnist = [value[1] for value in fedavg[fedavg["dataset"]=="FEMNIST"]["accs"].values[0]]
    fedavg_cifar = [value[1] for value in fedavg[fedavg["dataset"]=="CIFAR-10"]["accs"].values[0]]

    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=3, 
                             nrows=2, 
                             figsize=(8*3, 6*2),
                             gridspec_kw={"wspace": 0.11, "hspace":0.18})

    sns.lineplot(x=range(1, len(fedavg_femnist)+1),
                 y=fedavg_femnist,
                 color="black",
                 linewidth=4,
                 label="FedAvg",
                 zorder=100,
                 ax=axes[0, 0])
    sns.lineplot(data=results[(results["approach"]=="Random") & (results["dataset"]=="FEMNIST")].sort_values("spars_label", ascending=False), 
                 x="round", 
                 y="accuracy", 
                 hue="spars_label", 
                 palette=['#045a8d', '#2b8cbe', '#74a9cf', '#a6bddb', '#d0d1e6', '#d7d8e0'],
                 hue_order=["50%", "30%", "10%", "5%", "3%", "1%"],
                 linewidth=4,
                 ax=axes[0, 0])
    axes[0, 0].set_title("Random Sparsification on FEMNIST", fontsize=16)

    sns.lineplot(x=range(1, len(fedavg_femnist)+1),
                 y=fedavg_femnist,
                 color="black",
                 linewidth=4,
                 label="FedAvg",
                 zorder=100,
                 ax=axes[0, 1])
    sns.lineplot(data=results[(results["approach"]=="Top-k") & (results["dataset"]=="FEMNIST")].sort_values("spars_label", ascending=False), 
                 x="round", 
                 y="accuracy", 
                 hue="spars_label", 
                 palette=['#b30000', '#e34a33', '#fc8d59', '#fdbb84', '#fdd49e', '#ebdcc3'],
                 hue_order=["50%", "30%", "10%", "5%", "3%", "1%"],
                 linewidth=4,
                 ax=axes[0, 1])
    axes[0, 1].set_title("Top-k Sparsification on FEMNIST", fontsize=16)

    sns.lineplot(x=range(1, len(fedavg_femnist)+1),
                 y=fedavg_femnist,
                 color="black",
                 linewidth=4,
                 label="FedAvg",
                 zorder=100,
                 ax=axes[0, 2])
    sns.lineplot(data=results[(results["approach"]=="Threshold") & (results["dataset"]=="FEMNIST")].sort_values("spars_label", ascending=True), 
                 x="round", 
                 y="accuracy", 
                 hue="spars_label", 
                 palette=['#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc'],
                 linewidth=4,
                 ax=axes[0, 2])
    axes[0, 2].set_title("Threshold Sparsification on FEMNIST", fontsize=16)

    sns.lineplot(x=range(1, len(fedavg_cifar)+1),
                 y=fedavg_cifar,
                 color="black",
                 linewidth=2,
                 label="FedAvg",
                 zorder=100,
                 ax=axes[1, 0])
    sns.lineplot(data=results[(results["approach"]=="Random") & (results["dataset"]=="CIFAR-10")].sort_values("spars_label", ascending=False), 
                 x="round", 
                 y="accuracy", 
                 hue="spars_label", 
                 palette=['#045a8d', '#2b8cbe', '#74a9cf', '#a6bddb', '#d0d1e6', '#d7d8e0'],
                 hue_order=["50%", "30%", "10%", "5%", "3%", "1%"],
                 linewidth=2,
                 ax=axes[1, 0])
    axes[1, 0].set_title("Random Sparsification on CIFAR-10", fontsize=16)

    sns.lineplot(x=range(1, len(fedavg_cifar)+1),
                 y=fedavg_cifar,
                 color="black",
                 linewidth=2,
                 label="FedAvg",
                 zorder=100,
                 ax=axes[1, 1])
    sns.lineplot(data=results[(results["approach"]=="Top-k") & (results["dataset"]=="CIFAR-10")].sort_values("spars_label", ascending=False), 
                 x="round", 
                 y="accuracy", 
                 hue="spars_label", 
                 palette=['#b30000', '#e34a33', '#fc8d59', '#fdbb84', '#fdd49e', '#ebdcc3'],
                 hue_order=["50%", "30%", "10%", "5%", "3%", "1%"],
                 linewidth=2,
                 ax=axes[1, 1])
    axes[1, 1].set_title("Top-k Sparsification on CIFAR-10", fontsize=16)

    sns.lineplot(x=range(1, len(fedavg_cifar)+1),
                 y=fedavg_cifar,
                 color="black",
                 linewidth=2,
                 label="FedAvg",
                 zorder=100,
                 ax=axes[1, 2])
    sns.lineplot(data=results[(results["approach"]=="Threshold") & (results["dataset"]=="CIFAR-10")].sort_values("spars_label", ascending=True), 
                 x="round", 
                 y="accuracy", 
                 hue="spars_label", 
                 palette=['#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc'],
                 linewidth=2,
                 ax=axes[1, 2])
    axes[1, 2].set_title("Threshold Sparsification on CIFAR-10", fontsize=16)

    for ax in axes.reshape(-1):
        ax.grid(alpha=1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(facecolor="white", framealpha=0.8, loc="upper left").set_zorder(300)

    axes[1, 1].legend(facecolor="white", framealpha=0.8, loc="lower right").set_zorder(300)
    axes[1, 2].legend(facecolor="white", framealpha=0.8, loc="lower right").set_zorder(300)

    for i in range(3):
        axes[0, i].set_ylim(0, 85)
        axes[0, i].set_xlabel("")
        axes[1, i].set_ylim(5, 65)
        axes[1, i].set_xlabel("Federated Learning Round")
    for i in range(2):
        axes[i, 1].set_ylabel("")
        axes[i, 2].set_ylabel("")
        axes[i, 0].set_ylabel("Accuracy")

    plt.savefig(f"figures/accuracy_round_plots.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_individual_accuracy_rounds(keep_first_last=False):
    
    results = pd.read_csv("results/results.csv")
    results["accs"] = results["accs"].apply(lambda row: ast.literal_eval(row))
    results = results.explode("accs").reset_index(drop=True)
    results["round"] = results["accs"].apply(lambda row: row[0])
    results["accuracy"] = results["accs"].apply(lambda row: row[1])

    for dataset in ["FEMNIST", "CIFAR-10"]:

        if dataset=="CIFAR-10":
            baseline_acc = 60
            ylim = [5, 65]
            linewidth = 2
        else: # if dataset=="FEMNIST"
            baseline_acc = 80
            ylim = [0, 85]
            linewidth = 4

        for approach in ["Top-k", "Random", "Threshold"]:

            if approach=="Top-k":
                palette=['#000000', '#b30000', '#e34a33', '#fc8d59', '#fdbb84', '#fdd49e', '#ebdcc3']
                hue_order = ["100%", "50%", "30%", "10%", "5%", "3%", "1%"]
                legend_title = "Sparsification"
                ascending=False

            elif approach=="Random":
                palette=['#000000', '#045a8d', '#2b8cbe', '#74a9cf', '#a6bddb', '#d0d1e6', '#d7d8e0']
                hue_order=["100%", "50%", "30%", "10%", "5%", "3%", "1%"]
                legend_title = "Sparsification"
                ascending=False

            else: # if approach=="Threshold"
                palette=['#000000', '#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc']
                hue_order=None
                legend_title = "Threshold"
                ascending=True
                
            for keep in [True, False]:

                data = results[(results["approach"]==approach) & (results["dataset"]==dataset) & (results["keep_first_last"]==keep)]
                data = data.sort_values("spars_label", ascending=ascending)

                sns.set_theme(style='white', font_scale=1.25)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.lineplot(data=data, 
                             x="round", 
                             y="accuracy", 
                             hue="spars_label", 
                             palette=palette,
                             hue_order=hue_order,
                             linewidth=linewidth,
                             markers=True,
                             ax=ax)
                ax.grid(alpha=0.4)
                ax.set_ylim(ylim)
                plt.axhline(y=baseline_acc, linewidth=2, color='green')
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.legend(title=legend_title, facecolor="white", framealpha=1.0, loc="upper left")

                if keep_first_last:
                    ax.set_title(f"'{approach}' Sparsification on {dataset}\nwith keeping the first and last layers = {keep}", fontsize=16);
                    plt.savefig(f"figures/{dataset}_{approach}_keepfirstlast_{keep}.png", 
                                dpi=300,
                                bbox_inches="tight",
                                facecolor="white")
                else:
                    if keep == False:
                        ax.set_title(f"'{approach}' Sparsification on {dataset}", fontsize=16);
                        plt.savefig(f"figures/{dataset}_{approach}.png", 
                                    dpi=300,
                                    bbox_inches="tight",
                                    facecolor="white")
                        
def plot_parameter_difference():
    trainloaders, testloaders = data.femnist_data(path_to_data_folder="E:/Folders/femnist_data")
    model = models.create_model("femnist", "CNN500k")
    
    original_params = np.concatenate([layer.cpu().numpy().ravel() for layer in model.state_dict().values()])
    utils.train(model, trainloaders[0], lr=0.1, epochs=1)
    update_params = np.concatenate([layer.cpu().numpy().ravel() for layer in model.state_dict().values()])
    delta_params = np.subtract(update_params, original_params)

    sns.set_theme(style='white', font_scale=1.25)
    sns.set_style('ticks')
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))

    axes[0].hist(delta_params,
                     bins=101,
                     edgecolor="white", 
                     align="mid",
                     color="#2b8cbe")
    axes[0].set_xlabel("Size of Difference")
    axes[0].set_ylabel("Number of Parameters")
    axes[0].set_xlim([-0.05, 0.05])

    axes[1].hist(np.abs(delta_params),
                     bins=101,
                     edgecolor="white", 
                     align="mid",
                     color="#2b8cbe")
    axes[1].set_ylabel("Number of Parameters (Log Scale)")
    axes[1].set_xlabel("Size of Absolute Difference (Log Scale)")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_xlim([axes[1].get_xlim()[0], 0.15])

    fig.suptitle("The Size of the Difference to each of the Model Parameters Before and After Model Training", fontsize=16, y=0.935);

    plt.savefig(f"figures/size_difference_parameters.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")

def plot_threshold_sizes():
    threshold_sizes = pd.read_csv("results/threshold_sizes.csv")
    threshold_sizes["threshold"] = threshold_sizes["threshold"].astype("category")
    femnist_thresh = threshold_sizes[(threshold_sizes["dataset"]=="FEMNIST") & (threshold_sizes["keep_first_last"]==False)]
    cifar_thresh = threshold_sizes[(threshold_sizes["dataset"]=="CIFAR-10") & (threshold_sizes["keep_first_last"]==False)]
    palette=['#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc', '#006837', ]

    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, 
                             nrows=1, 
                             figsize=(16, 6),
                             gridspec_kw={"wspace": 0.1})

    sns.boxplot(data=femnist_thresh, 
                 y="Size (bytes)",
                 hue="threshold", 
                 x="threshold",
                 palette=palette,
                 legend=False,
                 linewidth=2.25,
                 flierprops={"marker": "x", "alpha":0.5},
                 showmeans=True,
                 meanprops={'marker': 'D', 'markeredgecolor': "black",
                        'markerfacecolor': "None", 'markersize': 7},
                 ax=axes[0]);

    sns.boxplot(data=cifar_thresh, 
                 y="Size (bytes)",
                 hue="threshold", 
                 x="threshold",
                 palette=palette,
                 legend=False,
                 linewidth=2.25,
                 flierprops={"marker": "x", "alpha":0.5},
                 showmeans=True,
                 meanprops={'marker': 'D', 'markeredgecolor': "black",
                        'markerfacecolor': "None", 'markersize': 7},
                 ax=axes[1]);

    axes[0].grid(alpha=0.4, axis="y")
    axes[1].grid(alpha=0.4, axis="y")
    axes[0].set_xlabel("Threshold")
    axes[1].set_xlabel("Threshold")

    axes[0].set_ylabel("Client Upload Size (bytes)")
    axes[1].set_ylabel("")

    axes[0].set_title("FEMNIST Dataset", fontsize=16)
    axes[1].set_title("CIFAR-10 Dataset", fontsize=16)
    fig.suptitle("Distribution of Client Upload Sizes at each Threshold Level")

    plt.savefig(f"figures/update_sizes_threshold.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")

def plot_individual_threshold_sizes(keep_first_last=False):

    threshold_sizes = pd.read_csv("results/threshold_sizes.csv")
    threshold_sizes["threshold"] = threshold_sizes["threshold"].astype("category")
    palette=['#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc']

    for dataset in ["FEMNIST", "CIFAR-10"]:

        for keep in [True, False]:

            data = threshold_sizes[(threshold_sizes["dataset"]==dataset) & (threshold_sizes["keep_first_last"]==keep)]
            sns.set_theme(style='white', font_scale=1.25)
            fig, axes = plt.subplots(ncols=2, 
                                     nrows=1, 
                                     figsize=(16, 6))
            for i in range(2):
                sns.boxplot(data=data, 
                             y="Size (bytes)",
                             hue="threshold", 
                             x="threshold",
                             palette=palette,
                             legend=False,
                             linewidth=2.25,
                             flierprops={"marker": "x", "alpha":0.5},
                             ax=axes[i])
                axes[i].grid(alpha=0.4)
                axes[i].set_xlabel("Threshold")
            axes[1].set_ylabel("Size (bytes - log scale)")
            axes[1].set(yscale='log');

            if keep_first_last:
                fig.suptitle(f"Size of update for {dataset} when keeping the first and last layers = {keep}", y=0.935)
                plt.savefig(f"figures/update_sizes_threshold_{dataset}_keep_{keep}.png", 
                            dpi=300,
                            bbox_inches="tight",
                            facecolor="white")
            else: 
                if keep==False:
                    fig.suptitle(f"Size of updates for {dataset}", y=0.935)
                    plt.savefig(f"figures/update_sizes_threshold_{dataset}.png", 
                                dpi=300,
                                bbox_inches="tight",
                                facecolor="white")

def plot_threshold_over_time():
    
    threshold_sizes = pd.read_csv("results/threshold_sizes.csv")
    threshold_sizes["threshold"] = threshold_sizes["threshold"].astype("category")
    femnist_thresh = threshold_sizes[(threshold_sizes["dataset"]=="FEMNIST") & (threshold_sizes["keep_first_last"]==False)]
    cifar_thresh = threshold_sizes[(threshold_sizes["dataset"]=="CIFAR-10") & (threshold_sizes["keep_first_last"]==False)]

    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, 
                             nrows=2, 
                             figsize=(16, 6),
                             gridspec_kw={"wspace": 0.13, "hspace":0.05})

    sns.lineplot(data=femnist_thresh[femnist_thresh["threshold"]==0.005],
                 x=range(len(femnist_thresh[femnist_thresh["threshold"]==0.005])),
                 y="Size (bytes)", 
                 color="#006837",
                 lw=1,
                 alpha=1,
                 zorder=100,
                 ax=axes[0, 0])

    sns.lineplot(data=cifar_thresh[cifar_thresh["threshold"]==0.005],
                 x=range(len(cifar_thresh[cifar_thresh["threshold"]==0.005])),
                 y="Size (bytes)", 
                 color="#006837",
                 lw=1,
                 alpha=1,
                 zorder=100,
                 ax=axes[1, 0])

    sns.lineplot(data=femnist_thresh[femnist_thresh["threshold"]==0.007],
                 x=range(len(femnist_thresh[femnist_thresh["threshold"]==0.007])),
                 y="Size (bytes)", 
                 color='#31a354',
                 lw=1,
                 alpha=1,
                 zorder=100,
                 ax=axes[0, 1])

    sns.lineplot(data=cifar_thresh[cifar_thresh["threshold"]==0.007],
                 x=range(len(cifar_thresh[cifar_thresh["threshold"]==0.007])),
                 y="Size (bytes)", 
                 color='#31a354',
                 lw=1,
                 alpha=1,
                 zorder=100,
                 ax=axes[1, 1])

    axes[0, 0].set_title("Upload Size for all Clients over all Rounds for\nThreshold 0.005")
    axes[0, 1].set_title("Upload Size for all Clients over all Rounds for\nThreshold 0.007")
    axes[0, 0].set_ylabel("Upload size\nfor FEMNIST")
    axes[1, 0].set_ylabel("Upload size\nfor CIFAR-10")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")

    for ax in axes.reshape(-1): 
        ax.tick_params(axis="both", pad=-3)
        ax.grid(alpha=0.5, zorder=-300)
        ax.set_xticks([])

    plt.savefig(f"figures/threshold_over_time.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")

    
def plot_threshold_vs_topk():
    
    results = pd.read_csv("results/threshold_vs_topk.csv")
    results["accs"] = results["accs"].apply(lambda row: ast.literal_eval(row))
    results = results.explode("accs").reset_index(drop=True)
    results["round"] = results["accs"].apply(lambda row: row[0])
    results["accuracy"] = results["accs"].apply(lambda row: row[1])
    sns.set_theme(style="white", font_scale=1.25)
    fig, axes = plt.subplots(ncols=5, nrows=2, 
                             figsize=(24, 8), 
                             gridspec_kw={"wspace": 0.2,
                                          "hspace": 0.3})
    titles=["30% Top-k vs ~29.4% Threshold",
            "5% Top-k vs ~5.6% Threshold",
            "1% Top-k vs ~0.8% Threshold",
            "0.5% Top-k vs ~0.5% Threshold",
            "0.4% Top-k vs ~0.4% Threshold",
            "25% Top-k vs ~24.5% Threshold",
            "3% Top-k vs ~3.6% Threshold",
            "1% Top-k vs ~1.2% Threshold",
            "0.4% Top-k vs ~0.4% Threshold",
            "0.2% Top-k vs ~0.2% Threshold"]
    for plot, ax in enumerate(axes.reshape(-1), start=1):
        sns.lineplot(data=results[results["plot"]==plot],
                     x="round",
                     y="accuracy",
                     hue="approach",
                     palette=["#e34a33", "#31a354"],
                     linewidth=1.5,
                     ax=ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(alpha=0.4, zorder=-300)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.get_legend().remove()
        ax.set_title(titles[plot-1])
    axes[0, 0].legend(title="Approach", framealpha=1)

    axes[0, 0].set_ylabel("Accuracy on FEMNIST")
    axes[1, 0].set_ylabel("Accuracy on CIFAR-10")

    for i in range(5):
        axes[1, i].set_xlabel("Federated Learning Round")

    plt.savefig(f"figures/topk_vs_threshold.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title