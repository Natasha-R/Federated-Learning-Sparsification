import data
import models
import utils
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mtick
import ast
import oapackage

def plot_class_distributions(loaders, dataset_name):
    
    classes_counts = pd.DataFrame([pd.Series(np.concatenate([batch[1].numpy() for batch in loader])).value_counts() for loader in loaders]).fillna(0).reset_index(drop=True)
    classes_counts = classes_counts.reindex(sorted(classes_counts.columns), axis=1)
    classes_counts["client"] = [f"Client {value}" for value in reversed(range(1, 11))]

    sns.set_theme(style="white", font_scale=1.25)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    
    if dataset_name=="CIFAR-10":
        classes_counts.plot(x="client", y=list(range(10)), kind="barh", stacked=True, width=0.8, ax=ax, colormap="tab10")
        ax.legend(title="Class", loc=(1.02, 0.1), handlelength=1.5, handletextpad=0.3)
    else:
        sns.set_palette("tab20", plt.cm.tab20.N)
        classes_counts.plot(x="client", y=list(range(62)), kind="barh", stacked=True, width=0.8, ax=ax)
        ax.get_legend().remove()
        
    ax.set_xlabel("Number of datapoints") 
    ax.set_ylabel("")
    ax.set_title(f"Distribution of {dataset_name} classes on 10 example clients", fontsize=16)
    ax.grid(alpha=0.4, axis="x")

    plt.savefig(f"figures/{dataset_name}_class_distribution.png", 
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
    ax.set(yscale="log");
    ax.set_xlabel("")
    ax.set_ylabel("Download size (bytes - log scale)");
    ax.text(x=0.75, y=0.8, s="* Public\ndataset\nsize", alpha=0.6, horizontalalignment="right", fontsize=17, transform=ax.transAxes);
    ax.set_title("Size of download for each client (CIFAR-10)", fontsize=17);
    ax.set_xticklabels(["FedAvg\n(Baseline)", "Sparsification", "One-shot\n(MA-Echo)", "Distillation"])
    plt.savefig("figures/download_size_cifar.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_sparsification_pareto():
    all_results = pd.read_csv("results/results.csv")
    all_results = all_results[all_results["keep_first_last"]==False].reset_index(drop=True)
    all_results["max_accuracy"] = all_results["accs"].apply(lambda row: max([value[1] for value in ast.literal_eval(row)]))

    palette = ['#e34a33', '#2b8cbe', '#31a354']
    palette_dict = {"Top-k":'#b30000', "Random":'#045a8d', "Threshold":'#006837'}

    for dataset in ["FEMNIST", "CIFAR-10"]:

        data = all_results[all_results["dataset"]==dataset].reset_index(drop=True)
        sns.set_theme(style='white', font_scale=1.25)
        fig, axes = plt.subplots(ncols=2, 
                                nrows=1, 
                                figsize=(16, 6))
        fig.suptitle(f"Maximum Accuracy vs Size (in bytes) on {dataset}", fontsize=16, y=0.935);

        pareto=oapackage.ParetoDoubleLong()
        for index in range(0, data.shape[0]):
            solution=oapackage.doubleVector((-data.loc[index, "bytes_size"], data.loc[index, "max_accuracy"]))
            pareto.addvalue(solution, index)
        optimal_solutions=data.loc[pareto.allindices(),:]

        for i in range(2):
            sns.lineplot(data=optimal_solutions,
                         x="bytes_size", 
                         y="max_accuracy",
                         color="black",
                         lw=2.5,
                         alpha=0.8,
                         zorder=-100,
                         ax=axes[i])

            sns.scatterplot(data=data, 
                         x="bytes_size", 
                         y="max_accuracy", 
                         hue="approach", 
                         hue_order=["Top-k", "Random", "Threshold"],
                         palette=palette,
                         s=200,
                         alpha=0.7,
                         ax=axes[i])
            axes[i].grid(alpha=0.4)
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            axes[i].set_ylabel("Accuracy")
            axes[i].set_xlabel("Size (bytes)")
            axes[i].legend(facecolor = 'white', framealpha=1.0, loc="lower right");

        axes[1].set_xlabel("Size (bytes - log scale)")
        axes[1].set(xscale='log');
        for _, row in optimal_solutions.iterrows():
            colour = palette_dict[row["approach"]]
            axes[1].text(x=row["bytes_size"]+2000, 
                         y=row["max_accuracy"]-2, 
                         s=row["spars_label"], 
                         c=colour, 
                         weight='bold', 
                         horizontalalignment="left", 
                         size=13, 
                         color='black', 
                         zorder=100)

        plt.savefig(f"figures/pareto_front_{dataset}.png", 
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white")
    
def plot_combined_pareto():
    combined_results = pd.read_csv("results/combined_results.csv")
    combined_results["Total Bytes Uploaded"] = combined_results["Total Bytes Uploaded"].astype(np.float64)

    class LegendTitle(object):
        def __init__(self, text_props=None):
            self.text_props = text_props or {}
            super(LegendTitle, self).__init__()

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
            handlebox.add_artist(title)
            return title

    sns.set_theme(style='white', font_scale=1.25)

    fig, axes = plt.subplots(ncols=2, 
                            nrows=1, 
                            figsize=(16, 6))

    for i, dataset in enumerate(["CIFAR-10", "FEMNIST"]):

        data = combined_results[combined_results["Dataset"]==dataset].reset_index(drop=True)

        sns.scatterplot(data=data[data["Type"]=="Baseline"], 
             x="Total Bytes Uploaded", 
             y="Accuracy", 
             hue="Method", 
             marker="X",
             color="black",
             palette=["#000000"],
             s=200,
             alpha=1,
             ax=axes[i])

        sns.scatterplot(data=data[data["Type"]=="Distillation"], 
             x="Total Bytes Uploaded", 
             y="Accuracy", 
             hue="Method", 
             palette=["#c51b8a", "#fa9fb5"],
             s=200,
             alpha=1,
             ax=axes[i])

        sns.scatterplot(data=data[data["Type"]=="One-shot"], 
             x="Total Bytes Uploaded", 
             y="Accuracy", 
             hue="Method", 
             palette=["#7a0177", "#bcbddc", "#3182bd"],
             marker="^",
             s=200,
             alpha=1,
             ax=axes[i])

        sns.scatterplot(data=data[data["Type"]=="Sparsification"], 
             x="Total Bytes Uploaded", 
             y="Accuracy", 
             hue="Method", 
             marker="s",
             palette=["#e34a33", "#fdbb84", "#31a354", "#addd8e"],
             s=200,
             alpha=1,
             zorder=-10,
             ax=axes[i])

        axes[i].grid(alpha=0.4)
        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        axes[i].set_ylabel("Accuracy")
        axes[i].set_xlabel("Bytes uploaded by each client")
        axes[i].legend(facecolor = 'white', framealpha=1.0, loc="lower right");
        axes[i].set_title(f"Accuracy vs Uploaded Bytes on {dataset}", fontsize=16);


        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend([handles[0]] + ["Distillation"] + handles[1:3] + ["One-shot"] + handles[3:6] + ["Sparsification"] + handles[6:10],
                     [labels[0]] + [""] + ["Distillation (10 (CIFAR) \n or 4 (FEMNIST) rounds)", "Distillation (2 rounds)"] 
                              + [""] + labels[3:6]  + [""] + labels[6:8] + ["Threshold (~5%)", "Threshold (~0.3%)"],
                       handler_map={str: LegendTitle({'fontsize': 15})},
           facecolor="white",
           framealpha=0.5,
           borderpad=0.8,
           loc=(0.425, 0.03))

    axes[1].set_ylim([0, 86])
    axes[0].set_ylim([0, 65])
    axes[0].get_legend().remove()

    plt.savefig(f"figures/pareto_front_combined.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_results(keep_first_last=False):
    
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
    utils.train(model, trainloaders[0], lr=0.1, epochs=10)
    update_params = np.concatenate([layer.cpu().numpy().ravel() for layer in model.state_dict().values()])
    delta_params = np.subtract(update_params, original_params)

    sns.set_theme(style='white', font_scale=1.25)
    sns.set_style('ticks')
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6), sharex=True)
    for i in range(2):
        axes[i].hist(delta_params,
                     bins=31,
                     edgecolor="white", 
                     align="mid",
                     color="#4c68ae")
        axes[i].set_xlabel("Size of difference")

    axes[0].set_ylabel("Number of parameters")
    axes[1].set_ylabel("Number of parameters (log scale)")
    axes[1].set_yscale("log")
    fig.suptitle("The size of the difference to each of the model parameters before and after model training", fontsize=16, y=0.935);

    plt.savefig(f"figures/size_difference_parameters.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")
    
def plot_threshold_sizes(keep_first_last=False):

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