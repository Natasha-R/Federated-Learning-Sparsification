import pandas as pd
import ast
import numpy as np

def goal_accuracy_table():
    original_results = pd.read_csv("results/results.csv")
    extra_topk = pd.read_csv("results/threshold_vs_topk.csv").iloc[:,:-1]
    original_results = original_results[original_results["keep_first_last"]==False].reset_index(drop=True)
    original_results = pd.concat([original_results, extra_topk]).drop_duplicates().reset_index(drop=True)
    results = original_results.copy()
    results["accs"] = results["accs"].apply(lambda row: ast.literal_eval(row))
    results = results.explode("accs").reset_index(drop=True)
    results["round"] = results["accs"].apply(lambda row: row[0])
    results["accuracy"] = results["accs"].apply(lambda row: row[1])
    results = results[["dataset", 
                       "approach", 
                       "spars_label", 
                       "sparsify_by", 
                       "bytes_size", 
                       "accuracy", 
                       "round"]].reset_index(drop=True)

    size_results = {"dataset":[], "approach":[], "spars_label":[], "round":[], "accuracy_reached":[]}
    for goal in range(10, 90, 5):
        for group in results.groupby(["dataset", "approach", "spars_label"]):
            first_round_over_thresh = group[1][group[1]["accuracy"]>=goal]["round"].values
            if len(first_round_over_thresh) > 0:
                size_results["dataset"].append(group[0][0])
                size_results["approach"].append(group[0][1])
                size_results["spars_label"].append(group[0][2])
                size_results["round"].append(first_round_over_thresh[0])
                size_results["accuracy_reached"].append(goal)
    size_results = pd.DataFrame(size_results)       

    size_results = size_results.merge(original_results[["dataset", "approach", "spars_label", "bytes_size"]],                 
                      how="inner",
                      on=["dataset", "approach", "spars_label"])
    size_results["bytes_communicated"] = size_results["bytes_size"] * size_results["round"]

    top = size_results.sort_values("bytes_communicated").groupby(["dataset", "accuracy_reached"]).head(1).sort_values(["dataset", "accuracy_reached"]).reset_index(drop=True)
    
size_results.to_csv("results/goal_accuracies.csv", index=False)
top.to_csv("results/top_goal_accuracy.csv", index=False)