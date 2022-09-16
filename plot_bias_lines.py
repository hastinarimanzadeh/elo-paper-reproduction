import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the scores as line plots")
    parser.add_argument("--figure-text", type=str, help="subplot title")
    parser.add_argument("output", type=str, help="figure file")
    parser.add_argument("simulations", type=str, nargs='+', help="json files")
    args = parser.parse_args()

    params_f1 = {}
    for datafile in args.simulations:
        with open(datafile) as f:
            try:
                all_data = json.load(f)
                if "beta" in all_data["params"]:
                    val = all_data["params"]["beta"]
                    if val not in params_f1:
                        params_f1[val] = {"f1_comparison": [], "f1_majority_vote": []}
                    params_f1[val]["f1_comparison"].append(all_data["comparisons"]["biases"])
                    params_f1[val]["f1_majority_vote"].append(all_data["majority_vote_biases"])
                    params_f1[val]["items"] = all_data["params"]["items"]
                    params_f1[val]["votes"] = all_data["params"]["individual_votes"]
                    params_f1[val]["raters"] = all_data["params"]["raters"]
                    params_f1[val]["comparisons"] = all_data["comparisons"]["number_of_comparisons"]
            except json.decoder.JSONDecodeError:
                pass

    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    markers = ["o", "v", "s", "^", "p", "<", "d", ">"]

    fig, ax = plt.subplots(figsize=(4, 3))

    vals = sorted(params_f1)
    for val, col, marker in zip(vals, cols, markers):
        biases = np.array(params_f1[val]["f1_comparison"])
        majority_biases = np.array(params_f1[val]["f1_majority_vote"])

        total_majority_votes = params_f1[val]["items"]*params_f1[val]["votes"]
        rel_comps = np.array(params_f1[val]["comparisons"])/total_majority_votes
        mean_biases = biases.mean(axis=0)
        mean_majority_bias = majority_biases.mean()

        yerr = 5*sem(biases, axis=0)
        ax.axhline(np.absolute(mean_majority_bias), ls='--', color="tab:orange",
                label="Majority-vote")
        ax.plot(rel_comps,
                np.absolute(mean_biases),
                marker=marker,
                linewidth=0.3,
                color=col,
                ms=1.5,
                markevery=10,
                label="Comparitive")
        ax.set_xlabel("Relative number of comparison tasks")
        ax.set_ylabel("|Bias|")
        if args.figure_text:
            ax.text(-0.2, 1.0, args.figure_text, transform=ax.transAxes)
        ax.fill_between(
                rel_comps,
                np.absolute(mean_biases) - yerr,
                np.absolute(mean_biases) + yerr,
                alpha=0.4)

        ax.axvline(x=1.0, ls='--', color='tab:pink')

    legend_title = "Method"
    ax.legend(
            title=legend_title,
            markerscale=2,
            handlelength=1.5)
    fig.tight_layout()
    fig.savefig(args.output)
