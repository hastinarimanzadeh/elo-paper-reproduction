import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the scores as line plots")
    parser.add_argument("parameter", type=str,
            default="perception_ambiguity", help="The parameter to plot")
    parser.add_argument("--figure-text", type=str, help="subplot title")
    parser.add_argument("output", type=str, help="figure file")
    parser.add_argument("simulations", type=str, nargs='+', help="json files")
    args = parser.parse_args()

    params_f1 = {}
    for datafile in args.simulations:
        with open(datafile) as f:
            try:
                all_data = json.load(f)
                if args.parameter in all_data["params"]:
                    val = all_data["params"][args.parameter]
                    if val not in params_f1:
                        params_f1[val] = {"f1_comparison": [], "f1_majority_vote": []}
                    params_f1[val]["f1_comparison"].append(all_data["comparisons"]["scores"])
                    params_f1[val]["f1_majority_vote"].append(all_data["majority_vote_scores"])
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
        scores = np.array(params_f1[val]["f1_comparison"])
        majority_scores = np.array(params_f1[val]["f1_majority_vote"])
        scores_diff = scores - majority_scores.reshape((-1, 1))

        total_majority_votes = params_f1[val]["items"]*params_f1[val]["votes"]
        rel_comps = np.array(params_f1[val]["comparisons"])/total_majority_votes
        mean_scores_diff = scores_diff.mean(axis=0)

        label = f"{val}"
        if args.parameter == "spammers":
            label = f"{val/params_f1[val]['raters']:.0%}"

        yerr = 5*sem(scores_diff, axis=0)
        ax.plot(rel_comps,
                mean_scores_diff,
                marker=marker,
                linewidth=0.3,
                color=col,
                ms=1.5,
                markevery=10,
                label=f"{val}")
        ax.set_xlabel("Relative number of comparison tasks")
        ax.set_ylabel("$f_1$ score difference")
        if args.figure_text:
            ax.text(-0.2, 1.0, args.figure_text, transform=ax.transAxes)
        ax.fill_between(
                rel_comps, 
                mean_scores_diff - yerr,
                mean_scores_diff + yerr,
                alpha=0.4)

        ax.axhline(y=0.0, ls='--', color='tab:gray')
        ax.axvline(x=1.0, ls='--', color='tab:pink')

    legend_title = None
    if args.parameter == "perception_ambiguity":
        legend_title = "Ambiguity in perception"
    elif args.parameter == "comparison_ambiguity":
        legend_title = "Ambiguity in comparison"
    elif args.parameter == "personal_threshold_variance":
        legend_title = "Personal threshold\nvariance ($\sigma$)"
    elif args.parameter == "spammers":
        legend_title = "Percentage of spam raters"
    elif args.parameter == "individual_votes":
        legend_title = "Majority-vote votes"

    ax.legend(
            title=legend_title,
            ncol=2,
            markerscale=2,
            handlelength=1.5)
    fig.tight_layout()
    fig.savefig(args.output)
