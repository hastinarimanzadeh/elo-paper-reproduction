import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the scores as line plots")
    parser.add_argument("output", type=str, help="figure file")
    parser.add_argument("simulations", type=str, nargs='+', help="json files")
    args = parser.parse_args()

    comps = []
    scores = []
    vote_f1s = {}
    vote_tasks = {}
    for datafile in args.simulations:
        with open(datafile) as f:
            try:
                all_data = json.load(f)
                scores.append(all_data["comparisons"]["scores"])
                comps = all_data["comparisons"]["number_of_comparisons"]

                votes = all_data["params"]["individual_votes"]
                if votes not in vote_f1s:
                    vote_f1s[votes] = []
                vote_f1s[votes].append(all_data["majority_vote_scores"])
                vote_tasks[votes] = votes*all_data["params"]["items"]
            except json.decoder.JSONDecodeError:
                pass

    fig, ax = plt.subplots(figsize=(4, 3))

    scores = np.array(scores)
    mean_scores = np.mean(scores, axis=0)
    ax.plot(comps,
            mean_scores,
            marker='o',
            linewidth=0.3,
            color="tab:blue",
            ms=1.5,
            markevery=10)

    yerr = 5*sem(scores, axis=0)
    ax.fill_between(
            comps,
            mean_scores - yerr,
            mean_scores + yerr,
            color="tab:blue",
            alpha=0.4)

    cols = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for v, col in zip(sorted(vote_f1s), cols):
        ax.scatter([vote_tasks[v]], [np.mean(vote_f1s[v])],
                marker="+", s=20, label=f"{v}", color=col)

    ax.legend(
            title="Majority-vote votes",
            ncol=2,
            markerscale=2,
            handlelength=1.5)

    ax.set_ylabel("Positive label $f_1$")
    ax.set_xlabel("Number of tasks")

    fig.tight_layout()
    fig.savefig(args.output)
