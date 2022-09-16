import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="plots heatmap of comparison vs parameter")
    parser.add_argument("parameter", type=str)
    parser.add_argument("output", type=str, help="figure file")
    parser.add_argument("results", type=str, nargs="+",
            help="path to result files")

    args = parser.parse_args()

    all_res = {}
    comps = []
    for filename in args.results:
        with open(filename) as f:
            try:
                result = json.load(f)
                p = result["params"][args.parameter]
                if p not in all_res:
                    all_res[p] = []
                score_diff = np.array(result["comparisons"]["scores"]) - \
                    result["majority_vote_scores"]
                all_res[p].append(score_diff)
                maj_vote = result["params"]["items"]*\
                        result["params"]["individual_votes"]
                comps = np.array(
                        result["comparisons"]["number_of_comparisons"])/maj_vote
            except json.decoder.JSONDecodeError:
                pass

    mat = np.array([np.mean(all_res[k], axis=0) for k in sorted(all_res)])
    fig, ax = plt.subplots(figsize=(4, 3.2),
                    gridspec_kw={
                        "bottom": 0.13, "top": 0.98,
                        "left": 0.15, "right": 0.92})
    c = ax.pcolormesh(comps, sorted(all_res), mat,
            shading="nearest", cmap='BrBG')
    ax.axvline(x=1, ls='--', color='tab:pink')

    ax.set_xlabel("Relative number of comparison tasks")
    if args.parameter == "personal_threshold_variance":
        ax.set_ylabel("Rater threshold variance")
    elif args.parameter == "perception_ambiguity":
        ax.set_ylabel("Perception ambiguity")
    elif args.parameter == "comparison_ambiguity":
        ax.set_ylabel("Comparison ambiguity")

    fig.colorbar(c, ax=ax)
    fig.savefig(args.output)
