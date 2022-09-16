import json
import glob
import argparse

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description="plots simulation result for scaling")
    parser.add_argument("exp", type=float, help="exponent")
    parser.add_argument("figure", type=str, help="figure file")
    args = parser.parse_args()

    cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    markers = ["o", "v", "s", "^", "p", "<", "d", ">"]

    fig, ((ax_unscaled, ax_scaled),
            (ax_empirical_unscaled, ax_empirical_scaled)) = plt.subplots(
                    nrows=2, ncols=2, figsize=(8, 8),
                    gridspec_kw={
                        "wspace": 0.20, "hspace": 0.20,
                        "bottom": 0.10, "top": 0.95,
                        "left": 0.10, "right": 0.99})
    sizes = [2048, 4096, 8192, 16384, 32768]
    average_vote_scores = []
    for size, col, m in zip(sizes, cols, markers):
        comparison_f1 = {}
        votes_f1 = []
        filenames = glob.glob(f"output/simulations/scaling/{size}/*.json")
        for filename in filenames:
            with open(filename) as f:
                try:
                    j = json.load(f)
                    comps = j["comparisons"]["number_of_comparisons"]
                    scores = j["comparisons"]["scores"]

                    for comp, f1 in zip(comps, scores):
                        if comp not in comparison_f1:
                            comparison_f1[comp] = []
                        comparison_f1[comp].append(f1)

                    votes_f1.append(j["majority_vote_scores"])
                except json.decoder.JSONDecodeError:
                    continue

        comps = np.array(sorted(comparison_f1))
        means = []
        sems = []
        votes = np.mean(votes_f1)
        for comp in comps:
            sems.append(stats.sem(comparison_f1[comp]))
            means.append(np.mean(comparison_f1[comp]))
        print(size, comps)
        yerr = 5*np.array(sems)
        ax_scaled.plot(comps/((size**args.exp)*np.log(size)), means,
                label=f"$N = {size}$", color=col,
                marker=m, ms=3.0)

        ax_scaled.fill_between(
                comps/((size**args.exp)*np.log(size)),
                means - yerr,
                means + yerr,
                alpha=0.4)

        ax_unscaled.plot(comps, means,
                label=f"$N = {size}$", color=col,
                marker=m, ms=3.0)

        ax_unscaled.fill_between(
                comps,
                means - yerr,
                means + yerr,
                alpha=0.4)

        average_vote_scores.append(votes)

    sizes = [25, 30, 35, 40]
    for size, col, m in zip(sizes, cols, markers):
        comparison_f1 = {}
        filenames = glob.glob(f"output/empirical/scaling/{size}/*.json")
        for filename in filenames:
            with open(filename) as f:
                try:
                    j = json.load(f)
                    comps = j["comparisons"]["number_of_comparisons"]
                    scores = j["comparisons"]["scores"]

                    for comp, f1 in zip(comps, scores):
                        if comp not in comparison_f1:
                            comparison_f1[comp] = []
                        comparison_f1[comp].append(f1)
                except json.decoder.JSONDecodeError:
                    continue

        comps = np.array(sorted(comparison_f1))
        means = []
        sems = []
        votes = np.mean(votes_f1)
        for comp in comps:
            sems.append(stats.sem(comparison_f1[comp]))
            means.append(np.mean(comparison_f1[comp]))
        print(size, comps)
        yerr = 5*np.array(sems)
        ax_empirical_scaled.plot(
                comps/((size**args.exp)*np.log(size)), means,
                label=f"$N = {size}$", color=col,
                marker=m, ms=3.0, markevery=5)

        ax_empirical_scaled.fill_between(
                comps/((size**args.exp)*np.log(size)),
                means - yerr,
                means + yerr,
                alpha=0.4)

        ax_empirical_unscaled.plot(comps, means,
                label=f"$N = {size}$", color=col,
                marker=m, ms=3.0, markevery=5)

        ax_empirical_unscaled.fill_between(
                comps,
                means - yerr,
                means + yerr,
                alpha=0.4)


        average_vote_scores.append(votes)

    ax_unscaled.axhline(np.mean(average_vote_scores), color="grey", ls='--')
    ax_scaled.axhline(np.mean(average_vote_scores), color="grey", ls='--')

    ax_scaled.text(-0.2, 1.01, "(b)", transform=ax_scaled.transAxes)
    # ax_scaled.set_xlabel(f"$n_{{comparisons}}/(N \\log N)$")
    # ax_scaled.set_ylabel("Positive label $f_1$")
    ax_scaled.set_ylim(0.65, 1.00)
    ax_scaled.legend()

    ax_unscaled.text(-0.2, 1.01, "(a)", transform=ax_unscaled.transAxes)
    # ax_unscaled.set_xlabel(f"$n_{{comparisons}}$")
    ax_unscaled.set_ylabel("Positive label $f_1$")
    ax_unscaled.ticklabel_format(axis="x", style="sci", scilimits=(4,4),
            useMathText=True)
    ax_unscaled.set_ylim(0.65, 1.00)


    ax_empirical_scaled.text(-0.2, 1.01, "(d)",
            transform=ax_empirical_scaled.transAxes)
    ax_empirical_scaled.set_xlabel(f"$n_{{comparisons}}/(N \\log N)$")
    # ax_empirical_scaled.set_ylabel("Positive label $f_1$")
    ax_empirical_scaled.set_ylim(0.60, 0.95)
    ax_empirical_scaled.legend()

    ax_empirical_unscaled.text(-0.2, 1.01, "(c)",
            transform=ax_empirical_unscaled.transAxes)
    ax_empirical_unscaled.set_xlabel(f"$n_{{comparisons}}$")
    ax_empirical_unscaled.set_ylabel("Positive label $f_1$")
    ax_empirical_unscaled.ticklabel_format(
            axis="x", style="sci", scilimits=(4,4), useMathText=True)
    ax_empirical_unscaled.set_ylim(0.60, 0.95)

    fig.savefig(args.figure)
