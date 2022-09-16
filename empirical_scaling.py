import argparse
import json
import sys

import numpy as np

from simulate_comps import score
from elo import Player, elo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= \
            "Measure scaling of accuracy based on number of comparisons in "\
            "real-world data")
    parser.add_argument("comparisons", type=str, help="comparisons file")
    parser.add_argument("items", type=int,
            help="number of items subsetted from the full set")
    parser.add_argument("--selection-seed", type=int, default=0,
            help="seed for the selecting items")
    parser.add_argument("--shuffle-seed", type=int, default=0,
            help="seed for the ordering of comparisons")
    parser.add_argument("--report-distance", type=int, default=1000,
            help="report accuracy and bias after every this many comparisons")
    parser.add_argument("--epochs", type=int, default=20,
            help="number of repetitions (epochs) of the same comparisons")
    parser.add_argument("--k-value", type=float, default=0.15,
            help="K parameter for the Elo method")
    args = parser.parse_args()

    shuffle_rng = np.random.default_rng(args.shuffle_seed)

    report = {"params": {
        "items": args.items, "comparisons": args.comparisons,
        "report_distance": args.report_distance,
        "shuffle_seed": args.shuffle_seed,
        "selection_seed": args.selection_seed,
        "epochs": args.epochs, "k_value": args.k_value}}

    all_items = set()
    all_comparisons = []
    with open(args.comparisons) as comps:
        for line in comps:
            i, j, s = line.split(",")
            all_comparisons.append({
                Player.player_a: i, Player.player_b: j,
                "SA": float(s), "SB": 1-float(s)})
            all_items.update([i , j])
    all_items = sorted(all_items)
    all_comparisons.sort(
            key=lambda c: (c[Player.player_a], c[Player.player_b], c["SA"]))

    no_repeat_comparisons = []
    pairs = set()
    shuffle_rng.shuffle(all_comparisons)
    for comp in all_comparisons:
        a = comp[Player.player_a]
        b = comp[Player.player_b]
        if (a, b) not in pairs:
            no_repeat_comparisons.append(comp)
            pairs.add((a, b))

    all_ratings = {i: 0 for i in all_items}
    for _ in range(args.epochs):
        shuffle_rng.shuffle(no_repeat_comparisons)
        for comp in no_repeat_comparisons:
            all_ratings = elo(comp, all_ratings, args.k_value)

    selection_rng = np.random.default_rng(args.selection_seed)
    items = selection_rng.choice(all_items, size=args.items, replace=False)

    items_set = set(items)
    median_rating = sorted([r for i, r in all_ratings.items()
        if i in items_set])[args.items//2]
    true_labels = {i: r > median_rating
            for i, r in all_ratings.items() if i in items_set}

    comparisons = [comp for comp in no_repeat_comparisons if
            comp[Player.player_a] in items_set and
            comp[Player.player_b] in items_set]

    comps = []
    scores = []
    for c in range(0, len(comparisons), args.report_distance):
        ratings = {item: 0 for item in items}
        round_comparisons = comparisons[0:c]

        for _ in range(args.epochs):
            shuffle_rng.shuffle(round_comparisons)
            for comp in round_comparisons:
                ratings = elo(comp, ratings, args.k_value)

        elo_labels = {item: rating > 0
                for item, rating in ratings.items()}

        comps.append(c)
        scores.append(score(true_labels, elo_labels))

    report["comparisons"] = {
            "number_of_comparisons": comps,
            "scores": scores}

    print(json.dumps(report))
