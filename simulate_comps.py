import argparse
import json
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from elo import Player, elo

class SimpleItem:
    def __init__(self, rating, bad_feature, item_id):
        self.item_id = item_id
        self.rating = rating
        self.bad_feature  = bad_feature

class SpamRater:
    def __init__(self):
        pass

    def rate(self, item, rng):
        return bool(rng.random() < 0.5)

    def compare(self, item1, item2, rng):
        return rng.choice([-1, 0, 1])

class SimpleRater:
    def __init__(
            self, bias,
            percption_ambiguity,
            comparison_ambiguity,
            bad_feature_importance,
            rater_id):
        self.bias = bias
        self.perc_ambiguity = percption_ambiguity
        self.comp_ambiguity = comparison_ambiguity
        self.bad_feature_imp = bad_feature_importance
        self.rater_id = rater_id

    def perceive(self, item):
        rng = np.random.default_rng((self.rater_id, item.item_id))
        theta = rng.normal(item.rating, self.perc_ambiguity)
        return np.sqrt(1 - self.bad_feature_imp**2) * theta + \
                self.bad_feature_imp*item.bad_feature*1.0

    def rate(self, item, rng):
        return bool(self.perceive(item) > self.bias)

    def compare(self, item1, item2, rng):
        p1 = self.perceive(item1)
        p2 = self.perceive(item2)

        if abs(p1 - p2) < self.comp_ambiguity:
            return 0
        else:
            return 1 if p1 > p2 else -1

def vote_items(items: list, raters: list, rng, n=3):
    raters_votes = {item.item_id: [] for item in items}
    for item in items:
        for _ in range(n):
            rater = rng.choice(raters, replace=False)
            raters_votes[item.item_id].append(rater.rate(item, rng))
    return raters_votes


def nth_combination(iterable, r, index):
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError

    c = 1
    k = min(r, n-r)
    for i in range(1, k+1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError

    result = []
    while r:
        c, n, r = c*r//n, n-1, r-1
        while index >= c:
            index -= c
            c, n = c*(n-r)//n, n-1
        result.append(pool[-1-n])

    return tuple(result)

def get_labels(labels_dict):
    return list(dict(sorted(
        labels_dict.items(), key=lambda i: i[0])).values())

def score(true_labels, estimated_labels):
    return precision_recall_fscore_support(
            get_labels(true_labels),
            get_labels(estimated_labels))[2][0]

def bias(true_labels, estimated_labels, bad_features):
    true_labels = {i: true_labels[i]
            for i in bad_features if bad_features[i]}
    estimated_labels = {i: estimated_labels[i]
            for i in bad_features if bad_features[i]}

    mean_true = np.mean(get_labels(true_labels))
    mean_estimated = np.mean(get_labels(estimated_labels))
    return mean_estimated - mean_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= \
            "Simulate raters and items to be rated")
    parser.add_argument("items", type=int, help="number of items")
    parser.add_argument("raters", type=int, help="number of raters")
    parser.add_argument("--individual-votes", type=int, default=3,
            help="number of votes for each item in the majority-vote")
    parser.add_argument("--spammers", type=int, default=0,
            help="number of spammers out of number of raters")
    parser.add_argument("--personal-threshold-variance", type=float,
            default=0.5, help="variance of rater personal thresholds")
    parser.add_argument("--perception-ambiguity", type=float, default=0.5,
            help="ambiguity in perception of the true rating of items")
    parser.add_argument("--comparison-ambiguity", type=float, default=0.5,
            help="ambiguity of comparison between two items")
    parser.add_argument("--alpha", type=float, default=1e50,
            help="alpha of beta distribution for the bad feature importance")
    parser.add_argument("--beta", type=float, default=1e50,
            help="beta of beta distribution for the bad feature importance")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--max-comparisons", type=int, default=10000,
            help="maximum number of combinations to compare")
    parser.add_argument("--report-distance", type=int, default=1000,
            help="report accuracy and bias after every this many comparisons")
    parser.add_argument("--epochs", type=int, default=20,
            help="number of repetitions (epochs) of the same comparisons")
    parser.add_argument("--k-value", type=float, default=0.15,
            help="K parameter for the Elo method")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    mu = 0
    sd = 1.0

    items = []
    for it in range(args.items):
        items.append(SimpleItem(
            rng.normal(mu, sd),
            bool(rng.choice([True, False])), it))

    if args.spammers > args.raters or args.spammers < 0:
        raise RuntimeError("Spammers should be in the range [0, raters]")

    real_raters = args.raters - args.spammers
    raters = []
    for r in range(real_raters):
        b = rng.normal(mu, args.personal_threshold_variance)
        p = rng.beta(args.alpha, args.beta)*2 - 1
        raters.append(SimpleRater(
            b, args.perception_ambiguity, args.comparison_ambiguity, p, r))
    raters += [SpamRater() for i in range(args.spammers)]

    report = {"params": {"items": args.items, "raters": args.raters,
        "individual_votes": args.individual_votes, "spammers": args.spammers,
        "personal_threshold_variance": args.personal_threshold_variance,
        "perception_ambiguity": args.perception_ambiguity,
        "comparison_ambiguity": args.comparison_ambiguity,
        "alpha": args.alpha, "beta": args.beta, "seed": args.seed,
        "max_comparisons": args.max_comparisons,
        "report_distance": args.report_distance,
        "epochs": args.epochs, "k_value": args.k_value}}

    true_labels = {item.item_id: item.rating > 0 for item in items}
    bad_features = {item.item_id: item.bad_feature for item in items}

    votes = vote_items(items, raters, rng, args.individual_votes)
    majority_vote_labels = {item_id: sum(vote) > args.individual_votes//2
            for item_id, vote in votes.items()}

    report["majority_vote_scores"] = score(
            true_labels, majority_vote_labels)
    report["majority_vote_biases"] = bias(
            true_labels, majority_vote_labels, bad_features)

    comps = []
    biases = []
    scores = []

    comp_indeces = rng.choice((args.items*(args.items-1))//2,
            args.max_comparisons, replace=False)
    all_comps = []
    rng.shuffle(comp_indeces)

    all_comps = []
    for c in range(0, args.max_comparisons + 1, args.report_distance):
        ratings = {item.item_id: 0 for item in items}

        for i in range(len(all_comps), c):
            item1, item2 = nth_combination(items, 2, comp_indeces[i])

            worker = rng.choice(raters)
            comp = worker.compare(item1, item2, rng)
            all_comps.append({
                        Player.player_a: item1.item_id,
                        Player.player_b: item2.item_id,
                        "SA": (comp+1)/2, "SB": 1 - (comp+1)/2})

        for rep in range(args.epochs):
            rng.shuffle(all_comps)
            for comp in all_comps:
                ratings = elo(comp, ratings, args.k_value)

        elo_labels = {item_id: rating > 0
                for item_id, rating in ratings.items()}
        comps.append(c)
        biases.append(bias(true_labels, elo_labels, bad_features))
        scores.append(score(true_labels, elo_labels))

    report["comparisons"] = {
            "number_of_comparisons": comps,
            "scores": scores, "biases": biases}

    print(json.dumps(report))
