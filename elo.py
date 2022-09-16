from enum import Enum
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Answer(Enum):
    left = "left"
    right = "right"
    equal = "equal"

class Player(Enum):
    player_a = "playerA"
    player_b = "playerB"

class Result(Enum):
    win = 1
    lose = 0
    tie = 0.5

def expected_score(ra, rb):
    return 1/(1 + 10**((ra - rb)/400))

def update_rating(r: float, real_score: float,
        expected_score:float, k_value: int):
    return r + k_value*(real_score - expected_score)

def make_match(task: dict, answer: dict):
    """ make a single elo match for a task and its corresponding answer """
    elo_match = {}
    elo_match[Player.player_a] = task["tweet1"]
    elo_match[Player.player_b] = task["tweet2"]

    if answer["answer"][0]["selection"] == Answer.left.value:
        elo_match["SA"] = Result.win.value
        elo_match["SB"] = Result.lose.value

    elif answer["answer"][0]["selection"] == Answer.right.value:
        elo_match["SA"] = Result.lose.value
        elo_match["SB"] = Result.win.value

    elif answer["answer"][0]["selection"] == Answer.equal.value:
        elo_match["SA"] = Result.tie.value
        elo_match["SB"] = Result.tie.value

    else:
        print(answer)
        print(elo_match)
        raise RuntimeError("bad match")

    return elo_match

def elo(match: dict, ratings, k):
    updated_ratings = ratings#.copy()
    tweet_id_a = match[Player.player_a]
    tweet_id_b = match[Player.player_b]
    ea = expected_score(updated_ratings[tweet_id_b],
            updated_ratings[tweet_id_a])
    eb = expected_score(updated_ratings[tweet_id_a],
            updated_ratings[tweet_id_b])
    updated_ratings[tweet_id_a] = update_rating(
            updated_ratings[tweet_id_a], match["SA"], ea, k)
    updated_ratings[tweet_id_b] = update_rating(
            updated_ratings[tweet_id_b], match["SB"], eb, k)

    return updated_ratings

def ratings_distance(rating1, rating2):
    if set(rating1) != set(rating2):
        raise RuntimeError("conversations are different!")
    size = len(rating1)
    return sum([(rating1[tid] - rating2[tid])**2 for tid in rating1])/size

def rankings_distance(rating1, rating2):
    if set(rating1) != set(rating2):
        raise RuntimeError("conversations are different!")
    root_tweet_ids = sorted(set(rating1))
    size = len(rating1)

    rankings1 = get_ranking(rating1)
    rankings2 = get_ranking(rating2)
    return sum((rankings2[u] - rankings1[u])**2 for u in root_tweet_ids)/size

def get_ranking(rating):
    root_tweet_ids = sorted(set(rating))
    convs_sorted = sorted(root_tweet_ids, key=rating.get, reverse=True)
    return {u: r for r, u in enumerate(convs_sorted)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate elo ratings")
    parser.add_argument("tasks", type=str, help="path to crowdsourcing tasks")
    parser.add_argument("answers", type=str, help="path to crowdsourcing answers")
    parser.add_argument("sample", type=int, help="number of items to use for simulating smaller run")
    parser.add_argument("seed", type=int, help="seed number for the generator")
    parser.add_argument("size", type=int, help="ensemble size")
    parser.add_argument("kvalue", type=int, help="k value for the elo ratings")
    parser.add_argument("true", type=str, help="simulated true ratings/rankings")
    parser.add_argument("ratingout", type=str, help="path to save the ratings")
    parser.add_argument("rankingout", type=str, help="path to save the rankings")
    parser.add_argument("avgout", type=str, help="path to save the averages")

    args = parser.parse_args()

    # Get all tasks and unique conversations
    all_tasks = {}
    conversations = set()
    with open(args.tasks) as tasks:
        for tsk in tasks:
            task = json.loads(tsk)
            idx = int(task["experiment_id"])

            all_tasks[idx] = task
            conversations.add(task["tweet1"])
            conversations.add(task["tweet2"])

    rng = np.random.default_rng(args.seed)
    conversations = set(rng.choice(
        sorted(conversations),
        size=args.sample,
        replace=False))

    # Get all matches
    elo_matches = []
    with open(args.answers) as answers:
        for ans in answers:
            answer = json.loads(ans)
            task = all_tasks[answer["task"]]
            if task["tweet1"] in conversations and task["tweet2"] in conversations:
                elo_matches.append(make_match(task, answer))

    # Get true ratings and tankings
    true_ratings = {}
    true_rankings = {}
    with open(args.true) as ftruth:
        for line in ftruth:
            res = json.loads(line)
            if res["conversation"] in conversations:
                true_ratings[res["conversation"]] = res["rating"]
                true_rankings[res["conversation"]] = res["ranking"]

    # Initializations
    ensemble_size = args.size
    k = args.kvalue
    average_rating_diffs = np.zeros(len(elo_matches))
    average_ranking_diffs = np.zeros(len(elo_matches))
    average_true_rating_diffs = np.zeros(len(elo_matches))
    average_true_ranking_diffs = np.zeros(len(elo_matches))

    # Ratings and rankings
    with open(args.ratingout, 'w') as ratout, open(args.rankingout, 'w') as rankout:
        for ens in tqdm(range(ensemble_size)):
            # Initializations
            ratings = {conv: 1500 for conv in conversations}
            convs_ratings = {conv: [] for conv in conversations}
            convs_rankings = {conv: [] for conv in conversations}

            rng.shuffle(elo_matches)

            rating_diffs = []
            ranking_diffs = []
            true_rating_diffs = []
            true_ranking_diffs = []

            for counter, match in enumerate(elo_matches):
                new_ratings = elo(match, ratings, k)
                rating_diffs.append(ratings_distance(ratings, new_ratings))
                ranking_diffs.append(rankings_distance(ratings, new_ratings))
                true_rating_diffs.append(ratings_distance(ratings, true_ratings))
                true_ranking_diffs.append(rankings_distance(ratings, true_ratings))
                ratings = new_ratings
                rankings = get_ranking(ratings)
                if counter%5 == 0:
                    for conv in conversations:
                        convs_ratings[conv].append(ratings[conv])
                        convs_rankings[conv].append(rankings[conv])

            average_rating_diffs += rating_diffs
            average_ranking_diffs += ranking_diffs
            average_true_rating_diffs += true_rating_diffs
            average_true_ranking_diffs += true_ranking_diffs

            print(json.dumps(convs_ratings), file=ratout)
            print(json.dumps(convs_rankings), file=rankout)



    average_rating_diffs /= (ensemble_size)
    average_ranking_diffs /= (ensemble_size)
    average_true_rating_diffs /= (ensemble_size)
    average_true_ranking_diffs /= (ensemble_size)

    result = {"cardinality": len(conversations), "k": k,
            "average_rating_diffs": list(average_rating_diffs),
            "average_ranking_diffs": list(average_ranking_diffs),
            "average_true_rating_diffs": list(average_true_rating_diffs),
            "average_true_ranking_diffs": list(average_true_ranking_diffs)}
    with open(args.avgout, 'w') as outfile:
        print(json.dumps(result), file=outfile)

