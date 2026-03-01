"""
Stratified city split generator.

Design principles:
  1. Geographic balance   — every country with >=4 cities contributes to all
                            three splits; countries with <=3 cities go to train.
  2. Density balance      — within each country, val and test pick cities at
                            different avg-population quantiles (not all the
                            same density tier).
  3. Cluster balance      — all 5 urban-fabric clusters (0-4) must appear in
                            both val and test.
  4. Special cases        — Edinburgh and Copenhagen have avg_pop=0 (broken
                            population data) → always assigned to train.
  5. France contributes 2 cities to val and 2 to test (10 cities, largest
                            country group) to reach exactly 12/12/56.

Final split: train=56  val=12  test=12  total=80

Run once, commit city_split.json to git — never changes again.

Usage:
    python data/generate_split.py \
        --json /workspace/PopulationDataset/final_clustered_samples.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).parent / "city_split.json"

# Cities with broken population data (avg_pop=0) — always in train
BROKEN_CITIES = {"Edinburgh, United Kingdom", "Copenhagen, Denmark"}


def _city_profiles(data: list) -> dict:
    """Compute avg_population and dominant cluster for each city."""
    profiles = defaultdict(lambda: {"pops": [], "clusters": []})
    for s in data:
        profiles[s["city"]]["pops"].append(s["population_15min_walk"])
        profiles[s["city"]]["clusters"].append(s.get("cluster", 0))
    result = {}
    for city, p in profiles.items():
        result[city] = {
            "avg_pop":     float(np.mean(p["pops"])),
            "dom_cluster": int(np.bincount(p["clusters"]).argmax()),
            "n":           len(p["pops"]),
            "country":     city.split(", ")[-1],
        }
    return result


def _pick_val_test(cities_sorted: list, profiles: dict, n_val: int, n_test: int):
    """
    Given cities sorted by avg_pop, pick n_val for val and n_test for test
    at spread-out quantile positions so the two sets cover different density
    levels.  Cities in BROKEN_CITIES are skipped.
    """
    eligible = [c for c in cities_sorted if c not in BROKEN_CITIES]
    n = len(eligible)
    if n < n_val + n_test:
        # Not enough cities — put all in train
        return [], []

    # Pick indices spread across the sorted list
    # val: lower quantile(s)   test: upper quantile(s)
    step = n / (n_val + n_test + 1)
    indices = [int(step * (i + 1)) for i in range(n_val + n_test)]

    val_cities  = [eligible[indices[i]] for i in range(n_val)]
    test_cities = [eligible[indices[i]] for i in range(n_val, n_val + n_test)]
    return val_cities, test_cities


def generate(json_file: str):
    with open(json_file) as f:
        raw = json.load(f)
    data = [s for s in raw if s.get("population_15min_walk") is not None]

    profiles = _city_profiles(data)

    # Group cities by country, sorted by avg_pop within each group
    country_groups: dict[str, list] = defaultdict(list)
    for city, p in profiles.items():
        country_groups[p["country"]].append(city)
    for country in country_groups:
        country_groups[country].sort(key=lambda c: profiles[c]["avg_pop"])

    train, val, test = [], [], []

    for country, cities in sorted(country_groups.items()):
        n = len(cities)

        if n <= 3:
            # Small group — all to train, keep their diversity for training
            train.extend(cities)

        elif country == "France":
            # France has 10 cities → contributes 2 val + 2 test for balance
            broken = [c for c in cities if c in BROKEN_CITIES]
            eligible = [c for c in cities if c not in BROKEN_CITIES]
            v, t = _pick_val_test(eligible, profiles, n_val=2, n_test=2)
            val.extend(v)
            test.extend(t)
            train.extend(broken + [c for c in eligible if c not in v + t])

        else:
            # Standard: 1 val + 1 test
            broken = [c for c in cities if c in BROKEN_CITIES]
            eligible = [c for c in cities if c not in BROKEN_CITIES]
            if len(eligible) >= 2:
                v, t = _pick_val_test(eligible, profiles, n_val=1, n_test=1)
                val.extend(v)
                test.extend(t)
                train.extend(broken + [c for c in eligible if c not in v + t])
            else:
                train.extend(cities)

    # ── Override specific cities to ensure all clusters appear in val+test ──
    # After the quantile-based assignment, check cluster coverage and
    # swap cities if clusters 3 or 4 are missing from val/test.
    def _cluster_coverage(city_list):
        return set(profiles[c]["dom_cluster"] for c in city_list)

    def _swap_for_cluster(target_list, source_list, want_cluster):
        """Move a city with want_cluster from source_list into target_list."""
        candidates = [c for c in source_list
                      if profiles[c]["dom_cluster"] == want_cluster
                      and c not in BROKEN_CITIES]
        if not candidates:
            return  # cluster not available
        # Pick the candidate closest to the median avg_pop of target_list
        target_med = float(np.median([profiles[c]["avg_pop"] for c in target_list]))
        best = min(candidates, key=lambda c: abs(profiles[c]["avg_pop"] - target_med))
        # Swap: replace the city in target_list whose cluster is most over-represented
        cluster_counts = defaultdict(list)
        for c in target_list:
            cluster_counts[profiles[c]["dom_cluster"]].append(c)
        most_common = max(cluster_counts, key=lambda k: len(cluster_counts[k]))
        swap_out = cluster_counts[most_common][0]
        target_list.remove(swap_out)
        source_list.remove(best)
        target_list.append(best)
        source_list.append(swap_out)

    for target in [val, test]:
        for cluster in [3, 4]:
            if cluster not in _cluster_coverage(target):
                _swap_for_cluster(target, train, cluster)

    # ── Verification ──────────────────────────────────────────────────────
    all_cities = set(profiles.keys())
    assigned   = set(train + val + test)
    assert assigned == all_cities, \
        f"Cities not assigned: {all_cities - assigned}"
    assert not (set(train) & set(val) & set(test)), \
        "Overlap between splits!"
    assert len(val)  == 12, f"Expected 12 val cities, got {len(val)}"
    assert len(test) == 12, f"Expected 12 test cities, got {len(test)}"
    assert len(train) == len(all_cities) - 24

    # ── Count samples ─────────────────────────────────────────────────────
    city_to_split = {c: "train" for c in train}
    city_to_split.update({c: "val"   for c in val})
    city_to_split.update({c: "test"  for c in test})
    counts = {"train": 0, "val": 0, "test": 0}
    for s in data:
        sp = city_to_split.get(s["city"])
        if sp:
            counts[sp] += 1

    split = {
        "seed":        42,
        "train_ratio": 0.70,
        "val_ratio":   0.15,
        "strategy":    "stratified_by_country_and_density",
        "n_cities":    {"train": len(train), "val": len(val), "test": len(test)},
        "n_samples":   counts,
        "train":       sorted(train),
        "val":         sorted(val),
        "test":        sorted(test),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nSplit saved to {OUT_PATH}")
    print(f"\n  train : {len(train):>2} cities  ({counts['train']:>6} samples)")
    print(f"  val   : {len(val):>2} cities  ({counts['val']:>6} samples)")
    print(f"  test  : {len(test):>2} cities  ({counts['test']:>6} samples)")

    for split_name, city_list in [("VAL", val), ("TEST", test)]:
        print(f"\n{split_name} cities:")
        for country in sorted({profiles[c]["country"] for c in city_list}):
            cities_in_country = [c for c in city_list
                                  if profiles[c]["country"] == country]
            for c in cities_in_country:
                p = profiles[c]
                print(f"  {c:<40} avg_pop={p['avg_pop']:>7.0f}  "
                      f"cluster={p['dom_cluster']}  n={p['n']}")

    val_clusters  = _cluster_coverage(val)
    test_clusters = _cluster_coverage(test)
    print(f"\nCluster coverage — val : {sorted(val_clusters)}")
    print(f"Cluster coverage — test: {sorted(test_clusters)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    generate(args.json)
