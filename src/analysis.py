"""
Analysis and visualization for experiment results.

Produces:
  1. Condition comparison (success rate, steps-to-goal across baselines)
  2. Generational learning curve (do later generations perform better?)
  3. Prior evolution analysis (how do priors change across generations?)
  4. Lineage tree visualization
  5. Signal discrimination analysis (hint vs distractor mention in priors)
"""
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# We'll use matplotlib for plots; fall back gracefully
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import math
import statistics


@dataclass
class ConditionSummary:
    condition: str
    num_trials: int
    successes: int
    success_rate: float
    avg_steps: float
    std_steps: float
    median_steps: float
    min_steps: int
    max_steps: int
    avg_agents_spawned: float


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def summarize_by_condition(results: list[dict]) -> dict[str, ConditionSummary]:
    """Aggregate results by experimental condition."""
    grouped = defaultdict(list)
    for r in results:
        grouped[r["condition"]].append(r)

    summaries = {}
    for condition, trials in grouped.items():
        steps = [t["steps_to_goal"] for t in trials if t["success"]]
        successes = sum(1 for t in trials if t["success"])
        agents = [t["num_agents_spawned"] for t in trials]

        summaries[condition] = ConditionSummary(
            condition=condition,
            num_trials=len(trials),
            successes=successes,
            success_rate=successes / len(trials) if trials else 0,
            avg_steps=statistics.mean(steps) if steps else 0,
            std_steps=statistics.stdev(steps) if len(steps) > 1 else 0,
            median_steps=statistics.median(steps) if steps else 0,
            min_steps=min(steps) if steps else 0,
            max_steps=max(steps) if steps else 0,
            avg_agents_spawned=statistics.mean(agents) if agents else 0,
        )
    return summaries


def print_summary_table(summaries: dict[str, ConditionSummary]):
    """Print a readable comparison table."""
    print("\n" + "=" * 85)
    print(f"{'Condition':<15} {'Trials':>7} {'Success':>8} {'Rate':>7} "
          f"{'AvgSteps':>9} {'StdSteps':>9} {'MedSteps':>9} {'AvgAgents':>10}")
    print("-" * 85)
    for s in summaries.values():
        print(f"{s.condition:<15} {s.num_trials:>7} {s.successes:>8} {s.success_rate:>7.2f} "
              f"{s.avg_steps:>9.1f} {s.std_steps:>9.1f} {s.median_steps:>9.1f} {s.avg_agents_spawned:>10.1f}")
    print("=" * 85)


def plot_condition_comparison(summaries: dict[str, ConditionSummary], output_dir: str):
    """Bar chart comparing conditions."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return

    conditions = list(summaries.keys())
    rates = [summaries[c].success_rate for c in conditions]
    avg_steps = [summaries[c].avg_steps for c in conditions]
    std_steps = [summaries[c].std_steps for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    axes[0].bar(conditions, rates, color=colors[:len(conditions)], alpha=0.8)
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Success Rate by Condition")
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(rates):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

    # Avg steps to goal
    axes[1].bar(conditions, avg_steps, yerr=std_steps, color=colors[:len(conditions)],
                alpha=0.8, capsize=5)
    axes[1].set_ylabel("Avg Steps to Goal")
    axes[1].set_title("Steps to Goal by Condition (successful trials)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved condition_comparison.png")


def analyze_generational_learning(results: list[dict], output_dir: str):
    """
    The key plot: do later generations find the goal faster?
    
    Aggregates steps-to-goal by agent generation across all experimental trials.
    """
    gen_steps = defaultdict(list)

    for r in results:
        if r["condition"] != "experimental":
            continue
        for gen_str, stats in r.get("generation_stats", {}).items():
            gen = int(gen_str)
            if stats["avg_steps"] > 0:
                gen_steps[gen].append(stats["avg_steps"])

    if not gen_steps:
        print("No generational data to analyze.")
        return

    generations = sorted(gen_steps.keys())
    means = [statistics.mean(gen_steps[g]) for g in generations]
    stds = [statistics.stdev(gen_steps[g]) if len(gen_steps[g]) > 1 else 0 for g in generations]
    counts = [len(gen_steps[g]) for g in generations]

    print("\nGenerational Learning Curve:")
    print(f"{'Generation':>12} {'AvgSteps':>10} {'Std':>10} {'N':>5}")
    for g, m, s, n in zip(generations, means, stds, counts):
        print(f"{g:>12} {m:>10.1f} {s:>10.1f} {n:>5}")

    if HAS_MATPLOTLIB and len(generations) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(generations, means, yerr=stds, marker='o', capsize=5,
                    linewidth=2, markersize=8, color='#2196F3')
        ax.set_xlabel("Agent Generation")
        ax.set_ylabel("Avg Steps to Goal")
        ax.set_title("Generational Learning Curve\n(Do offspring find the goal faster?)")
        ax.set_xticks(generations)

        # Add sample sizes
        for g, m, n in zip(generations, means, counts):
            ax.annotate(f"n={n}", (g, m), textcoords="offset points",
                       xytext=(10, 10), fontsize=9, alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "generational_learning.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved generational_learning.png")


def analyze_prior_evolution(results: list[dict], output_dir: str):
    """
    Analyze how priors change across generations.
    
    - Extract all priors grouped by generation
    - Compute simple text-based similarity (Jaccard over words)
    - Track which keywords appear/disappear across generations
    """
    gen_priors = defaultdict(list)

    for r in results:
        if r["condition"] != "experimental":
            continue
        for prior_entry in r.get("all_priors", []):
            gen = prior_entry["generation"]
            prior_text = prior_entry["prior"]
            gen_priors[gen].append(prior_text)

    if not gen_priors:
        print("No priors to analyze.")
        return

    print("\nPrior Evolution Analysis:")
    print("-" * 60)

    # Keyword frequency per generation
    directional_keywords = {"north", "south", "east", "west", "upper", "lower", "left", "right", "row", "column"}
    identity_keywords = {"red", "blue", "green", "yellow", "star", "circle", "triangle", "square"}
    trust_keywords = {"trust", "reliable", "true", "correct", "accurate"}
    distrust_keywords = {"ignore", "misleading", "false", "wrong", "distractor", "unreliable"}

    keyword_categories = {
        "directional": directional_keywords,
        "identity": identity_keywords,
        "trust": trust_keywords,
        "distrust": distrust_keywords,
    }

    gen_keyword_freq = {}
    for gen in sorted(gen_priors.keys()):
        priors = gen_priors[gen]
        all_words = set()
        for p in priors:
            all_words.update(p.lower().split())

        freqs = {}
        for cat_name, keywords in keyword_categories.items():
            count = sum(1 for w in all_words if w in keywords)
            freqs[cat_name] = count
        gen_keyword_freq[gen] = freqs

        print(f"\nGeneration {gen} ({len(priors)} priors):")
        for cat, count in freqs.items():
            print(f"  {cat}: {count} keyword mentions")
        if priors:
            print(f"  Sample prior: {priors[0][:150]}...")

    # Save prior texts for manual inspection
    prior_dump = {}
    for gen in sorted(gen_priors.keys()):
        prior_dump[f"gen_{gen}"] = gen_priors[gen]

    with open(os.path.join(output_dir, "priors_by_generation.json"), "w") as f:
        json.dump(prior_dump, f, indent=2)
    print(f"\nSaved priors_by_generation.json")


def analyze_signal_discrimination(results: list[dict], output_dir: str):
    """
    Do successful agents' priors mention more hints than distractors?
    
    This is indirect evidence that priors encode useful signal discrimination.
    """
    # This requires knowing which signals were hints vs distractors
    # We can check if prior text mentions goal-relevant info vs misleading info
    # For now, we track keyword overlap as a proxy
    
    successful_priors = []
    failed_priors = []

    for r in results:
        if r["condition"] != "experimental":
            continue
        if r["success"] and r["successful_agent_prior"]:
            successful_priors.append(r["successful_agent_prior"])
        for prior_entry in r.get("all_priors", []):
            # Priors from agents that didn't succeed
            agent_id = prior_entry.get("agent_id", "")
            if agent_id != r.get("successful_agent_id"):
                failed_priors.append(prior_entry["prior"])

    print(f"\nSignal Discrimination:")
    print(f"  Successful agent priors: {len(successful_priors)}")
    print(f"  Other priors: {len(failed_priors)}")

    if successful_priors:
        print(f"\n  Sample successful prior:")
        print(f"    {successful_priors[0][:300]}")
    if failed_priors:
        print(f"\n  Sample other prior:")
        print(f"    {failed_priors[0][:300]}")


def run_full_analysis(results_path: str, output_dir: str):
    """Run all analyses on saved results."""
    results = load_results(results_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Condition comparison
    summaries = summarize_by_condition(results)
    print_summary_table(summaries)
    plot_condition_comparison(summaries, output_dir)

    # 2. Generational learning
    analyze_generational_learning(results, output_dir)

    # 3. Prior evolution
    analyze_prior_evolution(results, output_dir)

    # 4. Signal discrimination
    analyze_signal_discrimination(results, output_dir)

    return summaries
