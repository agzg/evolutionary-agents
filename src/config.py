"""
Experiment configuration for Multi-Agent POMDP Door Navigation.

All tunable parameters live here so experiments are fully reproducible.
"""
from dataclasses import dataclass, field
from typing import Optional
import json
import time


@dataclass
class ExperimentConfig:
    # --- Environment ---
    use_graph_world: bool = True  # True = random geometric graph, False = rectangular grid
    num_doors: int = 4

    # Graph world params (when use_graph_world=True)
    num_nodes: int = 20  # total nodes in random geometric graph
    connection_radius: float = 0.35  # nodes within this distance are connected
    observation_hops: int = 1  # how far agent can see in the graph

    # Grid world params (when use_graph_world=False)
    grid_size: tuple[int, int] = (6, 6)
    num_walls: int = 5
    observation_radius: int = 1

    # --- Signals ---
    hints_per_door: int = 3
    distractors_per_door: int = 3
    max_signals_per_observation: int = 3  # how many signals agent sees when adjacent to door

    # --- Agent ---
    max_context_tokens: int = 1500  # approximate token budget for working memory
    max_context_entries: int = 15  # max (obs, action, reasoning) triples before compression
    interactions_per_lifetime: int = 50  # interactions before reproduction
    max_steps_per_trial: int = 300  # hard cap to prevent infinite loops

    # --- Reproduction ---
    prior_mutation_rate: float = 0.0  # 0.0 = exact copy, 1.0 = full rephrase (set >0 for mutation experiment)
    parent_resets_after_birth: bool = True  # reset parent interaction count after spawning child
    max_generations: int = 5  # prevent runaway spawning

    # --- Experiment ---
    success_threshold: int = 3  # stop after N successful agents
    num_trials: int = 10  # number of independent experiment runs
    num_root_agents: int = 1  # starting agents per trial
    random_seed: Optional[int] = 42

    # --- Baselines ---
    run_no_prior_baseline: bool = True
    run_random_walk_baseline: bool = True
    run_oracle_prior_baseline: bool = True
    run_random_prior_baseline: bool = True  # gibberish prior control

    # --- Model ---
    model_name: str = "claude-haiku-4-5-20251001"  # fast + cheap; upgrade to claude-sonnet-4-20250514 for final runs

    # --- Output ---
    results_dir: str = "results"

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**json.load(f))

    def experiment_id(self) -> str:
        return f"exp_{int(time.time())}"
