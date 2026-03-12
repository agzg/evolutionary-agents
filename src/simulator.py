"""
Experiment simulator.

Runs the full multi-agent POMDP experiment:
  1. Generates a world (grid or graph) with doors and signals
  2. Runs agents (experimental + baselines) through the environment
  3. Handles reproduction and prior inheritance
  4. Collects data for analysis

Each "trial" is an independent run with a fresh world.
Multiple trials give us statistical power.
"""
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.config import ExperimentConfig
from src.grid_world import GridWorld, Action, CellType
from src.graph_world import GraphWorld
from src.signals import generate_all_signals_for_grid, generate_all_signals_for_graph
from src.agent import Agent, AgentType, generate_random_prior
from src.reproduction import LineageTracker, birth_agent


@dataclass
class TrialResult:
    trial_id: int
    condition: str  # "experimental", "no_prior", "random_walk", "oracle", "random_prior"
    success: bool
    steps_to_goal: int  # 0 if failed
    total_interactions: int
    num_agents_spawned: int
    successful_agent_id: Optional[str] = None
    successful_agent_generation: int = 0
    successful_agent_prior: str = ""
    lineage_tree: dict = field(default_factory=dict)
    generation_stats: dict = field(default_factory=dict)
    all_priors: list[dict] = field(default_factory=list)
    grid_description: str = ""
    graph_stats: dict = field(default_factory=dict)


class Simulator:
    """
    Runs the multi-agent POMDP experiment.
    """

    def __init__(self, config: ExperimentConfig, llm):
        self.config = config
        self.llm = llm
        self.results: list[TrialResult] = []

    def run_all(self) -> list[TrialResult]:
        """Run all trials across all conditions."""
        all_results = []

        conditions = [("experimental", AgentType.EXPERIMENTAL)]
        if self.config.run_no_prior_baseline:
            conditions.append(("no_prior", AgentType.NO_PRIOR))
        if self.config.run_random_walk_baseline:
            conditions.append(("random_walk", AgentType.RANDOM_WALK))
        if self.config.run_oracle_prior_baseline:
            conditions.append(("oracle", AgentType.ORACLE))
        if self.config.run_random_prior_baseline:
            conditions.append(("random_prior", AgentType.RANDOM_PRIOR))

        for trial_idx in range(self.config.num_trials):
            # Use same world across conditions for fair comparison
            trial_seed = (self.config.random_seed or 0) + trial_idx
            rng = random.Random(trial_seed)

            # Build world template
            world_template = self._build_world(rng)

            for condition_name, agent_type in conditions:
                print(f"\n--- Trial {trial_idx + 1}/{self.config.num_trials}, Condition: {condition_name} ---")
                result = self._run_single_trial(
                    trial_idx, condition_name, agent_type, world_template, trial_seed
                )
                all_results.append(result)
                print(f"  Result: {'SUCCESS' if result.success else 'FAIL'} "
                      f"| Steps: {result.steps_to_goal} | Agents: {result.num_agents_spawned}")

        self.results = all_results
        return all_results

    def _build_world(self, rng: random.Random) -> dict:
        """Build a world template (layout + signals) that can be reused across conditions."""
        world_seed = rng.randint(0, 10**6)
        signal_seed = rng.randint(0, 10**6)

        if self.config.use_graph_world:
            world = GraphWorld(self.config, rng=random.Random(world_seed))
            signals = generate_all_signals_for_graph(
                world,
                self.config.hints_per_door, self.config.distractors_per_door,
                rng=random.Random(signal_seed),
            )
            return {
                "seed": world_seed,
                "signals": signals,
                "world_desc": world.get_description(),
                "graph_stats": world.get_graph_stats(),
            }
        else:
            world = GridWorld(self.config, rng=random.Random(world_seed))
            signals = generate_all_signals_for_grid(
                world.doors, world.goal_door, self.config.grid_size,
                self.config.hints_per_door, self.config.distractors_per_door,
                rng=random.Random(signal_seed),
            )
            return {
                "seed": world_seed,
                "signals": signals,
                "world_desc": world.get_grid_description(),
            }

    def _create_world_for_agent(self, template: dict, rng: random.Random):
        """Create a fresh world instance with a randomized starting position."""
        if self.config.use_graph_world:
            world = GraphWorld(self.config, rng=random.Random(template["seed"]))
            # Randomize starting position
            non_door = [n for n in world.nodes if n not in world.door_nodes]
            if non_door:
                world.agent_node = rng.choice(non_door)
        else:
            world = GridWorld(self.config, rng=random.Random(template["seed"]))
            # Randomize starting position
            empty = [(r, c) for r in range(world.rows) for c in range(world.cols)
                     if world.grid[r][c] == CellType.EMPTY]
            if empty:
                world.agent_pos = rng.choice(empty)
        world.set_signals(template["signals"])
        return world

    @staticmethod
    def _str_to_action(action_str: str) -> Action:
        """Convert a string action to a grid-world Action enum."""
        action_map = {
            "north": Action.NORTH, "south": Action.SOUTH,
            "east": Action.EAST, "west": Action.WEST,
            "stay": Action.STAY,
        }
        return action_map.get(action_str.lower(), Action.STAY)

    def _run_single_trial(
        self,
        trial_idx: int,
        condition_name: str,
        agent_type: AgentType,
        world_template: dict,
        trial_seed: int,
    ) -> TrialResult:
        """Run a single trial for one condition."""
        rng = random.Random(trial_seed)
        use_graph = self.config.use_graph_world

        lineage = LineageTracker()
        active_agents: list[Agent] = []
        agent_worlds: dict[str, object] = {}  # agent_id -> world instance
        success_count = 0
        total_interactions = 0
        result = TrialResult(
            trial_id=trial_idx,
            condition=condition_name,
            success=False,
            steps_to_goal=0,
            total_interactions=0,
            num_agents_spawned=0,
            grid_description=world_template["world_desc"],
            graph_stats=world_template.get("graph_stats", {}),
        )

        # Create initial root agent(s)
        for i in range(self.config.num_root_agents):
            agent_id = f"root_{condition_name}_{i}"
            prior = ""
            if agent_type == AgentType.ORACLE:
                prior = world_template["world_desc"]
            elif agent_type == AgentType.RANDOM_PRIOR:
                prior = generate_random_prior(rng)

            agent = Agent(
                agent_id=agent_id,
                llm=self.llm,
                config=self.config,
                agent_type=agent_type,
                prior=prior,
                generation=0,
                rng=rng,
                graph_mode=use_graph,
            )
            active_agents.append(agent)
            lineage.register(agent)
            result.num_agents_spawned += 1
            agent_worlds[agent_id] = self._create_world_for_agent(world_template, rng)

        # Main loop
        max_total_steps = self.config.max_steps_per_trial * 3  # generous cap
        step = 0

        while step < max_total_steps and success_count < self.config.success_threshold:
            if not active_agents:
                break

            for agent in list(active_agents):
                world = agent_worlds[agent.agent_id]
                obs = world.get_observation()

                action_str = agent.decide(obs)

                # Convert action for the world
                if use_graph:
                    neighbor_idx = world.parse_action(
                        action_str, list(world.nodes[world.agent_node].neighbors)
                    )
                    _, reward, done = world.step(neighbor_idx)
                else:
                    action_enum = self._str_to_action(action_str)
                    _, reward, done = world.step(action_enum)

                total_interactions += 1

                if reward > 0:
                    # Success!
                    success_count += 1
                    agent.state.success = True
                    lineage.mark_success(agent.agent_id, agent.state.total_steps)
                    result.success = True
                    result.steps_to_goal = agent.state.total_steps
                    result.successful_agent_id = agent.agent_id
                    result.successful_agent_generation = agent.state.generation
                    result.successful_agent_prior = agent.state.prior

                    if success_count >= self.config.success_threshold:
                        break

                    active_agents.remove(agent)
                    del agent_worlds[agent.agent_id]
                    continue

                if done:
                    # Failed (hit step limit)
                    lineage.mark_failure(agent.agent_id, agent.state.total_steps)
                    active_agents.remove(agent)
                    del agent_worlds[agent.agent_id]
                    continue

                # Check reproduction
                if agent.should_reproduce():
                    child = birth_agent(agent, self.llm, self.config, rng)
                    active_agents.append(child)
                    lineage.register(child)
                    result.num_agents_spawned += 1

                    # Collect prior for analysis
                    result.all_priors.append({
                        "agent_id": child.agent_id,
                        "parent_id": agent.agent_id,
                        "generation": child.state.generation,
                        "prior": child.state.prior,
                    })

                    # Create world for child
                    agent_worlds[child.agent_id] = self._create_world_for_agent(
                        world_template, rng
                    )

                    if self.config.parent_resets_after_birth:
                        agent.state.interaction_count = 0

            step += 1

        # Finalize
        result.total_interactions = total_interactions
        result.lineage_tree = lineage.get_tree()
        result.generation_stats = lineage.get_generation_stats()

        return result


def save_results(results: list[TrialResult], config: ExperimentConfig, output_dir: str):
    """Save all trial results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config.save(os.path.join(output_dir, "config.json"))

    # Save results
    serializable = []
    for r in results:
        serializable.append({
            "trial_id": r.trial_id,
            "condition": r.condition,
            "success": r.success,
            "steps_to_goal": r.steps_to_goal,
            "total_interactions": r.total_interactions,
            "num_agents_spawned": r.num_agents_spawned,
            "successful_agent_id": r.successful_agent_id,
            "successful_agent_generation": r.successful_agent_generation,
            "successful_agent_prior": r.successful_agent_prior,
            "lineage_tree": r.lineage_tree,
            "generation_stats": r.generation_stats,
            "all_priors": r.all_priors,
            "grid_description": r.grid_description,
            "graph_stats": r.graph_stats,
        })

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
