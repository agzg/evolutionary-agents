"""
Reproduction system: prior compression, agent birth, lineage tracking.

When an agent hits its interaction limit, it:
  1. Compresses its experience into a prior
  2. Optionally mutates the prior (controlled by mutation_rate)
  3. Spawns a child agent with the prior as initial knowledge
  4. The lineage is tracked for later analysis
"""
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.agent import Agent, AgentType, AgentState


@dataclass
class LineageNode:
    agent_id: str
    parent_id: Optional[str]
    generation: int
    prior: str
    success: bool = False
    steps_taken: int = 0
    children: list[str] = field(default_factory=list)


class LineageTracker:
    """Tracks the full parent→child tree across an experiment run."""

    def __init__(self):
        self.nodes: dict[str, LineageNode] = {}

    def register(self, agent: Agent):
        node = LineageNode(
            agent_id=agent.state.agent_id,
            parent_id=agent.state.parent_id,
            generation=agent.state.generation,
            prior=agent.state.prior,
        )
        self.nodes[agent.state.agent_id] = node
        if agent.state.parent_id and agent.state.parent_id in self.nodes:
            self.nodes[agent.state.parent_id].children.append(agent.state.agent_id)

    def mark_success(self, agent_id: str, steps: int):
        if agent_id in self.nodes:
            self.nodes[agent_id].success = True
            self.nodes[agent_id].steps_taken = steps

    def mark_failure(self, agent_id: str, steps: int):
        if agent_id in self.nodes:
            self.nodes[agent_id].steps_taken = steps

    def get_tree(self) -> dict:
        """Export lineage as a serializable dict."""
        return {
            aid: {
                "parent": node.parent_id,
                "generation": node.generation,
                "prior_preview": node.prior[:100] + "..." if len(node.prior) > 100 else node.prior,
                "full_prior": node.prior,
                "success": node.success,
                "steps_taken": node.steps_taken,
                "children": node.children,
            }
            for aid, node in self.nodes.items()
        }

    def get_generation_stats(self) -> dict[int, dict]:
        """Aggregate stats by generation."""
        gen_data: dict[int, list[LineageNode]] = {}
        for node in self.nodes.values():
            gen_data.setdefault(node.generation, []).append(node)

        stats = {}
        for gen, nodes in sorted(gen_data.items()):
            successes = [n for n in nodes if n.success]
            all_steps = [n.steps_taken for n in nodes if n.steps_taken > 0]
            stats[gen] = {
                "total_agents": len(nodes),
                "successes": len(successes),
                "success_rate": len(successes) / len(nodes) if nodes else 0,
                "avg_steps": sum(all_steps) / len(all_steps) if all_steps else 0,
                "min_steps": min(all_steps) if all_steps else 0,
                "max_steps": max(all_steps) if all_steps else 0,
            }
        return stats


def mutate_prior(prior: str, mutation_rate: float, llm, rng: random.Random) -> str:
    """
    Mutate a prior with some probability.

    mutation_rate = 0.0: exact copy (no mutation)
    mutation_rate = 1.0: completely rephrase
    """
    if mutation_rate <= 0 or not prior:
        return prior

    if rng.random() > mutation_rate:
        return prior  # no mutation this time

    # Ask LLM to rephrase/challenge one assumption
    try:
        result = llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=(
                "You are introducing slight variation into exploration knowledge. "
                "Take the following set of heuristics and change ONE of them — either "
                "rephrase it, weaken its certainty, or challenge its assumption. "
                "Keep all other heuristics intact. Return the full modified set."
            ),
            messages=[
                {"role": "user", "content": f"Original heuristics:\n{prior}\n\nModified version:"},
            ],
        )
        return result.content[0].text[:500]
    except Exception:
        return prior  # on failure, don't mutate


def birth_agent(
    parent: Agent,
    llm,
    config,
    rng: random.Random,
) -> Agent:
    """
    Create a child agent from a parent.

    1. Parent generates prior from its experience
    2. Prior is optionally mutated
    3. Child is created with the prior as initial knowledge
    """
    # Generate prior
    prior = parent.generate_prior()

    # Mutate if configured
    if config.prior_mutation_rate > 0:
        prior = mutate_prior(prior, config.prior_mutation_rate, llm, rng)

    # Create child
    child_id = f"agent_g{parent.state.generation + 1}_{uuid.uuid4().hex[:6]}"
    child = Agent(
        agent_id=child_id,
        llm=llm,
        config=config,
        agent_type=AgentType.EXPERIMENTAL,
        prior=prior,
        generation=parent.state.generation + 1,
        parent_id=parent.state.agent_id,
        rng=rng,
        graph_mode=parent.graph_mode,
    )

    parent.state.children_ids.append(child_id)

    return child
