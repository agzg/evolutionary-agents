"""
LLM Agent for POMDP navigation.

The agent:
  1. Receives text observations from the grid or graph world
  2. Maintains a bounded context buffer (simulating limited working memory)
  3. Uses an LLM to decide actions and reason about signals
  4. Compresses knowledge into priors for reproduction

The prior is the key mechanism: it's a natural language summary of what the agent
has learned, passed to offspring as their initial system prompt augmentation.
"""
import re
import json
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AgentType(Enum):
    EXPERIMENTAL = "experimental"  # normal agent with prior inheritance
    NO_PRIOR = "no_prior"  # baseline: no prior at all
    RANDOM_WALK = "random_walk"  # baseline: random actions
    ORACLE = "oracle"  # baseline: given perfect information
    RANDOM_PRIOR = "random_prior"  # control: given gibberish prior


@dataclass
class ContextEntry:
    """Single entry in the agent's working memory."""
    step: int
    observation_summary: str
    action_taken: str
    reasoning: str

    def to_text(self) -> str:
        return f"[Step {self.step}] Obs: {self.observation_summary} | Action: {self.action_taken} | Reasoning: {self.reasoning}"


@dataclass
class AgentState:
    agent_id: str
    agent_type: AgentType
    generation: int = 0
    parent_id: Optional[str] = None
    prior: str = ""  # inherited knowledge from parent
    context_buffer: list[ContextEntry] = field(default_factory=list)
    interaction_count: int = 0
    total_steps: int = 0
    success: bool = False
    trajectory: list = field(default_factory=list)  # positions visited
    signals_seen: list[dict] = field(default_factory=list)  # for analysis
    children_ids: list[str] = field(default_factory=list)


class Agent:
    """
    LLM-powered agent that navigates the POMDP grid or graph world.

    Uses the Anthropic API for:
      - Action selection given observations
      - Context compression when memory fills up
      - Prior generation for offspring
    """

    def __init__(
        self,
        agent_id: str,
        llm,  # anthropic.Anthropic client (or mock)
        config,
        agent_type: AgentType = AgentType.EXPERIMENTAL,
        prior: str = "",
        generation: int = 0,
        parent_id: Optional[str] = None,
        rng: Optional[random.Random] = None,
        graph_mode: bool = False,
    ):
        self.llm = llm
        self.config = config
        self.rng = rng or random.Random()
        self.graph_mode = graph_mode
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            generation=generation,
            parent_id=parent_id,
            prior=prior,
        )

    @property
    def agent_id(self) -> str:
        return self.state.agent_id

    def _build_system_prompt(self) -> str:
        """Construct the system prompt including any inherited prior."""
        if self.graph_mode:
            base = (
                "You are an agent navigating a graph-based world. Your goal is to find the GOAL DOOR.\n"
                "The graph has nodes, some of which are doors with signals - some are true hints, some are misleading distractors.\n"
                "You must figure out which signals to trust.\n\n"
                "Available actions: move to any neighboring node by saying 'node <id>', or 'stay'\n\n"
                "When you respond, use EXACTLY this format:\n"
                "REASONING: <one sentence about what you think is happening>\n"
                "ACTION: <node ID or stay>\n"
            )
        else:
            base = (
                "You are an agent navigating a grid world. Your goal is to find the GOAL DOOR.\n"
                "The grid has doors with signals - some are true hints, some are misleading distractors.\n"
                "You must figure out which signals to trust.\n\n"
                "Available actions: north, south, east, west, stay\n\n"
                "When you respond, use EXACTLY this format:\n"
                "REASONING: <one sentence about what you think is happening>\n"
                "ACTION: <north|south|east|west|stay>\n"
            )

        if self.state.agent_type == AgentType.ORACLE:
            base += f"\n[ORACLE INFO]: {self.state.prior}\n"
            if self.graph_mode:
                base += (
                    "IMPORTANT: Navigate to the goal step by step. Each turn you can ONLY move to "
                    "a directly neighboring node listed in your observation. Always respond with "
                    "'node <id>' using one of the neighbor IDs shown in 'You can move to:'.\n"
                )
        elif self.state.prior and self.state.agent_type != AgentType.NO_PRIOR:
            base += f"\n[INHERITED KNOWLEDGE FROM PREVIOUS EXPLORER]:\n{self.state.prior}\n"

        return base

    def _build_context_text(self) -> str:
        """Build the recent context from the buffer."""
        if not self.state.context_buffer:
            return "No previous observations yet."
        entries = [e.to_text() for e in self.state.context_buffer]
        return "\n".join(entries)

    def decide(self, observation) -> str:
        """
        Given an observation, decide on an action.

        Returns a string action:
          - Grid mode: "north", "south", "east", "west", or "stay"
          - Graph mode: raw LLM text like "node 5" or "stay"
        """
        if self.state.agent_type == AgentType.RANDOM_WALK:
            if self.graph_mode:
                neighbor_ids = getattr(observation, 'neighbor_ids', [])
                if neighbor_ids:
                    action_str = f"node {self.rng.choice(neighbor_ids)}"
                else:
                    action_str = "stay"
            else:
                action_str = self.rng.choice(["north", "south", "east", "west", "stay"])
            self.state.interaction_count += 1
            self.state.total_steps += 1
            return action_str

        obs_text = observation.to_text()

        # Build messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": (
                f"RECENT MEMORY:\n{self._build_context_text()}\n\n"
                f"CURRENT OBSERVATION:\n{obs_text}\n\n"
                f"Decide your next action. Remember: REASONING then ACTION."
            )}
        ]

        # Call LLM
        try:
            response = self._call_llm(messages)
            action_str, reasoning = self._parse_response(response)
        except Exception as e:
            response = None
            # Fallback to random on parse/API failure
            if self.graph_mode:
                neighbor_ids = getattr(observation, 'neighbor_ids', [])
                if neighbor_ids:
                    action_str = f"node {self.rng.choice(neighbor_ids)}"
                else:
                    action_str = "stay"
            else:
                action_str = self.rng.choice(["north", "south", "east", "west", "stay"])
            reasoning = f"[LLM error: {str(e)[:50]}]"

        # Debug logging for oracle agents (first 3 steps)
        if self.state.agent_type == AgentType.ORACLE and self.state.total_steps < 3:
            print(f"  [ORACLE DEBUG step {self.state.total_steps}] node={getattr(observation, 'current_node', '?')}")
            print(f"    Neighbors: {getattr(observation, 'neighbor_ids', 'N/A')}")
            print(f"    Raw response: {(response or 'ERROR')[:200]}")
            print(f"    Parsed action: \"{action_str}\"")

        # Update context buffer
        obs_summary = self._summarize_observation(observation)
        entry = ContextEntry(
            step=self.state.total_steps,
            observation_summary=obs_summary,
            action_taken=action_str,
            reasoning=reasoning,
        )
        self.state.context_buffer.append(entry)

        # Track for analysis
        if self.graph_mode:
            self.state.trajectory.append(getattr(observation, 'current_pos', (0, 0)))
            if getattr(observation, 'door_signals', []):
                nearby = getattr(observation, 'nearby_doors', [])
                if nearby:
                    self.state.signals_seen.append({
                        "step": self.state.total_steps,
                        "doors": nearby,
                        "signals": observation.door_signals,
                    })
        else:
            self.state.trajectory.append(observation.agent_pos)
            if observation.adjacent_door is not None:
                self.state.signals_seen.append({
                    "step": self.state.total_steps,
                    "door_id": observation.adjacent_door.door_id,
                    "door_label": observation.adjacent_door.label,
                    "signals": observation.door_signals,
                })

        # Compress if buffer is getting full
        if len(self.state.context_buffer) >= self.config.max_context_entries:
            self._compress_context()

        self.state.interaction_count += 1
        self.state.total_steps += 1

        return action_str

    def _call_llm(self, messages: list[dict], max_tokens: int = 300) -> str:
        """Call the LLM and return the response text."""
        # Separate system message from user messages for Anthropic API
        system = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        result = self.llm.messages.create(
            model=self.config.model_name,
            max_tokens=max_tokens,
            system=system,
            messages=api_messages,
        )
        return result.content[0].text

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response into action string and reasoning."""
        reasoning = ""

        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:200]  # cap length

        # Extract action
        action_match = re.search(r"ACTION:\s*(.+)", response, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()
            if self.graph_mode:
                # Return raw text for graph mode — simulator uses world.parse_action()
                return action_text, reasoning
            else:
                # Map to canonical grid action string
                action_word = action_text.split()[0].lower() if action_text else "stay"
                action_map = {
                    "north": "north", "south": "south",
                    "east": "east", "west": "west",
                    "stay": "stay", "up": "north",
                    "down": "south", "left": "west",
                    "right": "east",
                }
                return action_map.get(action_word, "stay"), reasoning

        return "stay", reasoning

    def _summarize_observation(self, obs) -> str:
        """Short summary of observation for context buffer."""
        if self.graph_mode:
            parts = [f"node={getattr(obs, 'current_node', '?')}"]
            if getattr(obs, 'current_is_door', False):
                parts.append(f"at {obs.current_door_label}")
            nearby = getattr(obs, 'nearby_doors', [])
            if nearby:
                parts.append(f"near {nearby[0]['label']}")
            signals = getattr(obs, 'door_signals', [])
            if signals:
                parts.append(f'signal: "{signals[0][:60]}"')
            return "; ".join(parts)
        else:
            parts = [f"pos=({obs.agent_pos[0]},{obs.agent_pos[1]})"]
            if obs.adjacent_door:
                parts.append(f"near {obs.adjacent_door.label}")
                if obs.door_signals:
                    parts.append(f'signal: "{obs.door_signals[0][:60]}"')
            return "; ".join(parts)

    def _compress_context(self):
        """
        Compress the oldest half of the context buffer into a summary.
        This is the key memory management mechanism.
        """
        if self.state.agent_type == AgentType.RANDOM_WALK:
            # Just truncate for random walk
            self.state.context_buffer = self.state.context_buffer[-5:]
            return

        half = len(self.state.context_buffer) // 2
        old_entries = self.state.context_buffer[:half]
        old_text = "\n".join(e.to_text() for e in old_entries)

        messages = [
            {"role": "system", "content": (
                "You are a memory compression system. Summarize the following exploration "
                "history into 2-3 key observations. Focus on: which doors you visited, what "
                "signals seemed reliable vs misleading, and any patterns about the goal location. "
                "Be extremely concise."
            )},
            {"role": "user", "content": f"Compress this history:\n{old_text}"}
        ]

        try:
            summary = self._call_llm(messages, max_tokens=200)
            summary_entry = ContextEntry(
                step=old_entries[0].step,
                observation_summary=f"[COMPRESSED MEMORY] {summary[:300]}",
                action_taken="n/a",
                reasoning="compressed",
            )
            self.state.context_buffer = [summary_entry] + self.state.context_buffer[half:]
        except Exception:
            # On failure, just truncate
            self.state.context_buffer = self.state.context_buffer[half:]

    def generate_prior(self) -> str:
        """
        Generate a compressed prior from the agent's experience.
        Called at reproduction time (every N interactions).

        This is the core mechanism: the agent distills its accumulated
        knowledge into a short natural language summary that gets
        passed to its offspring.
        """
        if self.state.agent_type == AgentType.RANDOM_WALK:
            return ""

        context_text = self._build_context_text()
        current_prior = self.state.prior or "None"

        messages = [
            {"role": "system", "content": (
                "You are distilling exploration knowledge into a brief guide for the next explorer.\n"
                "Based on the exploration history below, write a CONCISE set of rules/heuristics "
                "(max 5 bullet points) about:\n"
                "1. Where the goal door likely is (direction, region, row/column)\n"
                "2. Which door signals to TRUST and which to IGNORE\n"
                "3. What navigation strategy works best\n"
                "4. Any other critical knowledge\n\n"
                "Be specific and actionable. The next explorer will use this as their starting knowledge."
            )},
            {"role": "user", "content": (
                f"Previous inherited knowledge: {current_prior}\n\n"
                f"Your exploration history:\n{context_text}\n\n"
                f"Signals you encountered:\n{json.dumps(self.state.signals_seen[-10:], indent=1)}\n\n"
                "Distill your knowledge now:"
            )}
        ]

        try:
            prior = self._call_llm(messages)
            return prior[:500]  # cap prior length
        except Exception as e:
            return f"[Prior generation failed: {str(e)[:50]}]"

    def should_reproduce(self) -> bool:
        """Check if agent has hit the reproduction threshold."""
        return (
            self.state.interaction_count >= self.config.interactions_per_lifetime
            and self.state.agent_type == AgentType.EXPERIMENTAL
            and self.state.generation < self.config.max_generations
        )


def generate_random_prior(rng: random.Random) -> str:
    """Generate a random/gibberish prior for the random-prior baseline."""
    fragments = [
        "The goal is probably near the center.",
        "Ignore all color-based signals.",
        "Trust directional hints pointing east.",
        "Doors on the perimeter are never the goal.",
        "Move in circles to gather more information.",
        "The blue door is always misleading.",
        "Row 3 is important for some reason.",
        "Stay away from walls - the goal is in open space.",
        "The first signal you see at any door is always wrong.",
        "Navigate diagonally for best results.",
    ]
    k = rng.randint(2, 4)
    selected = rng.sample(fragments, k)
    return "Previous explorer's notes:\n" + "\n".join(f"- {s}" for s in selected)
