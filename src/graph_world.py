"""
Graph-based POMDP environment using Random Geometric Graphs (RGG).

Instead of a rectangular grid, the environment is a random geometric graph:
  - N nodes placed uniformly at random in a unit square
  - Edges connect nodes within distance r (the connection radius)
  - Some nodes are designated as doors (with hints/distractors)
  - One door is the goal
  - Agent traverses edges between nodes

This connects to network theory: graph topology (degree distribution,
clustering, path lengths) directly affects how hard the task is and
how information propagates through agent generations.

Formally:
    S = graph nodes × door states
    A = {move_to_neighbor_0, move_to_neighbor_1, ..., stay}
    T = deterministic traversal along edges
    R = +1 at goal door node, 0 otherwise
    Ω = local neighborhood in graph (k-hop) + subset of door signals
    O = observation function based on graph adjacency
"""
import random
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    text: str
    is_hint: bool  # True = real hint, False = distractor (agent never sees this)
    signal_id: str = ""


@dataclass
class GraphNode:
    node_id: int
    x: float  # position in unit square
    y: float
    neighbors: list[int] = field(default_factory=list)  # adjacent node ids
    is_door: bool = False
    is_goal: bool = False
    door_label: str = ""
    signals: list[Signal] = field(default_factory=list)

    def get_signal_sample(self, max_signals: int, rng: random.Random) -> list[str]:
        k = min(max_signals, len(self.signals))
        if k == 0:
            return []
        sampled = rng.sample(self.signals, k)
        return [s.text for s in sampled]


@dataclass
class GraphObservation:
    """What the agent perceives at a given step."""
    current_node: int
    current_pos: tuple[float, float]
    neighbor_ids: list[int]
    neighbor_labels: list[str]  # e.g., "node 5", "blue door", "node 12"
    nearby_doors: list[dict]  # doors within observation range
    door_signals: list[str]  # signals from current or adjacent door
    current_is_door: bool
    current_door_label: str
    step_number: int = 0
    graph_degree: int = 0  # how many connections this node has

    def to_text(self) -> str:
        """Convert observation to natural language for the LLM."""
        lines = []
        lines.append(f"Step {self.step_number}. You are at node {self.current_node} "
                      f"(position: {self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}).")

        if self.current_is_door:
            lines.append(f"This node is a door: {self.current_door_label}.")

        lines.append(f"This node has {self.graph_degree} connections.")
        lines.append(f"You can move to: {', '.join(self.neighbor_labels)}.")

        # Nearby doors
        if self.nearby_doors:
            door_strs = [f"{d['label']} (node {d['node_id']})" for d in self.nearby_doors]
            lines.append(f"Doors nearby: {', '.join(door_strs)}.")

        # Signals
        if self.door_signals:
            lines.append("Signals you can observe:")
            for i, sig in enumerate(self.door_signals, 1):
                lines.append(f"  Signal {i}: \"{sig}\"")

        # Spatial hints
        if self.current_pos[1] > 0.5:
            lines.append("You are in the upper region of the space.")
        else:
            lines.append("You are in the lower region of the space.")
        if self.current_pos[0] > 0.5:
            lines.append("You are in the right region of the space.")
        else:
            lines.append("You are in the left region of the space.")

        return "\n".join(lines)


class GraphWorld:
    """
    POMDP environment over a Random Geometric Graph.
    
    Parameters:
        num_nodes: total nodes in the graph
        connection_radius: nodes within this distance are connected
        num_doors: how many nodes become doors
        observation_hops: how far the agent can "see" in the graph (1 = immediate neighbors)
    """

    def __init__(self, config, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random(config.random_seed)

        self.num_nodes = getattr(config, 'num_nodes', 20)
        self.connection_radius = getattr(config, 'connection_radius', 0.35)
        self.observation_hops = getattr(config, 'observation_hops', 1)

        self.nodes: dict[int, GraphNode] = {}
        self.door_nodes: list[int] = []
        self.goal_node: Optional[int] = None
        self.agent_node: int = 0
        self.step_count: int = 0
        self.done: bool = False
        self.success: bool = False

        self._build_graph()

    def _build_graph(self):
        """Generate the random geometric graph."""
        # Place nodes uniformly in unit square
        for i in range(self.num_nodes):
            x = self.rng.uniform(0, 1)
            y = self.rng.uniform(0, 1)
            self.nodes[i] = GraphNode(node_id=i, x=x, y=y)

        # Connect nodes within radius
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dist = math.sqrt(
                    (self.nodes[i].x - self.nodes[j].x) ** 2 +
                    (self.nodes[i].y - self.nodes[j].y) ** 2
                )
                if dist <= self.connection_radius:
                    self.nodes[i].neighbors.append(j)
                    self.nodes[j].neighbors.append(i)

        # Ensure graph is connected — add edges to isolated components
        self._ensure_connected()

        # Designate doors — pick nodes with moderate degree (not leaves, not hubs)
        # This makes them interesting to find
        candidates = sorted(
            self.nodes.values(),
            key=lambda n: abs(len(n.neighbors) - self._avg_degree())
        )
        num_doors = min(self.config.num_doors, len(candidates))
        door_labels = ["red door", "blue door", "green door", "yellow door",
                       "star door", "circle door", "triangle door", "square door"]

        for i in range(num_doors):
            node = candidates[i]
            node.is_door = True
            node.door_label = door_labels[i % len(door_labels)]
            self.door_nodes.append(node.node_id)
            if i == 0:
                node.is_goal = True
                self.goal_node = node.node_id

        # Place agent at a random non-door node
        non_door = [n for n in self.nodes if n not in self.door_nodes]
        if non_door:
            self.agent_node = self.rng.choice(non_door)
        else:
            self.agent_node = 0

    def _avg_degree(self) -> float:
        if not self.nodes:
            return 0
        return sum(len(n.neighbors) for n in self.nodes.values()) / len(self.nodes)

    def _ensure_connected(self):
        """Ensure the graph is connected by adding edges between components."""
        visited = set()
        components = []

        for start in self.nodes:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in component:
                    continue
                component.add(node)
                visited.add(node)
                for neighbor in self.nodes[node].neighbors:
                    if neighbor not in component:
                        queue.append(neighbor)
            components.append(component)

        # Connect components by adding edges between closest nodes
        while len(components) > 1:
            c1 = components[0]
            best_dist = float('inf')
            best_pair = None
            best_comp_idx = 1

            for idx in range(1, len(components)):
                c2 = components[idx]
                for n1 in c1:
                    for n2 in c2:
                        dist = math.sqrt(
                            (self.nodes[n1].x - self.nodes[n2].x) ** 2 +
                            (self.nodes[n1].y - self.nodes[n2].y) ** 2
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (n1, n2)
                            best_comp_idx = idx

            if best_pair:
                n1, n2 = best_pair
                self.nodes[n1].neighbors.append(n2)
                self.nodes[n2].neighbors.append(n1)
                # Merge components
                components[0] = components[0] | components[best_comp_idx]
                components.pop(best_comp_idx)

    def set_signals(self, door_signals: dict[int, list[Signal]]):
        """Attach signals to door nodes. Keyed by door index (0, 1, 2, ...) not node id."""
        for i, node_id in enumerate(self.door_nodes):
            if i in door_signals:
                self.nodes[node_id].signals = door_signals[i]

    def get_observation(self) -> GraphObservation:
        """Return what the agent perceives from current node."""
        node = self.nodes[self.agent_node]

        # Get neighbors
        neighbor_labels = []
        for nid in node.neighbors:
            n = self.nodes[nid]
            if n.is_door:
                neighbor_labels.append(f"{n.door_label} (node {nid})")
            else:
                neighbor_labels.append(f"node {nid}")

        # Doors within observation range (k-hop neighborhood)
        nearby_doors = []
        reachable = self._k_hop_neighbors(self.agent_node, self.observation_hops)
        for rid in reachable:
            rnode = self.nodes[rid]
            if rnode.is_door:
                nearby_doors.append({
                    "node_id": rid,
                    "label": rnode.door_label,
                })

        # Signals: if current node or immediate neighbor is a door, sample signals
        door_signals = []
        if node.is_door:
            door_signals = node.get_signal_sample(self.config.max_signals_per_observation, self.rng)
        else:
            # Check immediate neighbors for doors
            for nid in node.neighbors:
                n = self.nodes[nid]
                if n.is_door:
                    door_signals = n.get_signal_sample(self.config.max_signals_per_observation, self.rng)
                    break

        return GraphObservation(
            current_node=self.agent_node,
            current_pos=(node.x, node.y),
            neighbor_ids=list(node.neighbors),
            neighbor_labels=neighbor_labels,
            nearby_doors=nearby_doors,
            door_signals=door_signals,
            current_is_door=node.is_door,
            current_door_label=node.door_label if node.is_door else "",
            step_number=self.step_count,
            graph_degree=len(node.neighbors),
        )

    def _k_hop_neighbors(self, start: int, k: int) -> set[int]:
        """Get all nodes within k hops of start."""
        visited = {start}
        frontier = {start}
        for _ in range(k):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.nodes[node].neighbors:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
        return visited - {start}

    def step(self, action: int) -> tuple[GraphObservation, float, bool]:
        """
        Execute action (move to a neighbor by index, or -1 to stay).
        Returns (observation, reward, done).
        """
        if self.done:
            return self.get_observation(), 0.0, True

        self.step_count += 1
        node = self.nodes[self.agent_node]

        # Move to neighbor
        if action >= 0 and action < len(node.neighbors):
            self.agent_node = node.neighbors[action]
        # else: stay (invalid action index or -1)

        # Check goal
        reward = 0.0
        if self.agent_node == self.goal_node:
            reward = 1.0
            self.done = True
            self.success = True

        # Step limit
        max_steps = getattr(self.config, 'max_steps_per_trial', 300)
        if self.step_count >= max_steps:
            self.done = True

        return self.get_observation(), reward, self.done

    def parse_action(self, action_text: str, current_neighbors: list[int]) -> int:
        """
        Parse LLM's text action into a neighbor index.
        Accepts: "node 5", "move to node 5", "5", "blue door", "stay", neighbor index, etc.
        """
        action_text = action_text.lower().strip()

        if action_text in ("stay", "wait", "-1"):
            return -1

        # Try to extract a node id (e.g. "node 5", "move to node 5")
        import re
        node_match = re.search(r'node\s*(\d+)', action_text)
        if node_match:
            target = int(node_match.group(1))
            if target in current_neighbors:
                return current_neighbors.index(target)
            # Explicit "node X" but X isn't a neighbor — don't misinterpret as index
            return -1

        # Try door label match (e.g. "red door", "move to blue door")
        for i, nid in enumerate(current_neighbors):
            n = self.nodes[nid]
            if n.is_door and n.door_label.lower() in action_text:
                return i

        # Try plain number (only when no "node" keyword was used)
        num_match = re.search(r'(\d+)', action_text)
        if num_match:
            num = int(num_match.group(1))
            if num in current_neighbors:
                return current_neighbors.index(num)

        return -1  # stay

    def get_graph_stats(self) -> dict:
        """Compute graph topology metrics for analysis."""
        degrees = [len(n.neighbors) for n in self.nodes.values()]
        clustering = self._avg_clustering()

        # Shortest path from agent start to goal
        goal_dist = self._shortest_path_length(self.agent_node, self.goal_node) if self.goal_node is not None else -1

        return {
            "num_nodes": self.num_nodes,
            "num_edges": sum(degrees) // 2,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "avg_clustering": clustering,
            "connection_radius": self.connection_radius,
            "goal_shortest_path": goal_dist,
            "num_doors": len(self.door_nodes),
        }

    def _shortest_path_length(self, start: int, end: int) -> int:
        """BFS shortest path."""
        if start == end:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            for neighbor in self.nodes[node].neighbors:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1  # unreachable

    def _avg_clustering(self) -> float:
        """Average clustering coefficient."""
        coefficients = []
        for node in self.nodes.values():
            neighbors = node.neighbors
            if len(neighbors) < 2:
                coefficients.append(0.0)
                continue
            # Count edges between neighbors
            neighbor_set = set(neighbors)
            edges = 0
            for n in neighbors:
                for nn in self.nodes[n].neighbors:
                    if nn in neighbor_set and nn != node.node_id:
                        edges += 1
            edges //= 2  # undirected
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            coefficients.append(edges / possible if possible > 0 else 0)
        return sum(coefficients) / len(coefficients) if coefficients else 0

    def get_description(self) -> str:
        """Full description for oracle agents or analysis."""
        goal = self.nodes[self.goal_node] if self.goal_node else None
        stats = self.get_graph_stats()
        desc = (
            f"Random Geometric Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges\n"
            f"Connection radius: {stats['connection_radius']:.2f}, Avg degree: {stats['avg_degree']:.1f}\n"
            f"Clustering coefficient: {stats['avg_clustering']:.3f}\n"
        )
        if goal:
            desc += f"Goal: {goal.door_label} at node {goal.node_id} (pos: {goal.x:.2f}, {goal.y:.2f})\n"
            desc += f"Shortest path from start to goal: {stats['goal_shortest_path']} hops\n"
        desc += "Doors:\n"
        for nid in self.door_nodes:
            n = self.nodes[nid]
            desc += f"  {n.door_label} at node {nid} (pos: {n.x:.2f}, {n.y:.2f}) {'[GOAL]' if n.is_goal else ''}\n"
        return desc

    def render_ascii(self) -> str:
        """Simple text representation of the graph."""
        lines = [f"Graph: {self.num_nodes} nodes, agent at node {self.agent_node}"]
        for nid in sorted(self.nodes.keys()):
            n = self.nodes[nid]
            marker = ""
            if nid == self.agent_node:
                marker = " [AGENT]"
            if n.is_goal:
                marker += " [GOAL]"
            elif n.is_door:
                marker += f" [{n.door_label}]"
            lines.append(
                f"  Node {nid}: ({n.x:.2f}, {n.y:.2f}) "
                f"deg={len(n.neighbors)} neighbors={n.neighbors}{marker}"
            )
        return "\n".join(lines)
