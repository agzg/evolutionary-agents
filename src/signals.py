"""
Hint and distractor signal generation for doors.

Works with both grid positions and graph node positions.
Each door gets a mix of true hints (pointing toward the goal) and 
distractors (misleading signals). Agents see these but never the labels.
"""
import random
import math
from dataclasses import dataclass


@dataclass 
class Signal:
    text: str
    is_hint: bool
    signal_id: str = ""


@dataclass
class DoorInfo:
    """Unified door representation for signal generation."""
    door_id: int
    x: float  # normalized position [0, 1]
    y: float
    label: str
    is_goal: bool


def _region(x: float, y: float) -> str:
    v = "upper" if y > 0.5 else "lower"
    h = "right" if x > 0.5 else "left"
    return f"{v}-{h}"


def _direction_from(src_x, src_y, dst_x, dst_y) -> str:
    dx = dst_x - src_x
    dy = dst_y - src_y
    parts = []
    if dy > 0.1:
        parts.append("upward")
    elif dy < -0.1:
        parts.append("downward")
    if dx > 0.1:
        parts.append("to the right")
    elif dx < -0.1:
        parts.append("to the left")
    return " and ".join(parts) if parts else "very nearby"


def generate_hints(door, goal, all_doors, num_hints, rng):
    pool = []
    region = _region(goal.x, goal.y)
    pool.append(f"The goal door is in the {region} region of the space.")
    if goal.y > 0.5:
        pool.append("The goal door is in the upper half of the space.")
    else:
        pool.append("The goal door is in the lower half of the space.")
    if goal.x > 0.5:
        pool.append("The goal door is in the right half of the space.")
    else:
        pool.append("The goal door is in the left half of the space.")
    pool.append(f"The goal door is the {goal.label}.")
    for d in all_doors:
        if not d.is_goal:
            pool.append(f"The goal is NOT the {d.label}.")
    direction = _direction_from(door.x, door.y, goal.x, goal.y)
    pool.append(f"From here, the goal is {direction}.")
    dist = math.sqrt((goal.x - door.x)**2 + (goal.y - door.y)**2)
    if door.is_goal:
        pool.append("You are very close to the goal right now.")
        pool.append("This door might be what you're looking for.")
    elif dist < 0.25:
        pool.append("The goal is nearby this location.")
    else:
        pool.append("The goal is far from this door.")
    pool.append(f"The goal door is at approximately x={goal.x:.1f}.")
    pool.append(f"The goal door is at approximately y={goal.y:.1f}.")

    rng.shuffle(pool)
    hints = []
    for i, text in enumerate(pool[:num_hints]):
        hints.append(Signal(text=text, is_hint=True, signal_id=f"h_{door.door_id}_{i}"))
    return hints


def generate_distractors(door, goal, all_doors, num_distractors, rng):
    pool = []
    if goal.y > 0.5:
        pool.append("The goal door is in the lower half of the space.")
    else:
        pool.append("The goal door is in the upper half of the space.")
    if goal.x > 0.5:
        pool.append("The goal door is in the left half of the space.")
    else:
        pool.append("The goal door is in the right half of the space.")
    for ng in all_doors:
        if not ng.is_goal:
            pool.append(f"The goal door is the {ng.label}.")
    pool.append(f"The goal is NOT the {goal.label}.")
    if door.is_goal:
        pool.append("The goal is far from here. Keep searching.")
    else:
        pool.append("You are very close to the goal right now.")
        pool.append("This door might be what you're looking for.")
    pool.append(f"The goal door is at approximately x={rng.uniform(0, 1):.1f}.")
    pool.append(f"The goal door is at approximately y={rng.uniform(0, 1):.1f}.")
    pool.append("The goal is behind the door you least expect.")
    pool.append("Trust your instincts about the goal's location.")
    pool.append("High-degree nodes are more likely to be the goal.")

    rng.shuffle(pool)
    distractors = []
    for i, text in enumerate(pool[:num_distractors]):
        distractors.append(Signal(text=text, is_hint=False, signal_id=f"d_{door.door_id}_{i}"))
    return distractors


def generate_all_signals_for_graph(graph_world, hints_per_door, distractors_per_door, rng):
    """Generate signals for all doors in a graph world."""
    door_infos = []
    goal_info = None
    for i, nid in enumerate(graph_world.door_nodes):
        node = graph_world.nodes[nid]
        info = DoorInfo(door_id=i, x=node.x, y=node.y, label=node.door_label, is_goal=node.is_goal)
        door_infos.append(info)
        if node.is_goal:
            goal_info = info

    all_signals = {}
    for info in door_infos:
        hints = generate_hints(info, goal_info, door_infos, hints_per_door, rng)
        dists = generate_distractors(info, goal_info, door_infos, distractors_per_door, rng)
        combined = hints + dists
        rng.shuffle(combined)
        all_signals[info.door_id] = combined
    return all_signals


def generate_all_signals_for_grid(doors, goal_door, grid_size, hints_per_door, distractors_per_door, rng):
    """Generate signals for all doors in a grid world."""
    rows, cols = grid_size
    door_infos = []
    goal_info = None
    for door in doors:
        info = DoorInfo(
            door_id=door.door_id,
            x=door.position[1] / max(cols - 1, 1),
            y=1.0 - door.position[0] / max(rows - 1, 1),
            label=door.label, is_goal=door.is_goal,
        )
        door_infos.append(info)
        if door.is_goal:
            goal_info = info

    all_signals = {}
    for info in door_infos:
        hints = generate_hints(info, goal_info, door_infos, hints_per_door, rng)
        dists = generate_distractors(info, goal_info, door_infos, distractors_per_door, rng)
        combined = hints + dists
        rng.shuffle(combined)
        all_signals[info.door_id] = combined
    return all_signals
