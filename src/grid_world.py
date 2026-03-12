"""
POMDP Grid World with doors containing hints and distractors.

Formally:
    S = grid positions × door states
    A = {north, south, east, west, stay}
    T = deterministic movement (blocked by walls)
    R = +1 at goal door, 0 otherwise
    Ω = partial view of grid + subset of door signals
    O = observation function (local neighborhood + sampled signals)
"""
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Action(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    STAY = "stay"


class CellType(Enum):
    EMPTY = "."
    WALL = "#"
    DOOR = "D"
    AGENT = "A"


@dataclass
class Signal:
    text: str
    is_hint: bool  # True = real hint, False = distractor (agent never sees this label)
    signal_id: str = ""


@dataclass
class Door:
    door_id: int
    position: tuple[int, int]
    signals: list[Signal] = field(default_factory=list)
    is_goal: bool = False
    label: str = ""  # e.g., "red door", "door with star symbol"

    def get_signal_sample(self, max_signals: int, rng: random.Random) -> list[str]:
        """Return a random subset of signal texts (agent doesn't see hint/distractor labels)."""
        k = min(max_signals, len(self.signals))
        sampled = rng.sample(self.signals, k)
        return [s.text for s in sampled]


@dataclass
class Observation:
    """What the agent perceives at a given step."""
    agent_pos: tuple[int, int]
    visible_cells: dict[tuple[int, int], CellType]  # relative positions -> cell types
    adjacent_door: Optional[Door] = None
    door_signals: list[str] = field(default_factory=list)  # signals from adjacent door (if any)
    step_number: int = 0

    def to_text(self) -> str:
        """Convert observation to natural language for the LLM."""
        lines = []
        lines.append(f"Step {self.step_number}. You are at position ({self.agent_pos[0]}, {self.agent_pos[1]}).")

        # Describe visible neighborhood
        directions = {
            (-1, 0): "north", (1, 0): "south",
            (0, -1): "west", (0, 1): "east",
            (-1, -1): "northwest", (-1, 1): "northeast",
            (1, -1): "southwest", (1, 1): "southeast",
        }
        walls = []
        doors = []
        empties = []
        for (dr, dc), cell_type in self.visible_cells.items():
            if (dr, dc) == (0, 0):
                continue
            dir_name = directions.get((dr, dc), f"offset({dr},{dc})")
            if cell_type == CellType.WALL:
                walls.append(dir_name)
            elif cell_type == CellType.DOOR:
                doors.append(dir_name)
            else:
                empties.append(dir_name)

        if walls:
            lines.append(f"Walls: {', '.join(walls)}.")
        if doors:
            lines.append(f"Doors visible: {', '.join(doors)}.")
        if empties:
            lines.append(f"Open paths: {', '.join(empties)}.")

        # Door signals
        if self.adjacent_door is not None:
            lines.append(f"You are next to {self.adjacent_door.label} (door {self.adjacent_door.door_id}).")
            if self.door_signals:
                lines.append("Signals from this door:")
                for i, sig in enumerate(self.door_signals, 1):
                    lines.append(f"  Signal {i}: \"{sig}\"")
            else:
                lines.append("No signals visible from this door right now.")

        return "\n".join(lines)


class GridWorld:
    """
    POMDP grid environment.

    The grid is a 2D array. Doors are placed at specific cells.
    One door is the goal. Agent must navigate to the goal door.
    """

    def __init__(self, config, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random(config.random_seed)
        self.rows, self.cols = config.grid_size
        self.grid: list[list[CellType]] = []
        self.doors: list[Door] = []
        self.goal_door: Optional[Door] = None
        self.agent_pos: tuple[int, int] = (0, 0)
        self.step_count: int = 0
        self.done: bool = False
        self.success: bool = False

        self._build_world()

    def _build_world(self):
        """Generate the grid, place doors and walls."""
        # Initialize empty grid
        self.grid = [[CellType.EMPTY for _ in range(self.cols)] for _ in range(self.rows)]

        # Place doors at random positions (not on edges for simplicity)
        interior_cells = [
            (r, c) for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
        ]
        self.rng.shuffle(interior_cells)

        door_positions = interior_cells[:self.config.num_doors]
        remaining = interior_cells[self.config.num_doors:]

        for i, pos in enumerate(door_positions):
            r, c = pos
            self.grid[r][c] = CellType.DOOR
            door = Door(
                door_id=i,
                position=pos,
                is_goal=(i == 0),  # first door is goal
                label=self._make_door_label(i),
            )
            self.doors.append(door)
            if door.is_goal:
                self.goal_door = door

        # Place walls
        wall_candidates = [p for p in remaining if p not in door_positions]
        num_walls = min(self.config.num_walls, len(wall_candidates))
        wall_positions = wall_candidates[:num_walls]
        for r, c in wall_positions:
            self.grid[r][c] = CellType.WALL

        # Place agent at a random empty cell
        empty_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.grid[r][c] == CellType.EMPTY
        ]
        self.agent_pos = self.rng.choice(empty_cells)

    def _make_door_label(self, door_id: int) -> str:
        labels = ["red door", "blue door", "green door", "yellow door",
                  "star door", "circle door", "triangle door", "square door"]
        return labels[door_id % len(labels)]

    def set_signals(self, door_signals: dict[int, list[Signal]]):
        """Attach generated signals to doors."""
        for door in self.doors:
            if door.door_id in door_signals:
                door.signals = door_signals[door.door_id]

    def get_observation(self) -> Observation:
        """Return what the agent can see from its current position."""
        r, c = self.agent_pos
        radius = self.config.observation_radius
        visible = {}

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    visible[(dr, dc)] = self.grid[nr][nc]
                else:
                    visible[(dr, dc)] = CellType.WALL  # out of bounds = wall

        # Check if adjacent to a door
        adjacent_door = None
        door_signals = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] == CellType.DOOR:
                    for door in self.doors:
                        if door.position == (nr, nc):
                            adjacent_door = door
                            door_signals = door.get_signal_sample(
                                self.config.max_signals_per_observation, self.rng
                            )
                            break
                    break  # only show one adjacent door per step

        return Observation(
            agent_pos=self.agent_pos,
            visible_cells=visible,
            adjacent_door=adjacent_door,
            door_signals=door_signals,
            step_number=self.step_count,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """
        Execute action, return (observation, reward, done).
        Reward: +1 if agent moves onto goal door, 0 otherwise.
        """
        if self.done:
            return self.get_observation(), 0.0, True

        self.step_count += 1
        r, c = self.agent_pos

        # Movement deltas
        deltas = {
            Action.NORTH: (-1, 0),
            Action.SOUTH: (1, 0),
            Action.WEST: (0, -1),
            Action.EAST: (0, 1),
            Action.STAY: (0, 0),
        }
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc

        # Check bounds and walls
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            if self.grid[nr][nc] != CellType.WALL:
                self.agent_pos = (nr, nc)

        # Check if on goal door
        reward = 0.0
        if self.goal_door and self.agent_pos == self.goal_door.position:
            reward = 1.0
            self.done = True
            self.success = True

        # Hard step limit
        if self.step_count >= self.config.max_steps_per_trial:
            self.done = True

        obs = self.get_observation()
        return obs, reward, self.done

    def render(self) -> str:
        """ASCII render of the grid (for debugging and/or visualization)."""
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row.append("A")
                elif self.grid[r][c] == CellType.WALL:
                    row.append("#")
                elif self.grid[r][c] == CellType.DOOR:
                    door = next((d for d in self.doors if d.position == (r, c)), None)
                    if door and door.is_goal:
                        row.append("G")
                    else:
                        row.append("D")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def get_grid_description(self) -> str:
        """Full grid description for oracle agents or analysis."""
        desc = f"Grid: {self.rows}x{self.cols}\n"
        desc += f"Goal door: {self.goal_door.label} at position {self.goal_door.position}\n"
        desc += f"All doors:\n"
        for d in self.doors:
            desc += f"  Door {d.door_id} ({d.label}) at {d.position} {'[GOAL]' if d.is_goal else ''}\n"
        return desc