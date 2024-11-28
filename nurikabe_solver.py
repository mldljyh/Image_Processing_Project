import numpy as np

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

class State:
    """Constants representing cell states in the Nurikabe puzzle.
    
    Attributes:
        UNKNOWN: Unsolved white cell (-2)
        SEA: Black cell in solution (-1)
        ISLAND: White cell in solution (0)
        Values > 0 represent numbered island cells
    """
    UNKNOWN = -2
    SEA = -1  
    ISLAND = 0

@dataclass
class Position:
    """Represents a position on the puzzle grid."""
    y: int
    x: int

class Solver:
    """Solves Nurikabe puzzles using logical deduction.
    
    Attributes:
        puzzle: The puzzle grid as a numpy array
        height: Height of the puzzle
        width: Width of the puzzle
        sea_size: Total number of cells that should be sea
        solved: Whether puzzle is solved
        islands: Dictionary mapping island centers to their cells
        seas: Dictionary mapping sea centers to their cells
    """
    def __init__(self, puzzle: np.ndarray):
        self.puzzle = puzzle
        self.height, self.width = puzzle.shape
        self.sea_size = self.width * self.height - np.sum(self.puzzle[self.puzzle > 0])
        self.solved = False
        self.islands: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.seas: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def solve_logic_initial(self) -> None:
        """Performs initial logical deductions and prepares puzzle data.
        
        This includes:
        - Identifying island centers
        - Marking impossible diagonal neighbors as sea
        - Marking cells between islands as sea
        - Marking neighbors of size-1 islands as sea
        """
        for x in range(self.width):
            for y in range(self.height):
                # Compose islands
                if self.puzzle[y, x] > 1:
                    self.islands[y, x] = [(y, x)]

                # Impossible diagonal neighbours (Sea)
                if 0 < x < self.width - 1 and 0 < y < self.height - 1 and self.puzzle[y, x] > 0:
                    if self.puzzle[y - 1, x - 1] > 0:
                        self.set_cell(y - 1, x, State.SEA)
                        self.set_cell(y, x - 1, State.SEA)
                    elif self.puzzle[y + 1, x - 1] > 0:
                        self.set_cell(y + 1, x, State.SEA)
                        self.set_cell(y, x - 1, State.SEA)

                # Sea between horizontal/vertical island centers (Sea)
                if x < self.width - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y, x + 2] > 0:
                        self.set_cell(y, x + 1, State.SEA)
                if y < self.height - 2:
                    if self.puzzle[y, x] > 0 and self.puzzle[y + 2, x] > 0:
                        self.set_cell(y + 1, x, State.SEA)

                # Neighbours of '1' island (Sea)
                if self.puzzle[y, x] == 1:
                    self.four_way(y, x, State.SEA)

    def set_cell(self, y: int, x: int, state: int, center: Optional[Tuple[int, int]] = None) -> None:
        """Sets a cell's state and updates tracking collections.
        
        Args:
            y: Y coordinate
            x: X coordinate
            state: New state for the cell
            center: Optional center coordinates for tracking islands/seas
            
        Raises:
            AssertionError: If attempting to change an island center
        """
        assert self.puzzle[y, x] <= 0, f"Cannot change island center at ({y}, {x})"

        if self.puzzle[y, x] != state:
            self.puzzle[y, x] = state

        # Tracking
        if state == State.SEA:
            if self.seas.get(center, None) is None:
                self.seas[center] = [(y, x)]
            elif (y, x) not in self.seas[center]:
                self.seas[center].append((y, x))
        elif state == State.ISLAND:
            if self.islands.get(center, None) is None:
                self.islands[y, x] = [(y, x)]
            elif (y, x) not in self.islands[center]:
                self.islands[center].append((y, x))

    def four_way(self, y, x, state=None, func=None, check_state=State.UNKNOWN):
        """Perform four-way operation."""
        directions = [
            (y-1, x), (y+1, x), (y, x-1), (y, x+1)
        ]
        
        for ny, nx in directions:
            if 0 <= ny < self.height and 0 <= nx < self.width and min(self.puzzle[ny, nx], 0) == check_state:
                if func:
                    func(ny, nx)
                if state:
                    self.set_cell(ny, nx, state, center=(ny, nx))

    def extension_ways(self, cells, check_state=State.UNKNOWN):
        """Finds ways cells can extend in."""
        ways = []
        for (cy, cx) in cells:
            self.four_way(cy, cx, None, lambda ny, nx: ways.append((ny, nx)), check_state=check_state)
        return ways

    def validate(self):
        """Validate the solution according to Nurikabe rules."""
        single_sea = len(self.seas) == 1
        full_sea = np.sum(self.puzzle == State.SEA) == self.sea_size
        no_unknowns = np.sum(self.puzzle == State.UNKNOWN) == 0

        # Additional validation: Check the size and independence of each island
        islands_valid = self._validate_islands()

        return single_sea and full_sea and no_unknowns and islands_valid

    def _validate_islands(self):
        """Check if each island has the correct size and is not connected to other islands."""
        visited = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.puzzle[y, x] > 0 and (y, x) not in visited:
                    expected_size = self.puzzle[y, x]
                    island_cells = self._collect_island_cells(y, x, visited)
                    if len(island_cells) != expected_size:
                        return False
        return True

    def _collect_island_cells(self, y, x, visited):
        """Collect all cells belonging to an island starting from (y, x)."""
        stack = [(y, x)]
        island_cells = []
        while stack:
            cy, cx = stack.pop()
            if (cy, cx) in visited:
                continue
            if self.puzzle[cy, cx] == State.ISLAND or self.puzzle[cy, cx] > 0:
                visited.add((cy, cx))
                island_cells.append((cy, cx))
                # Add adjacent cells
                for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if (self.puzzle[ny, nx] == State.ISLAND or self.puzzle[ny, nx] > 0) and (ny, nx) not in visited:
                            stack.append((ny, nx))
        return island_cells

    def _can_extend_to(self, center, cell):
        """Check if an island can extend to a cell without connecting to other islands."""
        y, x = cell
        if self.puzzle[y, x] > 0:  # Can't extend to numbered cells
            return False
            
        # Check neighboring cells for other islands
        for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if self.puzzle[ny, nx] > 0 and (ny, nx) != center:
                    return False
                # Check if cell belongs to another island
                for other_center, other_cells in self.islands.items():
                    if other_center != center and (ny, nx) in other_cells:
                        return False
        return True

    def _extend_islands(self):
        extended = 0
        # Sort islands by remaining cells needed (prioritize islands that need more cells)
        sorted_islands = sorted(
            [(center, cells) for center, cells in self.islands.items()],
            key=lambda x: self.puzzle[x[0]] - len(x[1]),
            reverse=True
        )
        
        for center, cells in sorted_islands:
            target_size = self.puzzle[center] if self.puzzle[center] > 0 else 0
            left = target_size - len(cells)
            
            if left > 0:
                ways = self.extension_ways(cells)
                valid_ways = [way for way in ways if self._can_extend_to(center, way)]
                
                if len(valid_ways) == 1:
                    ny, nx = valid_ways[0]
                    self.set_cell(ny, nx, State.ISLAND, center=center)
                    extended += 1
                elif len(valid_ways) > 0 and left == len(valid_ways):
                    # If remaining cells needed equals available valid ways, use all of them
                    for ny, nx in valid_ways:
                        self.set_cell(ny, nx, State.ISLAND, center=center)
                        extended += 1
        return extended

    def _wrap_full_islands(self):
        wrapped = 0
        for center, cells in list(self.islands.items()):
            if len(cells) == self.puzzle[center]:
                for (cy, cx) in cells:
                    self.four_way(cy, cx, State.SEA)
                del self.islands[center]
                wrapped += 1
        return wrapped

    def _prevent_pools(self):
        prevented = 0
        for x in range(self.width - 1):
            for y in range(self.height - 1):
                pool = [
                    (y, x), (y, x+1), (y+1, x), (y+1, x+1)
                ]
                pool_states = [self.puzzle[cell] for cell in pool]
                
                if pool_states.count(State.SEA) == 3 and State.UNKNOWN in pool_states:
                    unknown_idx = pool_states.index(State.UNKNOWN)
                    ny, nx = pool[unknown_idx]
                    self.set_cell(ny, nx, State.ISLAND)
                    prevented += 1
        return prevented

    def _merge_patches(self):
        merged = 0
        
        # Merge island patches
        for center, cells in list(self.islands.items()):
            ways = self.extension_ways(cells, check_state=State.ISLAND)
            ways = [way for way in ways if way not in cells]
            
            if ways:
                ny, nx = ways[0]
                ncenter = next((ncenter for ncenter, ncells in self.islands.items() if (ny, nx) in ncells), None)
                
                if ncenter and ncenter != center:
                    self.islands[ncenter].extend(cells)
                    del self.islands[center]
                    merged += 1
        
        # Merge sea patches
        for center, cells in list(self.seas.items()):
            ways = self.extension_ways(cells, check_state=State.SEA)
            ways = [way for way in ways if way not in cells]
            
            if ways:
                ny, nx = ways[0]
                ncenter = next((ncenter for ncenter, ncells in self.seas.items() if (ny, nx) in ncells), None)
                
                if ncenter and ncenter != center:
                    self.seas[ncenter].extend(cells)
                    del self.seas[center]
                    merged += 1
        
        return merged

    def _clean_isolated_zeros(self):
        """Remove isolated zeros that are completely surrounded by sea."""
        cleaned = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.puzzle[y, x] == State.ISLAND:  # Check zeros
                    surrounded_by_sea = True
                    # Check all four directions
                    for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                        if (0 <= ny < self.height and 0 <= nx < self.width and 
                            self.puzzle[ny, nx] != State.SEA):
                            surrounded_by_sea = False
                            break
                    
                    if surrounded_by_sea:
                        self.puzzle[y, x] = State.SEA
                        cleaned += 1
        return cleaned

    def solve_logic(self):
        """Apply logical solving techniques."""
        operations = 0
        
        # Prioritize extending islands
        while True:
            extended = self._extend_islands()
            if extended == 0:
                break
            operations += extended
        
        # Continue with other solving techniques
        operations += self._wrap_full_islands()
        operations += self._prevent_pools()
        operations += self._merge_patches()
        operations += self._clean_isolated_zeros()  # Add cleaning step
        
        return operations

    def _validate_input_constraints(self):
        """
        Check if each numbered cell's constraints can be satisfied by counting available cells.
        Returns False if any constraint is impossible to satisfy.
        """
        for y in range(self.height):
            for x in range(self.width):
                if self.puzzle[y, x] > 0:  # For each numbered cell
                    required_size = self.puzzle[y, x]
                    available_cells = self._count_available_cells(y, x)
                    if available_cells < required_size:
                        print(f"Island at ({y}, {x}) requires {required_size} cells but only {available_cells} available")
                        return False
        return True

    def _count_available_cells(self, start_y, start_x):
        """
        Count the number of cells that could be part of an island starting at (start_y, start_x).
        Stops counting when encountering SEA (-1) cells.
        """
        visited = set()
        queue = [(start_y, start_x)]
        count = 0

        while queue:
            y, x = queue.pop(0)
            if (y, x) in visited:
                continue

            visited.add((y, x))
            if self.puzzle[y, x] == 0 or count == 0:  # Count UNKNOWN (-2) and non-negative cells
                count += 1
                # Check adjacent cells
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    if (0 <= ny < self.height and 0 <= nx < self.width and 
                        (ny, nx) not in visited and 
                        self.puzzle[ny, nx] != State.SEA):
                        queue.append((ny, nx))

        return count

    def solve(self):
        """Main solving method."""
        self.solve_logic_initial()
        
        operations = self.solve_logic()
        if not self._validate_input_constraints():
            print("No solution exists: Island size constraints cannot be satisfied")
            self.solved = False
        else:
            self.solved = True

        return self.solved

def solve_nurikabe_puzzle(input_data):
    """
    Solve a Nurikabe puzzle given input data.
    
    :param input_data: List of tuples (number, x, y)
    :return: Solved puzzle as numpy array
    """
    # Determine puzzle size
    max_x = max(x for _, x, _ in input_data)
    max_y = max(y for _, _, y in input_data)
    puzzle_size = max(max_x, max_y) + 1
    
    # Initialize puzzle
    puzzle = np.full((puzzle_size, puzzle_size), State.UNKNOWN, dtype=np.int8)
    
    # Place initial numbers
    for number, x, y in input_data:
        puzzle[y, x] = number
    
    # Solve puzzle
    solver = Solver(puzzle)
    success = solver.solve()
    
    return solver.puzzle, success

def main():
    # Example input
    input_data = [
        (2, 0, 0),
        (3, 2, 0),
        (4, 6, 1),
        (1, 4, 2),
        (4, 5, 3),
        (2, 2, 5),
        (2, 4, 5),
        (5, 1, 6),
    ]
    
    # Solve puzzle
    solution, success = solve_nurikabe_puzzle(input_data)
    
    # Print solution
    print("Nurikabe Puzzle Solution:")
    print(solution)
    print(f"Solve Status: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()