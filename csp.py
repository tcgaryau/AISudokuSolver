import math
import copy
import itertools
import multiprocessing
import functools
from typing import Set, List


class Square:
    """
    Represent a square on a sudoku board.
    Each square has a row and col number; a flag indicating whether it has been assigned a value;
    a set containing all values that can be assigned as its domain; a set containing all unassigned
    neighbors of the square.
    """

    def __init__(self, row, col, domain, assigned=False):
        self.row = row
        self.col = col
        self.assigned = assigned
        self.domain = domain
        self.neighbors = set()
        self.row_neighbors = set()
        self.col_neighbors = set()
        self.sg_neighbors = set()
        # TODO: add a subgrid store

    def __str__(self):
        return f"Square {self.row} {self.col} domain: {self.domain} " \
               f"length of neighbors: {len(self.neighbors)}"


class Arc:
    """
    Represent an arc between two squares.
    Each arc has contains 4 variables representing the x and y positions of
    two squares.
    """

    def __init__(self, square1_x, square1_y, square2_x, square2_y):
        self.square1_x = square1_x
        self.square1_y = square1_y
        self.square2_x = square2_x
        self.square2_y = square2_y


class CSP:
    """
    Represents a constraint satisfaction problem.
    """
    def __init__(self, puzzle_data):
        """
        Initialize a CSP object.
        :param puzzle_data: a 2D array of integers representing a sudoku puzzle
        """
        self.puzzle_data = copy.deepcopy(puzzle_data)
        self.size_data = len(puzzle_data)
        self.board = None
        self.unassigned: Set[Square] = set()

    def init_board(self):
        """
        Initialize a sudoku board as a 2D array of Squares, and the domain of each square.
        """
        board = [[None for _ in range(self.size_data)] for _ in range(self.size_data)]
        for i, j in itertools.product(range(self.size_data), range(self.size_data)):
            value = self.puzzle_data[i][j]
            if value == 0:
                # domain = list(self._get_consistent_values(i, j))
                domain = list(range(1, self.size_data + 1))
                square = Square(i, j, domain)
                board[i][j] = square
                if len(domain) > 1:
                    self.unassigned.add(square)
            else:
                board[i][j] = Square(i, j, [value], True)
        self.board = board

    def init_arc_constraints(self):
        """
        Generate arc constraints for each square. This is used for the initial AC3 preprocessing
        """
        arcs = set()
        for i, j in itertools.product(range(self.size_data), range(self.size_data)):
            square = self.board[i][j]
            for arc in self._get_arcs(square):
                if arc.square1_x != arc.square2_x and arc.square1_y != arc.square2_y:
                    arcs.add(arc)
        return arcs

    def _get_neighbors(self, square):
        """
        Returns the neighbors of the square
        """
        return square.neighbors

    def _get_arcs(self, square):
        """
        Returns all the arcs from a square. Used with MAC arc generation
        """
        neighbors = self._get_neighbors(square)

        return [
            Arc(neighbor[0], neighbor[1], square.row, square.col)
            for neighbor in neighbors
        ]

    def init_binary_constraints(self):
        """
        Populate the neighbors field of every square on the current board.
        This also adds all the neighbors, row neighbors, col neighbors and
        subgrid neighbors to each square
        """
        board = self.board
        size = self.size_data
        sg_row_total = int(math.sqrt(size))
        sg_col_total = int(math.ceil(math.sqrt(size)))

        for i, j in itertools.product(range(size), range(size)):
            curr_square: Square = board[i][j]
            row = curr_square.row
            col = curr_square.col

            for n in range(size):
                square = board[row][n]
                if n != col:
                    curr_square.neighbors.add((square.row, square.col))
                    curr_square.col_neighbors.add((square.row, square.col))

            for m in range(size):
                square = board[m][col]
                if m != row:
                    curr_square.neighbors.add((square.row, square.col))
                    curr_square.row_neighbors.add((square.row, square.col))

            shift_row = row // sg_row_total * sg_row_total
            shift_col = col // sg_col_total * sg_col_total
            for m, n in itertools.product(range(sg_row_total), range(sg_col_total)):
                square = board[m + shift_row][n + shift_col]
                if m + shift_row != row and n + shift_col != col:
                    curr_square.neighbors.add((square.row, square.col))
                    curr_square.sg_neighbors.add((square.row, square.col))

    def solve_csp_multiprocess(self):
        """
        CSP algorithms. Use multiprocessing for searching.
        """
        # Return true if all squares have been assigned a value
        if len(self.unassigned) == 0:
            self.generate_puzzle_solution()
            return True

        next_empty = self.select_unassigned()
        values = self.find_least_constraining_value(next_empty)
        saved_values = copy.deepcopy(values)

        # Initialize the pool and send the task to each process
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for results in pool.imap_unordered(
                    functools.partial(
                        self.backtrack_mp_helper, next_empty, saved_values=saved_values), values,
                    chunksize=1):
                if results:
                    self.board = results[1]
                    pool.terminate()
                    self.generate_puzzle_solution()
                    return True
        return False

    def backtrack_mp_helper(self, target_cell, v, saved_values):
        """
        Recursive csp algorithm with constraint propagation. This is almost the same function
        as backtrack, but it is a wrapper to set up multiprocessing correctly.
        """
        if self.is_consistent(target_cell, v):
            # Add the value to assignment
            target_cell.domain = [v]
            target_cell.assigned = True
            self.unassigned.remove(target_cell)

            # MAC using ac-3
            is_inference, revised_list = self.mac(target_cell)
            if is_inference and self.backtrack():
                return True, self.board
            for square_x, square_y, value in revised_list:
                self.board[square_x][square_y].domain.append(value)
                self.unassigned.add(self.board[square_x][square_y])

            # Remove the value from assignment
            target_cell.domain = saved_values
            target_cell.assigned = False
            self.unassigned.add(target_cell)
        return False

    def backtrack(self):
        """CSP back tracking algorithm."""

        # Return true if all squares have been assigned a value
        if len(self.unassigned) == 0:
            self.generate_puzzle_solution()
            return True
        # MRV heuristic with degree heuristic for tiebreak
        next_empty = self.select_unassigned()

        # This uses the least constraining value heuristic to find the order of values
        values = self.find_least_constraining_value(next_empty)
        for v in values:
            if self.is_consistent(next_empty, v):
                # Add the value to assignment
                next_empty.domain = [v]
                next_empty.assigned = True
                self.unassigned.remove(next_empty)

                # MAC using ac-3
                is_inference, revised_list = self.mac(next_empty)

                # We put in a naked pair check which also makes sure the board is still valid
                # after removing values It then also backtracks.
                if is_inference and self.naked_pairs(revised_list) and self.backtrack():
                    return True

                # This is the backtrack part where it adds back onto the domain when a branch fails.
                for square_x, square_y, value in revised_list:
                    self.board[square_x][square_y].domain.append(value)
                    self.unassigned.add(self.board[square_x][square_y])

                # Remove the value from assignment
                next_empty.domain = values
                next_empty.assigned = False
                self.unassigned.add(next_empty)
        return False

    def is_consistent(self, next_empty, v):
        """ Check if the neighbors of next_empty is arc-consistent with it. """
        return not any(
            len(self.board[neighbor[0]][neighbor[1]].domain) == 1
            and v in self.board[neighbor[0]][neighbor[1]].domain
            for neighbor in next_empty.neighbors
        )

    def select_unassigned(self):
        """
        Select the best square to assign next using MRV and Degree heuristics.
        :return: a Square
        """
        squares = self.find_mrv()
        if len(squares) > 1:
            squares = self.find_max_degree(squares)
        return squares[0]

    def find_mrv(self):
        """
        Find the Squares with the Minimum Remaining Values
        :return: a list of Square
        """
        min_size = self.size_data
        mrv = []
        for square in self.unassigned:
            length = len(square.domain)
            if length < min_size:
                mrv = [square]
                min_size = length
            elif length == min_size:
                mrv.append(square)
        return mrv

    def find_max_degree(self, squares):
        """
        Find the squares with the maximum number of degrees from a five list of Squares.
        :param squares: a list of Squares to be evaluated
        :return: a list of Squares
        """
        max_degree = 0
        md = []
        for square in squares:
            degree = len(
                [neighbor for neighbor in square.neighbors
                 if not self.board[neighbor[0]][neighbor[1]].assigned])
            if degree > max_degree:
                md = [square]
                max_degree = degree
            elif degree == max_degree:
                md.append(square)
        return md

    def mac(self, square):
        """
        Maintaining Arc Consistency using AC-3 algorithm.
        :param square: a Square
        """
        arcs = {
            Arc(neighbor[0], neighbor[1], square.row, square.col)
            for neighbor in square.neighbors
            if not self.board[neighbor[0]][neighbor[1]].assigned
        }
        return self.ac3(arcs)

    def ac3(self, arcs):
        """
        Check if the domain of a square is consistent with its neighbors.
        :param arcs: a set of Arcs
        :return: a boolean
        """
        revised_list = []
        while len(arcs) > 0:
            arc = arcs.pop()

            if self.revise(arc, revised_list):
                if len(self.board[arc.square1_x][arc.square1_y].domain) == 0:
                    return False, revised_list
                for neighbor in self.board[arc.square1_x][arc.square1_y].neighbors:
                    if neighbor is not (arc.square2_x, arc.square2_y):
                        arcs.add(
                            Arc(neighbor[0], neighbor[1], arc.square1_x, arc.square1_y))
        return True, revised_list

    def naked_pairs(self, revised_list):
        """
        Naked Pairs where it checks for each row, col and subgrid and check if two cells have the
        same domain of size 2. If that happens it would remove those numbers from the corresponding
        row, col or subgrid.
        """
        for cell in self.unassigned:
            if len(cell.domain) != 2:
                continue
            first_val = cell.domain[0]
            second_val = cell.domain[1]
            neighbor_list = [list(cell.row_neighbors), list(
                cell.col_neighbors), list(cell.sg_neighbors)]
            for neighbor_group in neighbor_list:
                cells_to_modify: List[Square] = []
                naked_found = 0
                # naked_found = False
                for neighbor in neighbor_group:
                    neighbor_domain = self.board[neighbor[0]][neighbor[1]].domain
                    if len(neighbor_domain) == 2 and first_val in neighbor_domain \
                            and second_val in neighbor_domain:
                        naked_found += 1
                        continue
                    cells_to_modify.append(self.board[neighbor[0]][neighbor[1]])
                if naked_found > 1:
                    return False
                if naked_found == 1:
                    for cell_to_modify in cells_to_modify:
                        if first_val in cell_to_modify.domain:
                            cell_to_modify.domain.remove(first_val)
                            revised_list.append((cell_to_modify.row,
                                                 cell_to_modify.col, first_val))
                        if second_val in cell_to_modify.domain:
                            cell_to_modify.domain.remove(second_val)
                            revised_list.append((cell_to_modify.row,
                                                 cell_to_modify.col, second_val))
                        if len(cell_to_modify.domain) == 0:
                            return False
        return True

    def revise(self, arc, revised_list):
        """
        Check if the domain of a square is consistent with its neighbors.
        :param arc: an Arc
        :param revised_list: a List
        :return: a boolean
        """
        if len(self.board[arc.square2_x][arc.square2_y].domain) > 1:
            return False

        for x in self.board[arc.square1_x][arc.square1_y].domain:
            if x in self.board[arc.square2_x][arc.square2_y].domain:
                self.board[arc.square1_x][arc.square1_y].domain.remove(x)
                revised_list.append((arc.square1_x, arc.square1_y, x))
                if len(self.board[arc.square1_x][arc.square1_y].domain) == 1:
                    self.unassigned.remove(self.board[arc.square1_x][arc.square1_y])
                    # self.board[arc.square1_x][arc.square1_y].assigned = True
                return True
        return False

    def find_least_constraining_value(self, square):
        """
        Find the value that will eliminate the least number of values in the domain of its neighbors
        :param square: a Square
        :return: a value
        """
        neighbour_frequency = {val: 0 for val in square.domain}

        for val in square.domain:
            for neighbour in [neighbor for neighbor in square.neighbors if
                              not self.board[neighbor[0]][neighbor[1]].assigned]:
                if val in self.board[neighbour[0]][neighbour[1]].domain:
                    neighbour_frequency[val] += 1

        return sorted(neighbour_frequency, key=neighbour_frequency.get)

    def solve(self):
        """
        Initialize the board and constraints, and then solve the puzzle using CSP.
        :return: if successfully found a solution, bool
        """
        self.init_board()
        self.init_binary_constraints()
        self.ac3(self.init_arc_constraints())
        return self.solve_csp_multiprocess()

    def generate_puzzle_solution(self):
        """ Generate puzzle data from a Board. """
        for i, j in itertools.product(range(self.size_data), range(self.size_data)):

            self.puzzle_data[i][j] = self.board[i][j].domain[0]

    def return_board(self):
        """
        :return: puzzle data, a 2D array
        """
        return self.puzzle_data
