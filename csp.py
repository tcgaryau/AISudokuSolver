import math
import copy
import itertools
import multiprocessing
import functools
from typing import Set, List, Tuple


class Square:
    """
    Represent a square on a sudoku board.
    Each square has a row and col number; a flag indicating whether it has been assigned a value; a set containing
    all values that can be assigned as its domain; a set containing all unassigned neighbors of the square.
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
        return f"Square {self.row} {self.col} domain: {self.domain} length of neighbors: {len(self.neighbors)}"


class Arc:
    """
    Represent an arc between two squares.
    Each arc has a start square and an end square.
    """

    def __init__(self, square1_x, square1_y, square2_x, square2_y):
        self.square1_x = square1_x
        self.square1_y = square1_y
        self.square2_x = square2_x
        self.square2_y = square2_y


class CSP:
    def __init__(self, puzzle_data):
        self.puzzle_data = copy.deepcopy(puzzle_data)
        self.size_data = len(puzzle_data)
        self.board = None
        self.unassigned: Set[Square] = set()

    def init_board(self):
        """ Initialize a sudoku board as a 2D array of Squares, and the domain of each square. """
        size = self.size_data
        board = [[None for _ in range(size)] for _ in range(size)]
        for i, j in itertools.product(range(size), range(size)):
            value = self.puzzle_data[i][j]
            if value == 0:
                # domain = list(self._get_consistent_values(i, j))
                domain = list(range(1, size + 1))
                square = Square(i, j, domain)
                board[i][j] = square
                if len(domain) > 1:
                    self.unassigned.add(square)
            else:
                board[i][j] = Square(i, j, [value], True)
        self.board = board

    def init_constraints(self):
        """ Generate arc constraints for each square. """
        arcs = set()
        size = self.size_data
        for i, j in itertools.product(range(size), range(size)):
            square = self.board[i][j]
            for arc in self._get_arcs(square):
                if arc.square1_x != arc.square2_x and arc.square1_y != arc.square2_y:
                    arcs.add(arc)
        return arcs

    def _get_neighbors(self, square):
        return square.neighbors

    def _get_arcs(self, square):
        neighbors = self._get_neighbors(square)

        return [
            Arc(neighbor[0], neighbor[1], square.row, square.col)
            for neighbor in neighbors
        ]


    def init_binary_constraints(self):
        """ Populate the neighbors field of every square on the current board. """
        board = self.board
        size = self.size_data
        sg_row_total = int(math.sqrt(size))
        sg_col_total = int(math.ceil(math.sqrt(size)))

        for i, j in itertools.product(range(size), range(size)):
            curr_square: Square = board[i][j]
            # for curr_square in self.unassigned:
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
            # curr_square.neighbors = set(list(curr_square.row_neighbors) + list(curr_square.col_neighbors)
            #                             + list(curr_square.sg_neighbors))

    def solve_csp_multiprocess(self):
        """ CSP algorithms. """
        import sys
        sys.setrecursionlimit(10000000)

        # Return true if all squares have been assigned a value
        if len(self.unassigned) == 0:
            self.generate_puzzle_solution()
            return True

        next_empty = self.select_unassigned()
        values = self.find_least_constraining_value(next_empty)
        saved_values = copy.deepcopy(values)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            def quit_process(args):
                for arg in args:
                    if isinstance(arg, tuple):
                        self.board = arg[1]
                        pool.terminate()
                        return True

            starts = [(next_empty, v, saved_values) for v in saved_values]
            print("Starting Length of starts: ", len(starts))

            # using starmap_async (original)
            # results = pool.starmap_async(self.solve_csp_mp_helper, starts,
            #                    callback=quit_process, chunksize=1)
            # results.wait()

            # using imap_unordered
            for results in pool.imap_unordered(
                    functools.partial(self.solve_csp_mp_helper, next_empty, saved_values=saved_values), values,
                    chunksize=1):
                if results:
                    self.board = results[1]
                    pool.terminate()
                    self.generate_puzzle_solution()
                    return True

        return False

        # if self.solve_csp():
        #     print("Finished solving")
        #     return True

    def solve_csp_mp_helper(self, target_cell, v, saved_values):

        if self.is_consistent(target_cell, v):
            # Add the value to assignment
            target_cell.domain = [v]
            target_cell.assigned = True
            self.unassigned.remove(target_cell)

            # MAC using ac-3
            is_inference, revised_list = self.mac(target_cell)
            if is_inference and self.solve_csp():
                return True, self.board
            for square_x, square_y, value in revised_list:
                self.board[square_x][square_y].domain.append(value)
                self.unassigned.add(self.board[square_x][square_y])
                # square.domain.append(value)

            # Remove the value from assignment
            target_cell.domain = saved_values
            target_cell.assigned = False
            self.unassigned.add(target_cell)
        return False

    def solve_csp(self):
        """ CSP algorithms. """

        # Return true if all squares have been assigned a value
        if len(self.unassigned) == 0:
            self.generate_puzzle_solution()
            return True

        next_empty = self.select_unassigned()
        values = self.find_least_constraining_value(next_empty)
        saved_values = copy.deepcopy(values)
        for v in saved_values:
            if self.is_consistent(next_empty, v):
                # Add the value to assignment
                next_empty.domain = [v]
                next_empty.assigned = True
                self.unassigned.remove(next_empty)

                # MAC using ac-3
                is_inference, revised_list = self.mac(next_empty)
                # if is_inference:
                #     valid_naked_pair = self.naked_pairs(revised_list)
                if is_inference and self.naked_pairs(revised_list) and self.solve_csp():
                    return True
                for square_x, square_y, value in revised_list:
                    self.board[square_x][square_y].domain.append(value)
                    self.unassigned.add(self.board[square_x][square_y])

                # Remove the value from assignment
                next_empty.domain = saved_values
                next_empty.assigned = False
                self.unassigned.add(next_empty)


        return False

    def is_consistent(self, next_empty, v):
        return not any(
            len(self.board[neighbor[0]][neighbor[1]].domain) == 1 and v in self.board[neighbor[0]][neighbor[1]].domain
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
                [neighbor for neighbor in square.neighbors if not self.board[neighbor[0]][neighbor[1]].assigned])
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
                        arcs.add(Arc(neighbor[0], neighbor[1], arc.square1_x, arc.square1_y))
        return True, revised_list

    def naked_pairs(self, revised_list):
        # unassigned_cells = [cell for cell in self.unassigned if len(cell.domain) == 2]

        for cell in self.unassigned:
            if len(cell.domain) != 2:
                continue
            first_val = cell.domain[0]
            second_val = cell.domain[1]
            neighbor_list = [list(cell.row_neighbors), list(cell.col_neighbors), list(cell.sg_neighbors)]
            for neighbor_group in neighbor_list:
                cells_to_modify: List[Square] = []
                naked_found = False
                for neighbor in neighbor_group:
                    neighbor_domain = self.board[neighbor[0]][neighbor[1]].domain
                    if len(neighbor_domain) == 2 and first_val in neighbor_domain and second_val in neighbor_domain:
                        naked_found = True
                        continue
                    cells_to_modify.append(self.board[neighbor[0]][neighbor[1]])
                if naked_found:
                    for cell_to_modify in cells_to_modify:
                        if first_val in cell_to_modify.domain:
                            cell_to_modify.domain.remove(first_val)
                            revised_list.append((cell_to_modify.row, cell_to_modify.col, first_val))
                        if second_val in cell_to_modify.domain:
                            cell_to_modify.domain.remove(second_val)
                            revised_list.append((cell_to_modify.row, cell_to_modify.col, second_val))
                        if len(cell_to_modify.domain) == 0:
                            return False
        return True

    def revise(self, arc, revised_list):
        """
        Check if the domain of a square is consistent with its neighbors.
        :param arc: an Arc
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
                return True
        return False

    def find_least_constraining_value(self, square):
        """
        Find the value that will eliminate the least number of values in the domain of its neighbors.
        :param square: a Square
        :return: a value
        """
        current_domain = square.domain
        neighbour_frequency = {val: 0 for val in current_domain}

        for val in current_domain:
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
        arcs = self.init_constraints()
        self.ac3(arcs)
        # return self.solve_csp()
        return self.solve_csp_multiprocess()

    def generate_puzzle_solution(self):
        """ Generate puzzle data from a Board. """
        for i, j in itertools.product(range(self.size_data), range(self.size_data)):
            # print(self.board[i][j].domain[0])
            self.puzzle_data[i][j] = self.board[i][j].domain[0]


    def return_board(self):
        """
        :return: puzzle data, a 2D array
        """
        print("Puzzle data: ", self.puzzle_data)
        return self.puzzle_data


def test():
    # # puzzle = [
    # #     [0, 0, 0, 5, 0, 3, 0, 0, 2],
    # #     [0, 0, 1, 0, 6, 8, 0, 4, 3],
    # #     [0, 0, 0, 0, 4, 0, 0, 0, 0],
    # #     [0, 0, 0, 0, 2, 0, 0, 0, 5],
    # #     [9, 0, 0, 0, 0, 0, 7, 0, 1],
    # #     [0, 8, 0, 9, 0, 0, 0, 0, 0],
    # #     [0, 1, 0, 0, 0, 0, 0, 5, 0],
    # #     [0, 0, 9, 0, 0, 0, 0, 8, 0],
    # #     [0, 0, 6, 0, 0, 0, 0, 3, 9]
    # # ]
    # # puzzle = [
    # #     [4, 0, 1, 0, 0, 0, 6, 0, 0],
    # #     [0, 9, 0, 3, 0, 6, 0, 5, 0],
    # #     [0, 0, 0, 0, 9, 0, 0, 0, 0],
    # #     [0, 2, 0, 0, 0, 0, 0, 0, 9],
    # #     [0, 0, 0, 1, 0, 9, 0, 0, 0],
    # #     [7, 0, 0, 0, 0, 0, 0, 0, 6],
    # #     [0, 0, 0, 0, 2, 0, 0, 0, 0],
    # #     [0, 8, 0, 5, 0, 7, 0, 6, 0],
    # #     [1, 0, 3, 0, 0, 0, 7, 0, 2]
    # # ]
    # # puzzle = [
    # #     [4, 1, 7, 3, 6, 9, 8, 0, 5],
    # #     [0, 3, 0, 0, 0, 0, 0, 0, 0],
    # #     [0, 0, 0, 7, 0, 0, 0, 0, 0],
    # #     [0, 2, 0, 0, 0, 0, 0, 6, 0],
    # #     [0, 0, 0, 0, 8, 0, 4, 0, 0],
    # #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    # #     [0, 0, 0, 6, 0, 3, 0, 7, 0],
    # #     [5, 0, 0, 2, 0, 0, 0, 0, 0],
    # #     [1, 0, 4, 0, 7, 5, 2, 9, 3],
    # # ]
    # # puzzle = [
    # #     [1, 4, 3, 7, 2, 8, 9, 5, 0],
    # #     [9, 0, 0, 3, 0, 5, 0, 0, 1],
    # #     [0, 0, 1, 8, 0, 6, 4, 0, 0],
    # #     [0, 0, 8, 1, 0, 2, 9, 0, 0],
    # #     [7, 0, 0, 0, 0, 0, 0, 0, 8],
    # #     [0, 0, 6, 7, 0, 8, 2, 0, 0],
    # #     [0, 0, 2, 6, 0, 9, 5, 0, 0],
    # #     [8, 0, 0, 2, 0, 3, 0, 0, 9],
    # #     [0, 0, 5, 0, 1, 0, 3, 0, 0]
    # # ]
    # test_puzzle = [
    #     [2, 6, 0, 0, 0, 3, 0, 1, 5],
    #     [4, 7, 0, 0, 0, 0, 0, 0, 8],
    #     [5, 8, 1, 0, 0, 4, 7, 6, 3],
    #     [0, 3, 0, 4, 8, 9, 0, 7, 0],
    #     [0, 0, 6, 0, 0, 2, 8, 3, 0],
    #     [0, 0, 8, 3, 1, 0, 0, 0, 0],
    #     [6, 9, 0, 0, 0, 8, 0, 0, 7],
    #     [3, 0, 0, 0, 9, 0, 2, 0, 0],
    #     [0, 1, 0, 5, 0, 0, 0, 9, 6]
    # ]
    invalid_puzzle = [
        [3, 6, 9, 0, 8, 4, 1, 5, 7],
        [1, 5, 8, 2, 9, 7, 0, 6, 4],
        [4, 7, 2, 6, 5, 1, 3, 8, 9],
        [7, 2, 1, 5, 6, 3, 4, 9, 8],
        [9, 8, 5, 4, 7, 2, 6, 3, 1],
        [6, 3, 4, 9, 1, 8, 5, 7, 2],
        [8, 1, 3, 7, 4, 5, 9, 2, 6],
        [5, 9, 7, 1, 2, 6, 8, 4, 3],
        [2, 4, 6, 8, 3, 9, 7, 1, 5]
    ]
    # hard_puzzle = [[4, 0, 0, 0, 0, 0, 8, 0, 5],
    #                [0, 3, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 7, 0, 0, 0, 0, 0],
    #                [0, 2, 0, 0, 0, 0, 0, 6, 0],
    #                [0, 0, 0, 0, 8, 0, 4, 0, 0],
    #                [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                [0, 0, 0, 6, 0, 3, 0, 7, 0],
    #                [5, 0, 0, 2, 0, 0, 0, 0, 0],
    #                [1, 0, 4, 0, 0, 0, 0, 0, 0],
    #                ]
    # sixteen_puzzle = [[0, 4, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 10, 0, 0, 0],
    #                   [1, 7, 0, 6, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 3, 11, 9, 0],
    #                   [0, 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 4, 0, 16, 0],
    #                   [0, 13, 0, 0, 0, 0, 3, 15, 0, 0, 11, 14, 12, 0, 0, 16],
    #                   [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 15, 0, 5, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 10, 0, 0, 6, 0],
    #                   [0, 0, 0, 0, 0, 0, 13, 0, 0, 16, 0, 0, 0, 0, 2, 0],
    #                   [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 7, 0],
    #                   [10, 0, 0, 7, 16, 0, 0, 0, 0, 0, 14, 0, 0, 4, 0, 0],
    #                   [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
    #                   [0, 0, 9, 16, 0, 5, 0, 2, 0, 15, 7, 8, 0, 1, 0, 0],
    #                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
    #                   [0, 0, 0, 0, 12, 8, 0, 0, 16, 0, 0, 0, 0, 6, 0, 0],
    #                   [8, 0, 0, 10, 1, 13, 0, 0, 0, 4, 0, 0, 0, 2, 3, 0],
    #                   [14, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]]
    # sixteen_puzzle_1 = [[0, 8, 16, 0, 9, 0, 0, 0, 5, 0, 7, 0, 0, 0, 0, 12],
    #                     [6, 0, 0, 0, 5, 0, 0, 0, 0, 14, 0, 0, 0, 4, 16, 0],
    #                     [0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 15, 12, 0, 6, 0, 0],
    #                     [0, 0, 0, 4, 11, 15, 0, 0, 0, 0, 0, 6, 8, 0, 0, 0],
    #                     [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    #                     [0, 0, 10, 0, 0, 14, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 5, 0, 0, 6, 0, 2, 0, 0, 3, 0, 0, 0, 4, 7],
    #                     [2, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 16, 0, 12, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
    #                     [0, 0, 9, 0, 0, 0, 16, 13, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 16, 0, 12, 8, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0],
    #                     [0, 0, 0, 9, 4, 1, 0, 0, 13, 0, 0, 0, 0, 7, 0, 0],
    #                     [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 9, 0, 0],
    #                     [0, 0, 15, 2, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 8, 13]]
    # twenty_five_puzzle = [
    #     [0, 3, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    #     [21, 0, 0, 14, 8, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0],
    #     [25, 0, 0, 0, 0, 4, 0, 9, 0, 13, 8, 0, 6, 0, 0, 18, 0, 0, 0, 0, 14, 19, 0, 10, 0],
    #     [0, 0, 9, 6, 0, 0, 0, 17, 10, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 18, 0, 2, 24, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 16, 4],
    #     [0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 23, 0, 2, 10],
    #     [0, 0, 0, 0, 0, 0, 7, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 3, 0, 0, 6, 0],
    #     [0, 0, 16, 0, 0, 12,
    #      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 14, 10, 0, 22, 0, 6, 25, 0, 0, 23, 0, 0, 0, 17, 18, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 10, 6, 16, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 13, 18, 17],
    #     [2, 0, 0, 0, 0, 0, 25, 8, 0, 0, 0, 21, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 19, 24],
    #     [0, 0, 23, 0, 0, 0, 0, 0, 5, 0, 9, 2, 0, 24, 11, 0, 22, 7, 0, 0, 0, 12, 0, 0, 0],
    #     [0, 7, 11, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 16, 6, 0, 3, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 20, 0, 22, 3, 0, 0, 6, 0, 16, 0, 0, 0, 0, 0, 2, 17, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 10, 25, 24, 11, 20, 0, 3, 0, 0],
    #     [23, 0, 0, 0, 0, 0,
    #      0, 16, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 0, 7, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 14, 0, 0, 0, 0, 20, 0, 0, 0, 0, 9, 0, 0, 0, 0],
    #     [16, 0, 0, 0, 0, 13, 0, 0, 25, 17, 5, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21],
    #     [12, 0, 0, 8, 0, 7, 0, 4, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0], [0, 0, 20, 0, 0, 0, 0, 15, 0, 0, 24, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 7, 0, 0],
    #     [0, 22, 24, 0, 7, 5, 21, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 4, 0, 0],
    #     [0, 0, 0, 0, 15, 0, 0, 0, 7, 18, 0, 22, 0, 4, 16, 0, 0, 8, 0, 0, 0, 0, 0, 23, 0],
    #     [6, 0, 0, 0, 0, 0, 16, 0, 19, 0, 0, 0, 25, 0, 0, 0, 0, 0, 4, 0, 0, 0, 14, 0, 0],
    #     [0, 0, 0, 9, 12, 3, 0, 0, 0, 0, 13, 8, 23, 14, 17, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0]]
    # empty_twenty_five_puzzle = [[0 for _ in range(25)] for _ in range(25)]

    import time
    empty9x9 = [[],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                []]
    empty12x12 = [[],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  []]
    empty16x16 = [[],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  [],
                  []]

    # 25x25
    grp1_25x25_1 = [[0, 0, 1, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 13],
                    [17, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 21, 0, 23, 2, 0, 0, 0, 0, 0, 6, 0, 7, 0, 0],
                    [0, 0, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 1, 17, 0, 0, 6, 0, 0, 0, 2, 0, 0, 5],
                    [0, 0, 0, 23, 0, 17, 0, 0, 0, 0, 5, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 3, 10, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 8, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 4, 0],
                    [0, 17, 18, 0, 0, 0, 0, 0, 0, 3, 23, 0, 21, 0, 0, 0, 0, 10, 0, 0, 16, 7, 0, 0, 0],
                    [0, 0, 0, 0, 9, 14, 0, 0, 10, 0, 0, 0, 0, 0, 0, 13, 23, 0, 20, 0, 0, 0, 0, 0, 0],
                    [10, 3, 0, 0, 1, 0, 0, 0, 24, 0, 0, 0, 19, 8, 4, 0, 0, 0, 0, 0, 0, 14, 0, 20, 0],
                    [11, 0, 0, 14, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0],
                    [0, 0, 0, 20, 19, 0, 0, 0, 0, 0, 22, 5, 0, 14, 0, 21, 0, 0, 12, 0, 0, 10, 0, 3, 15],
                    [0, 0, 17, 0, 0, 12, 0, 3, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 12, 17, 0, 5, 13, 0, 1, 0, 0, 0, 0],
                    [0, 0, 3, 1, 0, 0, 0, 0, 5, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 20],
                    [14, 12, 13, 4, 0, 0, 15, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 22, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 18, 0, 0, 0, 16, 0, 0, 0, 0, 24, 0, 9, 15, 10, 0],
                    [0, 0, 0, 0, 0, 23, 5, 0, 16, 0, 19, 8, 0, 0, 0, 0, 0, 9, 2, 0, 0, 0, 12, 13, 14],
                    [0, 0, 0, 0, 0, 0, 22, 0, 0, 9, 21, 0, 0, 20, 0, 0, 0, 0, 0, 0, 11, 0, 0, 19, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 9, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 18],
                    [19, 0, 0, 8, 23, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 10, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 5, 17, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 17, 0, 3, 0, 0, 0, 0, 20, 0, 0, 23, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 0, 0, 0, 10, 0, 25, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 12],
                    [0, 0, 0, 0, 0, 0, 21, 0, 15, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 6, 18, 5, 0, 0, 0],
                    [23, 0, 0, 0, 0, 0, 0, 11, 0, 1, 0, 18, 0, 0, 0, 4, 7, 12, 24, 0, 0, 25, 0, 0, 9]]
    grp1_25x25_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 12, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 15, 0, 0],
                    [0, 6, 0, 0, 0, 0, 0, 0, 3, 0, 13, 0, 0, 12, 24, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0],
                    [14, 0, 0, 0, 21, 0, 19, 0, 22, 0, 16, 0, 7, 0, 0, 23, 0, 0, 0, 13, 0, 0, 0, 0, 18],
                    [0, 12, 23, 0, 0, 0, 9, 6, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0],
                    [0, 19, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 11, 0, 0, 0, 10, 13, 0],
                    [0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 19, 8, 0, 25, 0, 0, 0, 0, 16, 0, 0, 24],
                    [0, 0, 25, 0, 13, 0, 17, 23, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 19, 0],
                    [9, 0, 0, 0, 5, 20, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 13, 0, 12, 0, 0, 0, 1, 0, 0],
                    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 23, 0, 0, 0, 25, 0, 0],
                    [12, 1, 18, 0, 0, 0, 25, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 7, 0],
                    [25, 11, 8, 0, 0, 0, 23, 0, 0, 16, 18, 0, 0, 10, 2, 4, 0, 0, 0, 0, 0, 19, 22, 0, 0],
                    [0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 13, 0, 0, 0, 0, 22, 0, 0, 0, 0, 5, 2, 0, 0],
                    [0, 7, 0, 0, 3, 0, 0, 1, 0, 0, 0, 19, 0, 0, 0, 0, 0, 25, 0, 18, 11, 0, 24, 12, 0],
                    [0, 0, 0, 0, 0, 0, 0, 18, 7, 11, 12, 0, 0, 25, 13, 0, 19, 0, 0, 22, 0, 0, 0, 0, 23],
                    [0, 9, 21, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 11, 16, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 10, 0, 0, 0],
                    [0, 0, 7, 0, 0, 0, 0, 0, 0, 10, 0, 9, 23, 0, 19, 18, 0, 0, 0, 0, 0, 0, 0, 0, 12],
                    [0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 3, 0, 10, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 9, 0, 20, 19, 0, 0, 0, 0, 0, 7, 11, 0, 23, 3, 10, 12, 18, 0, 0, 0, 0],
                    [0, 0, 0, 0, 18, 6, 0, 9, 0, 0, 0, 4, 0, 5, 0, 14, 24, 0, 0, 0, 0, 0, 0, 0, 0],
                    [11, 0, 0, 0, 0, 0, 5, 0, 23, 0, 0, 0, 6, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 20, 0, 0, 7, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 8, 19, 5, 23, 21, 0, 4],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 15, 0, 0, 0, 0, 0, 24, 0]]
    grp1_25x25_3 = [[0, 0, 0, 6, 23, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 22, 19, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 6, 15, 0, 0, 0, 20, 17, 0, 0, 0, 0],
                    [0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 6, 0, 13, 0, 0, 0, 0, 0],
                    [0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 23, 14, 0],
                    [0, 0, 0, 0, 19, 7, 0, 0, 0, 21, 0, 0, 14, 25, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 12],
                    [0, 0, 0, 23, 0, 13, 17, 4, 3, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 25, 9],
                    [0, 12, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 14],
                    [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 16],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 10, 0, 15, 17, 0, 0, 0, 11, 8, 5, 0, 0, 1],
                    [0, 0, 0, 16, 20, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [4, 0, 0, 0, 0, 0, 25, 1, 0, 0, 19, 0, 0, 0, 0, 7, 5, 0, 21, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 23, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    [6, 0, 0, 7, 18, 11, 0, 0, 0, 3, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0],
                    [0, 21, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 23, 0, 0, 0, 4, 19, 0, 0, 7],
                    [0, 0, 21, 0, 0, 0, 14, 0, 0, 20, 0, 0, 0, 0, 0, 0, 13, 25, 0, 0, 7, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 15, 0, 14, 0, 0, 0, 0, 6, 1, 0, 21, 0],
                    [0, 0, 0, 18, 7, 0, 0, 13, 0, 0, 0, 0, 0, 8, 0, 24, 1, 0, 0, 0, 25, 0, 20, 15, 2],
                    [17, 4, 2, 0, 0, 1, 18, 6, 0, 0, 0, 3, 19, 0, 0, 0, 0, 0, 16, 0, 0, 0, 22, 0, 0],
                    [0, 24, 3, 0, 0, 0, 10, 7, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 12, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0, 0, 0, 20, 0, 0, 22, 0, 0, 0, 0, 0, 23, 0, 12, 0, 3, 0, 0, 19],
                    [0, 0, 20, 25, 0, 12, 0, 0, 0, 22, 10, 0, 0, 0, 0, 0, 17, 3, 18, 14, 0, 8, 6, 0, 0],
                    [0, 7, 0, 0, 0, 0, 13, 0, 4, 0, 0, 0, 0, 0, 0, 0, 10, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 6, 0, 0, 1, 0, 0, 11, 0, 0, 21, 0, 0, 0, 9, 0, 22, 0, 0, 0],
                    [0, 0, 1, 0, 0, 10, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 13, 0]]
    grp2_25x25_1 = [[5, 0, 0, 7, 0, 0, 22, 25, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 18, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 18, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0],
                    [0, 15, 0, 0, 0, 1, 2, 0, 5, 0, 0, 22, 0, 0, 25, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0],
                    [0, 17, 22, 24, 0, 11, 0, 0, 0, 0, 1, 0, 4, 9, 0, 0, 0, 15, 18, 0, 0, 0, 7, 0, 0],
                    [0, 12, 0, 0, 0, 0, 8, 0, 15, 24, 16, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 25],
                    [2, 0, 1, 0, 0, 24, 0, 0, 0, 19, 0, 0, 0, 5, 0, 0, 0, 0, 0, 21, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 20, 0, 0, 6, 0, 0, 0, 11],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 12],
                    [0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 18, 0],
                    [0, 13, 0, 0, 18, 0, 0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 22, 2, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 5],
                    [21, 11, 0, 6, 15, 0, 23, 0, 0, 0, 0, 0, 22, 20, 0, 10, 5, 0, 0, 8, 25, 0, 4, 17, 7],
                    [20, 8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 13, 0, 0, 0, 0, 0, 18, 0, 23, 0, 10],
                    [0, 7, 0, 0, 0, 20, 0, 6, 0, 0, 0, 0, 3, 0, 0, 0, 2, 12, 0, 16, 0, 8, 0, 19, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 17, 21, 0, 0, 6, 0, 0, 22, 0, 0, 0, 0],
                    [3, 21, 8, 0, 1, 0, 0, 16, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0],
                    [25, 20, 0, 22, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 23, 0, 0, 13, 11, 0, 20],
                    [17, 0, 0, 0, 14, 23, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 1, 4, 0, 0, 0, 10, 16, 0, 0],
                    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 23, 0, 0, 10, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 24, 2, 0, 0, 0, 0, 5, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0, 14, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 12, 0, 0, 0, 0, 0],
                    [24, 0, 14, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0, 0, 15, 0],
                    [23, 18, 15, 0, 12, 0, 0, 10, 0, 0, 0, 0, 0, 1, 16, 9, 0, 13, 0, 14, 0, 0, 0, 0, 0]]
    grp2_25x25_2 = [[9, 11, 0, 0, 15, 0, 1, 0, 0, 8, 10, 0, 0, 0, 16, 25, 0, 18, 0, 20, 0, 0, 21, 0, 24],
                    [0, 0, 0, 0, 0, 0, 0, 17, 2, 0, 0, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 13, 14, 0],
                    [0, 0, 0, 0, 0, 18, 0, 0, 23, 0, 0, 0, 0, 19, 11, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0],
                    [0, 0, 17, 23, 0, 0, 0, 6, 0, 0, 5, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 19, 0],
                    [1, 0, 0, 13, 3, 0, 0, 25, 0, 0, 14, 22, 24, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 25, 0, 0, 0, 0, 0, 14, 0, 0, 19, 0, 0, 22, 0],
                    [0, 0, 14, 0, 0, 10, 0, 9, 0, 0, 0, 0, 0, 8, 6, 0, 0, 0, 0, 0, 3, 17, 0, 0, 0],
                    [4, 0, 0, 0, 0, 0, 0, 0, 15, 21, 16, 0, 19, 0, 0, 0, 2, 0, 0, 6, 0, 23, 0, 25, 0],
                    [0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 24, 0, 0, 18, 0, 14, 12, 0, 7, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 19, 0, 0],
                    [0, 0, 0, 9, 0, 7, 0, 0, 0, 5, 2, 0, 1, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0],
                    [14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 21, 0, 0, 3],
                    [0, 0, 0, 0, 0, 11, 0, 0, 22, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0],
                    [0, 23, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16],
                    [0, 0, 0, 0, 0, 19, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 13, 11, 14, 0, 3, 0, 6, 0],
                    [0, 0, 0, 6, 0, 3, 25, 7, 0, 0, 0, 14, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [22, 17, 0, 0, 0, 0, 0, 11, 0, 10, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0],
                    [0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0],
                    [19, 3, 22, 2, 0, 0, 0, 0, 0, 11, 0, 5, 0, 0, 8, 4, 10, 0, 0, 12, 0, 0, 0, 0, 0],
                    [0, 10, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 12, 0, 0, 17, 0, 0, 0, 0, 0, 0, 6, 3, 0],
                    [0, 12, 0, 4, 0, 0, 0, 18, 20, 0, 0, 0, 0, 0, 0, 19, 0, 0, 24, 0, 0, 16, 0, 17, 0],
                    [0, 0, 0, 0, 14, 0, 9, 0, 16, 0, 4, 11, 0, 0, 13, 5, 15, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 6, 25, 0, 20, 8, 24, 0, 0]]
    grp2_25x25_3 = [[0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 9, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
                    [9, 0, 0, 0, 0, 11, 8, 0, 0, 0, 10, 0, 0, 4, 0, 0, 0, 0, 6, 18, 12, 0, 0, 0, 22],
                    [0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 20, 0, 0, 17, 3, 0, 0, 0, 16, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 24, 0, 0, 2, 0, 19, 0, 0, 3, 8, 0, 0],
                    [0, 20, 0, 22, 0, 16, 0, 3, 0, 0, 0, 0, 0, 18, 0, 7, 14, 0, 0, 0, 4, 0, 10, 0, 0],
                    [0, 2, 0, 0, 10, 0, 0, 0, 0, 4, 0, 0, 25, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 22, 0],
                    [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 15, 0, 0, 0],
                    [0, 0, 20, 0, 0, 0, 0, 0, 22, 0, 0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 4, 2, 10, 0, 0, 0],
                    [22, 0, 0, 0, 0, 0, 10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 14, 0, 19, 0, 0, 17, 0, 0],
                    [0, 0, 0, 0, 19, 0, 0, 0, 8, 0, 0, 0, 0, 14, 11, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0],
                    [21, 10, 15, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 0, 0, 0],
                    [0, 0, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 20, 4, 0, 14],
                    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 22, 0, 1, 0, 0, 0, 0, 0, 19, 0],
                    [0, 8, 0, 16, 17, 0, 7, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 3],
                    [0, 14, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 22, 0, 0, 0, 0],
                    [0, 0, 23, 0, 12, 0, 0, 0, 4, 0, 16, 11, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 18, 0],
                    [0, 0, 17, 0, 21, 12, 0, 0, 0, 10, 0, 0, 0, 0, 2, 0, 23, 0, 0, 0, 0, 4, 0, 0, 25],
                    [0, 0, 0, 13, 0, 0, 23, 15, 0, 18, 0, 3, 0, 0, 24, 0, 0, 6, 1, 0, 19, 17, 0, 0, 20],
                    [0, 5, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 1, 0, 6],
                    [0, 22, 0, 0, 0, 0, 0, 0, 0, 21, 0, 17, 0, 25, 12, 18, 0, 7, 0, 20, 10, 0, 14, 13, 0],
                    [0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 7],
                    [0, 19, 0, 15, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 25, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 3, 0, 9, 0, 10, 16, 0, 17, 0, 11, 0, 0],
                    [0, 16, 18, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 3, 14, 0, 0, 0, 0, 1]]
    grp3_25x25_1 = [[0, 0, 0, 13, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 16, 1],
                    [0, 0, 0, 0, 0, 0, 10, 16, 0, 1, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 12, 11, 0],
                    [24, 0, 21, 0, 0, 0, 0, 0, 0, 23, 0, 0, 20, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                    [9, 0, 11, 7, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 18],
                    [0, 1, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 9, 0, 0, 0, 0, 3, 6, 0, 0, 25, 15, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 19, 0, 17, 0, 3, 0, 0, 7, 20, 0, 0, 0, 0],
                    [16, 0, 15, 20, 0, 14, 0, 0, 0, 13, 0, 3, 0, 0, 23, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 9, 23, 19, 0, 0, 25, 0, 0, 0, 14, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0],
                    [0, 6, 0, 24, 0, 11, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 25, 5, 4, 0, 0, 0, 13],
                    [0, 5, 0, 25, 17, 0, 0, 0, 0, 0, 0, 8, 0, 6, 0, 14, 0, 0, 0, 0, 0, 0, 23, 0, 7],
                    [0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 5, 16, 25, 0, 0, 0, 0, 13, 0, 0, 22, 20, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 17, 0, 0, 0, 0, 4, 0],
                    [5, 0, 0, 0, 0, 0, 0, 0, 1, 10, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 23, 7, 19, 0, 0],
                    [0, 0, 0, 0, 0, 13, 21, 0, 2, 8, 0, 0, 0, 0, 0, 6, 24, 0, 0, 3, 0, 0, 0, 0, 15],
                    [0, 0, 0, 0, 0, 12, 5, 0, 0, 25, 0, 0, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0, 0, 17, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 18, 0, 9, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0, 25],
                    [0, 0, 0, 21, 0, 0, 7, 0, 0, 0, 16, 0, 15, 20, 0, 0, 0, 0, 19, 0, 0, 0, 13, 1, 4],
                    [0, 0, 0, 14, 0, 8, 6, 0, 0, 0, 0, 0, 12, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 25, 17, 0, 5, 0, 0, 0, 0, 2, 21, 0, 1, 0, 0, 10, 0, 0, 0, 0, 0, 0],
                    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 15, 0, 0, 4, 0, 10, 0, 0, 0, 0, 11, 9, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0],
                    [0, 0, 13, 0, 0, 18, 0, 0, 0, 0, 0, 5, 17, 0, 0, 0, 0, 25, 12, 0, 10, 0, 0, 0, 14],
                    [0, 0, 22, 0, 0, 0, 0, 13, 0, 21, 0, 0, 23, 0, 0, 0, 0, 9, 0, 11, 15, 0, 0, 0, 16]]
    grp3_25x25_2 = [[1, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 2, 0, 0, 0, 0, 5, 0],
                    [20, 0, 17, 5, 0, 0, 0, 16, 25, 0, 0, 0, 11, 0, 0, 0, 0, 22, 3, 0, 0, 0, 0, 0, 10],
                    [6, 23, 0, 24, 0, 0, 0, 0, 0, 17, 0, 18, 0, 0, 3, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 3, 0, 0, 0, 0],
                    [10, 0, 4, 0, 21, 0, 3, 18, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 23, 7, 0, 0, 0],
                    [0, 0, 0, 0, 19, 0, 1, 3, 21, 0, 0, 0, 25, 0, 12, 0, 0, 5, 0, 0, 0, 0, 0, 18, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 18, 24, 16, 19, 0, 0, 10, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 18, 23, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 12, 25, 14, 2, 0, 10, 0, 16, 15, 0, 13, 0, 0, 20, 0, 0, 0, 0, 0, 0, 3, 0, 21, 0],
                    [22, 0, 0, 0, 0, 0, 0, 0, 7, 0, 21, 3, 0, 0, 0, 0, 0, 0, 10, 0, 12, 2, 0, 0, 0],
                    [15, 0, 21, 19, 1, 0, 22, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 14, 0, 0, 11, 20, 0, 0, 0],
                    [0, 0, 0, 0, 20, 0, 17, 0, 0, 14, 0, 0, 18, 9, 0, 0, 0, 0, 0, 15, 0, 0, 16, 0, 0],
                    [0, 0, 0, 0, 0, 0, 11, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 5],
                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 13, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 18, 22, 0, 0, 0, 21, 0, 15],
                    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16, 0, 0, 0, 0, 14],
                    [0, 0, 2, 0, 0, 0, 15, 0, 0, 0, 0, 17, 0, 0, 5, 0, 0, 0, 24, 0, 0, 22, 0, 0, 0],
                    [0, 0, 0, 0, 22, 0, 0, 0, 6, 23, 0, 4, 19, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0],
                    [0, 5, 0, 20, 0, 0, 25, 0, 0, 2, 0, 11, 23, 0, 0, 0, 22, 0, 9, 0, 15, 0, 0, 10, 16],
                    [0, 0, 0, 0, 0, 0, 18, 24, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23],
                    [0, 0, 0, 0, 0, 19, 0, 0, 0, 1, 0, 0, 0, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 20, 11, 5, 0, 0, 0, 0, 12, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 2],
                    [13, 0, 0, 0, 25, 0, 16, 0, 0, 10, 0, 5, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19],
                    [3, 0, 6, 0, 24, 23, 0, 5, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 2, 0, 25, 0, 0, 13]]
    grp3_25x25_3 = [[16, 0, 0, 0, 23, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 13, 0, 0, 0, 6, 4, 11, 0, 0],
                    [0, 0, 0, 0, 0, 25, 20, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 7, 19],
                    [0, 12, 0, 19, 7, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 11, 0, 2, 0, 3, 0, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0, 9, 0, 0, 0, 13, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0, 0, 0, 8, 18],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 13, 0, 9, 0, 0, 0, 14, 0, 0, 0, 0, 0, 24],
                    [0, 7, 0, 12, 0, 0, 0, 23, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0],
                    [5, 0, 0, 22, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 4],
                    [0, 0, 19, 0, 0, 0, 13, 0, 12, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0],
                    [0, 20, 0, 17, 0, 19, 0, 3, 4, 0, 0, 7, 0, 0, 0, 1, 10, 0, 8, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [14, 0, 0, 16, 0, 0, 0, 0, 0, 0, 15, 25, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0],
                    [4, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 16],
                    [22, 0, 0, 0, 0, 8, 18, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 4, 0, 0, 2, 0],
                    [0, 0, 0, 0, 1, 3, 2, 0, 0, 4, 6, 0, 0, 0, 7, 8, 0, 16, 0, 18, 0, 9, 0, 0, 13],
                    [0, 0, 7, 0, 0, 0, 24, 9, 13, 0, 16, 0, 0, 18, 0, 3, 0, 0, 0, 2, 17, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 15, 17, 0, 10, 0, 25, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 9, 0, 0, 0, 0, 5, 0, 10, 0, 0, 17, 15, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
                    [0, 0, 0, 10, 0, 6, 0, 0, 0, 3, 0, 0, 7, 0, 13, 0, 8, 5, 0, 0, 0, 0, 0, 22, 0],
                    [0, 0, 13, 11, 0, 0, 0, 0, 9, 0, 5, 0, 8, 14, 0, 0, 3, 25, 0, 4, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 17, 0, 0, 0, 0, 0],
                    [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 17, 0, 8, 21, 0, 0, 16, 0, 0, 23],
                    [1, 21, 17, 0, 0, 4, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 24, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0],
                    [0, 16, 0, 0, 0, 0, 0, 21, 0, 1, 20, 15, 0, 0, 4, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0],
                    [19, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 5, 0, 0, 2, 20, 15, 0, 0, 0, 0, 0, 0]]
    grp4_25x25_1 = [[0, 0, 12, 0, 0, 0, 21, 0, 0, 11, 0, 0, 0, 7, 0, 0, 0, 0, 3, 17, 0, 0, 9, 0, 24],
                    [0, 0, 0, 0, 17, 0, 0, 18, 0, 0, 11, 0, 0, 0, 0, 0, 22, 0, 0, 0, 1, 13, 20, 0, 0],
                    [4, 1, 2, 8, 9, 0, 0, 0, 3, 0, 0, 0, 24, 20, 0, 0, 6, 0, 0, 0, 0, 0, 0, 22, 11],
                    [21, 22, 24, 23, 0, 0, 4, 10, 5, 0, 0, 0, 9, 18, 1, 0, 0, 15, 0, 0, 3, 8, 0, 0, 0],
                    [11, 0, 0, 7, 0, 24, 6, 0, 2, 23, 17, 4, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
                    [0, 0, 0, 0, 0, 17, 9, 21, 0, 0, 0, 15, 0, 19, 0, 0, 0, 0, 18, 0, 0, 0, 0, 16, 14],
                    [0, 0, 0, 5, 0, 4, 22, 11, 0, 10, 0, 0, 0, 16, 17, 0, 0, 12, 0, 1, 13, 9, 25, 0, 8],
                    [0, 0, 6, 0, 3, 0, 18, 1, 0, 0, 0, 0, 14, 21, 7, 0, 0, 0, 9, 23, 19, 0, 0, 2, 0],
                    [0, 9, 0, 17, 8, 0, 15, 25, 0, 0, 12, 0, 0, 4, 0, 0, 2, 0, 0, 11, 20, 0, 21, 0, 0],
                    [0, 13, 7, 0, 0, 23, 3, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 10, 0, 18, 0, 4, 22],
                    [13, 0, 18, 0, 5, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 0, 8, 0, 1, 0, 7, 23],
                    [0, 0, 0, 16, 23, 0, 0, 7, 0, 0, 1, 25, 0, 0, 5, 0, 0, 0, 0, 0, 24, 0, 14, 0, 0],
                    [0, 0, 0, 11, 25, 0, 0, 12, 0, 22, 0, 0, 0, 23, 21, 20, 0, 14, 4, 0, 0, 0, 0, 0, 0],
                    [8, 12, 20, 19, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 11, 24, 0, 0, 0, 6, 0, 0, 17, 10, 0],
                    [14, 2, 0, 0, 0, 0, 8, 0, 19, 25, 6, 16, 0, 3, 9, 11, 0, 5, 0, 12, 0, 0, 0, 20, 15],
                    [12, 17, 0, 0, 0, 0, 0, 5, 0, 21, 18, 0, 6, 0, 0, 2, 9, 0, 0, 24, 4, 0, 10, 0, 20],
                    [2, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 25, 19, 0, 0, 21, 22, 16, 0, 0, 24, 0],
                    [20, 0, 0, 24, 16, 0, 10, 0, 0, 0, 0, 0, 17, 1, 0, 0, 0, 23, 0, 5, 18, 25, 0, 3, 0],
                    [0, 8, 0, 0, 14, 25, 17, 0, 0, 0, 24, 9, 19, 5, 0, 6, 0, 0, 20, 0, 0, 11, 23, 1, 0],
                    [0, 0, 11, 25, 6, 20, 1, 0, 0, 7, 0, 0, 16, 14, 0, 0, 0, 0, 10, 15, 17, 12, 0, 0, 21],
                    [22, 23, 0, 0, 0, 21, 0, 16, 0, 0, 0, 8, 0, 0, 18, 7, 0, 24, 0, 0, 0, 14, 13, 0, 17],
                    [0, 7, 0, 15, 0, 0, 20, 0, 0, 6, 0, 24, 0, 2, 14, 13, 0, 0, 11, 3, 0, 0, 5, 0, 25],
                    [3, 21, 0, 10, 0, 7, 25, 14, 15, 19, 0, 0, 0, 9, 0, 22, 0, 6, 0, 0, 2, 0, 0, 0, 0],
                    [9, 0, 0, 0, 0, 18, 5, 0, 0, 0, 23, 19, 15, 0, 10, 0, 0, 1, 0, 0, 0, 0, 11, 0, 0],
                    [0, 16, 0, 0, 20, 3, 0, 24, 13, 4, 0, 0, 0, 17, 0, 0, 0, 0, 0, 25, 0, 21, 12, 15, 0]]
    grp4_25x25_2 = [[0, 0, 12, 0, 0, 0, 21, 0, 0, 11, 0, 0, 0, 7, 0, 0, 0, 0, 3, 17, 0, 0, 9, 0, 24],
                    [0, 0, 0, 0, 17, 0, 0, 18, 0, 0, 11, 0, 0, 0, 0, 0, 22, 0, 0, 0, 1, 13, 20, 0, 0],
                    [4, 1, 2, 8, 9, 0, 0, 0, 3, 0, 0, 0, 24, 20, 0, 0, 6, 0, 0, 0, 0, 0, 0, 22, 11],
                    [21, 22, 24, 23, 0, 0, 4, 10, 5, 0, 0, 0, 9, 18, 1, 0, 0, 15, 0, 0, 3, 8, 0, 0, 0],
                    [11, 0, 0, 7, 0, 24, 6, 0, 2, 23, 17, 4, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
                    [0, 0, 0, 0, 0, 17, 9, 21, 0, 0, 0, 15, 0, 19, 0, 0, 0, 0, 18, 0, 0, 0, 0, 16, 14],
                    [0, 0, 0, 5, 0, 4, 22, 11, 0, 10, 0, 0, 0, 16, 17, 0, 0, 12, 0, 1, 13, 9, 25, 0, 8],
                    [0, 0, 6, 0, 3, 0, 18, 1, 0, 0, 0, 0, 14, 21, 7, 0, 0, 0, 9, 23, 19, 0, 0, 2, 0],
                    [0, 9, 0, 17, 8, 0, 15, 25, 0, 0, 12, 0, 0, 4, 0, 0, 2, 0, 0, 11, 20, 0, 21, 0, 0],
                    [0, 13, 7, 0, 0, 23, 3, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 10, 0, 18, 0, 4, 22],
                    [13, 0, 18, 0, 5, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 0, 0, 0, 8, 0, 1, 0, 7, 23],
                    [0, 0, 0, 16, 23, 0, 0, 7, 0, 0, 1, 25, 0, 0, 5, 0, 0, 0, 0, 0, 24, 0, 14, 0, 0],
                    [0, 0, 0, 11, 25, 0, 0, 12, 0, 22, 0, 0, 0, 23, 21, 20, 0, 14, 4, 0, 0, 0, 0, 0, 0],
                    [8, 12, 20, 19, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 11, 24, 0, 0, 0, 6, 0, 0, 17, 10, 0],
                    [14, 2, 0, 0, 0, 0, 8, 0, 19, 25, 6, 16, 0, 3, 9, 11, 0, 5, 0, 12, 0, 0, 0, 20, 15],
                    [12, 17, 0, 0, 0, 0, 0, 5, 0, 21, 18, 0, 6, 0, 0, 2, 9, 0, 0, 24, 4, 0, 10, 0, 20],
                    [2, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 25, 19, 0, 0, 21, 22, 16, 0, 0, 24, 0],
                    [20, 0, 0, 24, 16, 0, 10, 0, 0, 0, 0, 0, 17, 1, 0, 0, 0, 23, 0, 5, 18, 25, 0, 3, 0],
                    [0, 8, 0, 0, 14, 25, 17, 0, 0, 0, 24, 9, 19, 5, 0, 6, 0, 0, 20, 0, 0, 11, 23, 1, 0],
                    [0, 0, 11, 25, 6, 20, 1, 0, 0, 7, 0, 0, 16, 14, 0, 0, 0, 0, 10, 15, 17, 12, 0, 0, 21],
                    [22, 23, 0, 0, 0, 21, 0, 16, 0, 0, 0, 8, 0, 0, 18, 7, 0, 24, 0, 0, 0, 14, 13, 0, 17],
                    [0, 7, 0, 15, 0, 0, 20, 0, 0, 6, 0, 24, 0, 2, 14, 13, 0, 0, 11, 3, 0, 0, 5, 0, 25],
                    [3, 21, 0, 10, 0, 7, 25, 14, 15, 19, 0, 0, 0, 9, 0, 22, 0, 6, 0, 0, 2, 0, 0, 0, 0],
                    [9, 0, 0, 0, 0, 18, 5, 0, 0, 0, 23, 19, 15, 0, 10, 0, 0, 1, 0, 0, 0, 0, 11, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    grp4_25x25_3 = [[1, 0, 4, 0, 25, 0, 19, 0, 0, 10, 21, 8, 0, 14, 0, 6, 12, 9, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 19, 23, 24, 0, 22, 12, 0, 0, 16, 6, 0, 20, 0, 18, 0, 25, 14, 13, 10, 11, 0, 1, 15],
                    [0, 0, 0, 0,0, 0, 21, 5, 0, 20, 11, 10, 0, 1, 0, 4, 8, 24, 23, 15, 18, 0, 16, 22, 19],
                    [0, 7, 21, 8, 18, 0, 0, 0, 11, 0, 5, 0, 0, 24, 0, 0, 0, 17, 22, 1, 9, 6, 25, 0, 0],
                    [0, 13, 15, 0, 22, 14, 0, 18, 0, 16, 0, 0, 0, 4, 0, 0, 0, 19, 0, 0, 0, 24, 20, 21, 17],
                    [12, 0, 11, 0, 6, 0, 0, 0, 0, 15, 0, 0, 0, 0, 21, 25, 19, 0, 4, 0, 22, 14, 0, 20, 0],
                    [8, 0, 0, 21, 0, 16, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 17, 23, 18, 22, 0, 0, 0, 24, 6],
                    [4, 0, 14, 18, 7, 9, 0, 22, 21, 19, 0, 0, 0, 2, 0, 5, 0, 0, 0, 6, 16, 15, 0, 11, 12],
                    [22, 0, 24, 0, 23, 0, 0, 11, 0, 7, 0, 0, 4, 0, 14, 0, 2, 12, 0, 8, 5, 19, 0, 25, 9],
                    [20, 0, 0, 0, 5, 0, 0, 0, 0, 17, 9, 0, 12, 18, 0, 1, 0, 0, 7, 24, 0, 0, 0, 13, 4],
                    [13, 0, 0, 5, 0, 2, 23, 14, 4, 18, 22, 0, 17, 0, 0, 20, 0, 1, 9, 21, 12, 0, 0, 8, 11],
                    [14, 23, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 20, 25, 0, 3, 4, 13, 0, 11, 21, 9, 5, 18, 22],
                    [7, 0, 0, 11, 17, 20, 24, 0, 0, 0, 3, 4, 1, 12, 0, 0, 6, 14, 0, 5, 25, 13, 0, 0, 0],
                    [0, 0, 16, 9, 0, 17, 11, 7, 10, 25, 0, 0, 0, 13, 6, 0, 0, 18, 0, 0, 19, 4, 0, 0, 20],
                    [6, 15, 0, 19, 4, 13, 0, 0, 5, 0, 18, 11, 0, 0, 9, 8, 22, 16, 25, 10, 7, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 10, 19, 3, 0, 1, 0, 22, 9, 4, 11, 15, 0, 20, 0, 0, 8, 23, 0, 25],
                    [0, 24, 8, 13, 1, 0, 0, 4, 20, 0, 17, 14, 0, 0, 18, 0, 16, 22, 5, 0, 11, 0, 10, 0, 0],
                    [23, 10, 0, 0, 0, 0, 0, 0, 18, 0, 6, 0, 16, 0, 0, 17, 1, 0, 13, 0, 0, 3, 19, 12, 0],
                    [25, 5, 0, 14, 11, 0, 17, 0, 8, 24, 13, 0, 19, 23, 15, 9, 0, 0, 12, 0, 20, 0, 22, 0, 7],
                    [0, 0, 17, 4, 0, 22, 15, 0, 23, 11, 12, 25, 0, 0, 0, 0, 18, 8, 0, 7, 0, 0, 14, 0, 13],
                    [19, 6, 23, 22, 8, 0, 0, 1, 25, 4, 14, 2, 0, 3, 7, 13, 10, 11,16, 0, 0, 0, 0, 0, 0],
                    [0, 4, 0, 17, 0, 3, 0, 24, 0, 8, 20, 23, 11, 10, 25, 22, 0, 0, 0, 12, 13, 2, 18, 6, 0],
                    [0, 0, 7, 16, 0, 0, 6, 17, 2, 21, 0, 18, 0, 0, 0, 19, 0, 0, 8, 0, 0, 0, 0, 4, 0],
                    [18, 9, 25, 1, 2, 11, 0, 0, 13, 22, 4, 0, 21, 0, 5, 0, 23, 7, 0, 0, 15, 0, 3, 0, 8],
                    [0, 21, 10, 0, 0, 12, 0, 20, 16, 0, 19, 0, 0, 0, 0, 15, 14, 4, 2, 18, 23, 25, 11, 7, 0]]
    grp5_25x25_1 = [[0, 0, 0, 0, 0, 20, 0, 0, 9, 0, 25, 14, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
                    [21, 0, 6, 18, 20, 1, 0, 0, 0, 0, 0, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 22],
                    [0, 0, 0, 3, 22, 0, 8, 0, 0, 7, 23, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 10, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 24, 11, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 9, 10, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 24, 0, 0, 14, 0, 0, 12, 0, 17, 0, 1, 0, 6, 25, 0, 0, 0, 21, 0],
                    [0, 0, 14, 0, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 22, 10, 0, 7, 3, 0],
                    [0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 14, 21, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [6, 21, 18, 0, 0, 0, 0, 0, 0, 0, 11, 0, 10, 0, 0, 0, 16, 0, 5, 0, 0, 0, 12, 0, 0],
                    [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 17, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 25, 0, 0, 0, 4, 0, 16, 3, 0, 9, 0, 22, 0, 18, 0, 17, 15, 14, 0, 11, 0, 24, 0],
                    [8, 0, 0, 0, 14, 0, 25, 19, 23, 0, 0, 0, 15, 0, 0, 0, 21, 0, 0, 0, 4, 3, 9, 10, 7],
                    [0, 4, 0, 0, 0, 24, 0, 7, 5, 0, 0, 17, 0, 0, 2, 1, 6, 0, 0, 20, 0, 21, 19, 16, 13],
                    [0, 13, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 7, 0, 0, 4, 1, 0, 0, 18, 6],
                    [11, 0, 22, 9, 4, 0, 5, 0, 0, 0, 0, 12, 0, 17, 23, 25, 0, 0, 0, 6, 13, 19, 16, 15, 0],
                    [24, 0, 0, 17, 0, 0, 0, 0, 0, 18, 0, 1, 0, 0, 0, 22, 2, 0, 13, 21, 7, 9, 10, 0, 0],
                    [0, 2, 15, 5, 0, 11, 0, 0, 22, 0, 21, 16, 0, 3, 0, 0, 4, 12, 0, 0, 0, 0, 18, 0, 0],
                    [0, 18, 20, 0, 0, 17, 1, 2, 0, 0, 0, 0, 11, 0, 0, 0, 9, 16, 3, 0, 0, 25, 0, 0, 8],
                    [10, 12, 7, 25, 0, 0, 16, 0, 3, 6, 0, 13, 0, 9, 0, 15, 0, 0, 17, 0, 0, 0, 24, 5, 0],
                    [3, 0, 11, 4, 9, 0, 0, 8, 0, 5, 0, 0, 0, 12, 0, 6, 0, 25, 20, 0, 0, 0, 15, 19, 0],
                    [14, 8, 24, 16, 17, 0, 19, 12, 0, 23, 6, 0, 0, 0, 5, 21, 0, 22, 0, 11, 0, 7, 0, 0, 10],
                    [2, 0, 19, 13, 5, 0, 0, 0, 0, 20, 0, 0, 21, 16, 0, 7, 0, 0, 0, 12, 0, 6, 0, 0, 0],
                    [0, 20, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 9, 16, 0, 12, 22, 25, 14],
                    [12, 0, 0, 0, 0, 0, 21, 3, 0, 0, 9, 0, 0, 13, 22, 0, 17, 15, 14, 18, 0, 0, 0, 0, 0]]
    grp5_25x25_2 = [[8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 10, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 13, 0, 24, 18, 0, 0, 19, 11, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                    [0, 23, 0, 0, 12, 0, 0, 8, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24],
                    [0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 8, 0, 7, 22, 0, 0, 6, 0, 0, 0, 0, 21, 0, 0, 10],
                    [14, 4, 0, 0, 0, 0, 15, 0, 23, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 17, 0, 0, 12],
                    [12, 5, 0, 0, 0, 0, 1, 0, 3, 0, 0, 11, 25, 0, 0, 4, 0, 14, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 22, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 5, 20, 0, 0, 0, 6, 3, 0, 0, 4, 0],
                    [0, 0, 0, 0, 0, 0, 21, 0, 6, 24, 0, 0, 0, 0, 0, 0, 0, 0, 18, 19, 15, 0, 0, 0, 0],
                    [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 7, 0, 0, 16, 0, 0, 22, 0, 0, 23, 6, 13],
                    [7, 0, 0, 9, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 16, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 5, 16, 13, 23, 0, 22, 0, 0, 6, 0, 9, 11, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0, 0, 22, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 17, 0, 23, 0, 16, 0, 0, 0, 0, 0, 15, 3, 0, 0, 18, 0, 0, 0, 0, 0, 20, 0, 0, 0],
                    [0, 0, 0, 0, 25, 0, 18, 2, 9, 0, 0, 0, 24, 0, 6, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
                    [0, 15, 18, 0, 22, 0, 0, 0, 11, 5, 0, 0, 0, 6, 0, 0, 12, 23, 0, 0, 0, 0, 20, 7, 0],
                    [23, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 12, 0, 0, 7, 11, 0, 0, 0, 0, 0, 0, 0, 5, 6],
                    [17, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 10, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 24, 0, 4, 0, 0, 0, 15, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 11, 0, 3, 6, 0, 0, 20, 16, 0, 23, 0, 0, 0],
                    [0, 0, 0, 18, 0, 0, 0, 4, 0, 16, 0, 7, 0, 0, 0, 3, 17, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 20, 14, 0, 3, 0, 5, 0, 0, 0, 1, 18, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 7],
                    [0, 21, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 4, 3, 0, 23, 0, 22, 0, 0, 0, 5, 0, 0, 0],
                    [4, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [15, 0, 0, 11, 9, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0]]
    grp5_25x25_3 = [[18, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 17, 14, 0, 24, 0],
                    [0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 8, 0, 6, 4, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 3, 0, 0],
                    [0, 0, 0, 14, 13, 0, 0, 0, 0, 0, 21, 0, 0, 23, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 7, 0, 0, 0, 11, 0, 13, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 20, 0, 21, 23, 0, 0, 25, 0, 17, 5],
                    [0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 10, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 19, 0, 0, 11, 0, 4, 10, 0, 1, 0, 0, 0, 8, 0, 14, 0, 9, 6, 24, 0, 0, 0, 0, 7],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 8, 18, 0, 0, 0, 0, 23, 0, 0, 0],
                    [0, 0, 0, 0, 22, 0, 0, 25, 0, 13, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0],
                    [0, 0, 0, 0, 2, 22, 8, 0, 0, 0, 0, 25, 0, 20, 0, 9, 12, 0, 0, 0, 24, 0, 0, 0, 16],
                    [0, 0, 19, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 24, 0, 11, 0, 0, 2, 20, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 13, 0, 0, 9, 12, 0, 0, 8, 0, 23, 1, 0, 0, 25, 0, 0, 0, 0, 0, 0],
                    [0, 5, 0, 0, 20, 0, 17, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 1, 0, 0, 18],
                    [23, 0, 22, 7, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 12, 0, 0, 0, 25, 0, 0, 0, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 12, 0, 5, 0, 0, 0, 24, 1, 0, 0, 0, 2, 14],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 22, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 8, 5, 0, 17, 0, 0, 0, 0, 20, 0, 0],
                    [10, 0, 0, 0, 5, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 7, 0, 0],
                    [13, 0, 5, 0, 19, 24, 11, 0, 0, 0, 0, 14, 0, 0, 0, 4, 0, 0, 12, 0, 0, 21, 0, 0, 0],
                    [17, 8, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 21, 0, 0, 11, 0, 0, 0, 0, 18, 0],
                    [0, 0, 0, 2, 0, 7, 0, 0, 5, 0, 0, 3, 0, 0, 21, 0, 0, 20, 16, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 15, 22, 0, 0, 0, 0, 0, 12, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [7, 0, 0, 0, 9, 0, 0, 0, 0, 0, 24, 0, 1, 25, 13, 15, 8, 0, 18, 0, 22, 3, 0, 16, 19]]
    all_25x25 = [grp1_25x25_1, grp1_25x25_2, grp1_25x25_3, grp2_25x25_1, grp2_25x25_2, grp2_25x25_3, grp3_25x25_1, grp3_25x25_2, grp3_25x25_3, grp4_25x25_1, grp4_25x25_2, grp4_25x25_3, grp5_25x25_1, grp5_25x25_2, grp5_25x25_3]

    # 16x16
    grp1_16x16_1 = [[0, 8, 16, 0, 9, 0, 0, 0, 5, 0, 7, 0, 0, 0, 0, 12],
                    [6, 0, 0, 0, 5, 0, 0, 0, 0, 14, 0, 0, 0, 4, 16, 0],
                    [0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 15, 12, 0, 6, 0, 0],
                    [0, 0, 0, 4, 11, 15, 0, 0, 0, 0, 0, 6, 8, 0, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
                    [0, 0, 10, 0, 0, 14, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 5, 0, 0, 6, 0, 2, 0, 0, 3, 0, 0, 0, 4, 7],
                    [2, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 16, 0, 12, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0],
                    [0, 0, 9, 0, 0, 0, 16, 13, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 16, 0, 12, 8, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 0],
                    [0, 0, 0, 9, 4, 1, 0, 0, 13, 0, 0, 0, 0, 7, 0, 0],
                    [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                    [0, 0, 15, 2, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 8, 13]]
    grp2_16x16_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 5, 2, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 6, 0, 11, 0, 0, 7, 0, 12, 0, 0, 0, 0, 0, 0, 0],
                    [0, 8, 0, 0, 13, 11, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0],
                    [0, 0, 0, 16, 0, 7, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0],
                    [8, 3, 0, 0, 6, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0],
                    [0, 9, 0, 0, 8, 0, 0, 1, 0, 0, 5, 0, 0, 0, 0, 12],
                    [0, 1, 0, 0, 0, 0, 0, 0, 2, 16, 0, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 9, 12, 0, 6, 0, 5, 0, 0, 0, 0, 0, 0],
                    [0, 13, 1, 0, 0, 0, 0, 4, 0, 3, 16, 0, 0, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 4, 12, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 7, 0, 16, 0],
                    [13, 0, 11, 8, 5, 4, 0, 0, 0, 9, 0, 2, 0, 10, 15, 0],
                    [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                    [0, 0, 5, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 12, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 11, 12, 14, 0, 0, 0]]
    grp3_16x16_1 = [[0, 9, 8, 0, 0, 7, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
                    [10, 0, 0, 5, 0, 0, 15, 0, 4, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 6, 10, 0, 2, 0, 0, 0, 0, 4, 0],
                    [0, 4, 0, 0, 0, 8, 12, 13, 0, 0, 7, 0, 0, 9, 0, 0],
                    [0, 0, 0, 8, 3, 0, 0, 11, 0, 0, 0, 0, 0, 6, 0, 0],
                    [0, 0, 0, 6, 10, 0, 0, 0, 16, 0, 0, 0, 11, 8, 14, 0],
                    [0, 14, 0, 0, 0, 2, 0, 12, 0, 0, 0, 6, 4, 0, 7, 0],
                    [0, 1, 0, 0, 0, 0, 8, 14, 13, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0, 0, 6, 13, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
                    [0, 0, 0, 12, 0, 0, 3, 0, 0, 0, 4, 15, 0, 0, 8, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 0, 6, 0, 0, 12, 14, 16, 4, 0, 0, 0, 0],
                    [0, 0, 7, 0, 0, 0, 0, 0, 0, 11, 0, 5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 4, 0, 3],
                    [14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0]]
    grp4_16x16_1 = [[0, 0, 0, 0, 0, 0, 0, 2, 10, 0, 0, 12, 0, 9, 0, 0],
                    [12, 0, 0, 0, 0, 9, 0, 15, 1, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 13, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 6, 5, 0, 0, 0, 0, 0, 3, 16, 0, 0, 14, 0, 0, 0],
                    [0, 3, 0, 0, 0, 11, 0, 0, 0, 2, 0, 14, 0, 0, 0, 0],
                    [14, 0, 0, 15, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 6],
                    [0, 0, 11, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 16, 5],
                    [0, 0, 0, 0, 9, 0, 0, 12, 0, 0, 0, 0, 11, 0, 0, 0],
                    [0, 0, 0, 14, 16, 12, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
                    [0, 1, 0, 12, 14, 0, 0, 13, 0, 0, 0, 0, 10, 0, 0, 0],
                    [0, 0, 7, 0, 8, 0, 10, 0, 0, 0, 12, 0, 0, 0, 0, 0],
                    [11, 2, 0, 0, 0, 0, 0, 9, 0, 0, 5, 15, 0, 16, 0, 0],
                    [3, 0, 0, 6, 0, 0, 0, 0, 11, 0, 0, 0, 0, 14, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 15, 0, 5, 11],
                    [0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 10, 0]]
    grp5_16x16_1 = [[0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 10, 14, 13, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 16, 15, 0, 1, 0, 0, 0, 6],
                    [0, 11, 0, 10, 9, 1, 0, 0, 0, 0, 0, 2, 0, 14, 0, 0],
                    [0, 0, 0, 8, 0, 16, 0, 14, 0, 0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 15, 0, 13, 7, 0, 0, 12, 0, 0, 0, 0, 10],
                    [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 14, 0, 0, 0, 4, 0, 13, 0, 0, 0, 12],
                    [0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 6, 0, 0, 9, 0, 14, 0, 0, 0, 0, 13, 1, 0],
                    [15, 0, 0, 0, 5, 0, 3, 6, 0, 0, 0, 0, 0, 2, 0, 0],
                    [13, 16, 2, 0, 0, 0, 0, 0, 3, 12, 6, 0, 0, 0, 0, 9],
                    [0, 0, 0, 0, 0, 13, 0, 4, 9, 0, 1, 7, 0, 8, 6, 0],
                    [16, 15, 0, 0, 0, 0, 0, 0, 8, 1, 0, 10, 0, 0, 7, 2],
                    [0, 0, 0, 11, 0, 0, 0, 0, 2, 14, 0, 0, 0, 0, 0, 4],
                    [5, 10, 0, 2, 16, 12, 7, 0, 15, 11, 4, 6, 0, 9, 0, 14],
                    [1, 0, 0, 4, 2, 8, 0, 9, 0, 5, 16, 3, 11, 12, 10, 0]]
    grp1_16x16_2 = [[0, 0, 0, 0, 0, 0, 0, 10, 6, 0, 0, 0, 0, 1, 0, 0],
                    [12, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 9, 11, 0, 0, 0],
                    [0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 16, 0],
                    [0, 0, 0, 0, 0, 8, 15, 0, 0, 0, 0, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 4, 0, 0, 3, 11, 0, 0, 0, 0, 0, 0],
                    [16, 6, 0, 0, 0, 0, 12, 11, 0, 7, 0, 13, 2, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 16, 0, 12, 0],
                    [0, 0, 0, 11, 0, 15, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 12, 0, 0, 0, 0, 3, 0, 16, 2, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 13, 15, 0, 6, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 7, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 9, 0, 1, 3, 7, 0, 14],
                    [0, 13, 0, 0, 0, 12, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 3, 9, 13, 0, 10, 0, 8, 0, 0, 0, 0, 16, 6, 12],
                    [0, 0, 0, 15, 0, 0, 0, 0, 13, 3, 0, 0, 0, 0, 0, 0]]
    grp2_16x16_2 = [[6, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 3, 14, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 0, 0, 16, 0, 7, 14, 0, 0, 6, 0],
                    [0, 0, 0, 0, 3, 0, 13, 0, 0, 0, 5, 0, 7, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 14, 0, 0, 0],
                    [0, 0, 0, 2, 0, 13, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 14, 15, 4, 3, 0, 0, 0, 0, 1, 8, 16, 0, 0],
                    [0, 0, 0, 10, 0, 0, 0, 0, 13, 0, 8, 15, 0, 12, 0, 0],
                    [2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 11, 16, 14],
                    [4, 0, 0, 11, 5, 0, 6, 13, 0, 0, 0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 4, 0, 0, 11, 0, 6, 0, 0, 0, 0, 0, 0],
                    [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0],
                    [10, 5, 0, 0, 0, 0, 14, 0, 11, 0, 0, 0, 0, 8, 7, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 1, 0, 0],
                    [0, 0, 12, 7, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0],
                    [0, 14, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    grp3_16x16_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                    [0, 7, 4, 0, 0, 10, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0],
                    [11, 0, 5, 0, 0, 3, 0, 0, 0, 0, 10, 0, 0, 0, 0, 15],
                    [0, 14, 0, 0, 0, 9, 0, 0, 3, 0, 12, 0, 10, 0, 0, 11],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0],
                    [13, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                    [2, 0, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 0, 10, 15, 0],
                    [15, 10, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 5, 0, 0],
                    [8, 0, 10, 0, 0, 0, 4, 7, 0, 0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 2, 0, 0, 6, 0, 0, 0, 5, 3, 15, 14, 0, 10],
                    [0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                    [16, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 13, 2],
                    [0, 11, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1],
                    [0, 0, 0, 0, 15, 0, 3, 16, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 13, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0, 2, 0, 0, 0, 6, 11, 0, 3, 0]]
    grp4_16x16_2 = [[0, 4, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 10, 0, 0, 0],
                    [1, 7, 0, 6, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 3, 11, 9, 0],
                    [0, 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 4, 0, 16, 0],
                    [0, 13, 0, 0, 0, 0, 3, 15, 0, 0, 11, 14, 12, 0, 0, 16],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 15, 0, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 10, 0, 0, 6, 0],
                    [0, 0, 0, 0, 0, 0, 13, 0, 0, 16, 0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 7, 0],
                    [10, 0, 0, 7, 16, 0, 0, 0, 0, 0, 14, 0, 0, 4, 0, 0],
                    [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                    [0, 0, 9, 16, 0, 5, 0, 2, 0, 15, 7, 8, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    [0, 0, 0, 0, 12, 8, 0, 0, 16, 0, 0, 0, 0, 6, 0, 0],
                    [8, 0, 0, 10, 1, 13, 0, 0, 0, 4, 0, 0, 0, 2, 3, 0],
                    [14, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]]
    grp5_16x16_2 = [[0, 0, 0, 0, 0, 0, 0, 2, 10, 0, 0, 12, 0, 9, 0, 0],
                    [12, 0, 0, 0, 0, 9, 0, 15, 1, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 13, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 6, 5, 0, 0, 0, 0, 0, 3, 16, 0, 0, 14, 0, 0, 0],
                    [0, 3, 0, 0, 0, 11, 0, 0, 0, 2, 0, 14, 0, 0, 0, 0],
                    [14, 0, 0, 15, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 6],
                    [0, 0, 11, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 16, 5],
                    [0, 0, 0, 0, 9, 0, 0, 12, 0, 0, 0, 0, 11, 0, 0, 0],
                    [0, 0, 0, 14, 16, 12, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
                    [0, 1, 0, 12, 14, 0, 0, 13, 0, 0, 0, 0, 10, 0, 0, 0],
                    [0, 0, 7, 0, 8, 0, 10, 0, 0, 0, 12, 0, 0, 0, 0, 0],
                    [11, 2, 0, 0, 0, 0, 0, 9, 0, 0, 5, 15, 0, 16, 0, 0],
                    [3, 0, 0, 6, 0, 0, 0, 0, 11, 0, 0, 0, 0, 14, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 15, 0, 5, 11],
                    [0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 10, 0]]
    grp1_16x16_3 = [[0, 0, 0, 0, 5, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [4, 0, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0],
                    [0, 16, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
                    [0, 0, 0, 8, 0, 0, 10, 0, 3, 0, 7, 0, 0, 0, 0, 13],
                    [0, 2, 15, 0, 12, 11, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 7, 0, 0, 14, 0, 12, 6, 0],
                    [8, 0, 0, 0, 0, 0, 0, 13, 0, 11, 0, 10, 0, 0, 0, 0],
                    [7, 0, 6, 0, 0, 14, 15, 5, 0, 0, 0, 8, 0, 0, 0, 0],
                    [0, 0, 0, 10, 14, 0, 0, 6, 4, 16, 0, 0, 0, 3, 0, 0],
                    [0, 0, 0, 14, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 16, 0, 0, 7, 0, 9, 0, 0, 0, 13, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 10, 0, 3, 0, 0, 0, 0, 0, 0],
                    [9, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 7, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 15, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 14],
                    [16, 14, 0, 6, 3, 0, 8, 0, 0, 0, 0, 15, 0, 0, 0, 0]]
    grp2_16x16_3 = [[3, 0, 0, 0, 2, 0, 15, 0, 7, 6, 1, 0, 11, 16, 8, 0],
                    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0],
                    [1, 0, 0, 11, 3, 0, 0, 14, 0, 0, 0, 16, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 4, 0, 0, 14, 0, 0],
                    [15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10],
                    [0, 0, 0, 0, 0, 16, 0, 2, 10, 0, 0, 4, 0, 0, 0, 0],
                    [0, 0, 6, 0, 0, 14, 0, 1, 0, 0, 0, 0, 16, 0, 2, 12],
                    [0, 0, 0, 0, 0, 0, 10, 0, 2, 0, 0, 0, 0, 0, 13, 0],
                    [0, 0, 16, 0, 0, 0, 0, 0, 0, 5, 0, 0, 10, 0, 0, 11],
                    [0, 13, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 9],
                    [0, 0, 10, 0, 0, 0, 5, 16, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 15, 0],
                    [0, 10, 3, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 4],
                    [0, 16, 0, 2, 0, 10, 0, 0, 0, 4, 0, 1, 0, 12, 0, 0],
                    [0, 15, 0, 0, 0, 0, 2, 3, 16, 0, 0, 0, 14, 0, 0, 0]]
    grp3_16x16_3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 0, 0, 0, 4],
                    [0, 0, 10, 0, 0, 0, 0, 0, 12, 15, 0, 0, 0, 0, 0, 6],
                    [0, 9, 0, 0, 0, 0, 2, 10, 0, 4, 0, 7, 0, 15, 16, 0],
                    [0, 0, 0, 7, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 9, 0],
                    [11, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0],
                    [0, 0, 4, 0, 7, 0, 0, 6, 0, 1, 0, 0, 0, 0, 8, 0],
                    [0, 0, 0, 0, 14, 0, 0, 0, 16, 0, 11, 0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 8, 11, 0, 0, 12, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 0, 0, 0, 12, 15, 0, 11, 0, 0, 16],
                    [0, 1, 0, 0, 12, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 10],
                    [9, 15, 0, 0, 0, 16, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 0, 0, 10, 0, 8],
                    [6, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0, 5, 0, 0, 7, 0, 0, 1, 0, 0, 0],
                    [0, 11, 13, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0]]
    grp4_16x16_3 = [[0, 3, 0, 0, 0, 4, 0, 7, 0, 0, 0, 0, 8, 11, 0, 0],
                    [0, 0, 12, 15, 0, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0],
                    [9, 0, 0, 14, 0, 0, 2, 0, 0, 0, 0, 0, 16, 1, 12, 10],
                    [0, 11, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 14, 0],
                    [0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 0, 0, 0, 0],
                    [0, 7, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 8, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 1, 2, 0, 0, 0],
                    [0, 0, 0, 11, 0, 0, 0, 0, 0, 1, 0, 0, 0, 13, 0, 0],
                    [0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 8, 0, 0, 0, 14, 13, 11, 3, 7, 0, 0, 1, 0, 0, 0],
                    [5, 0, 0, 0, 4, 3, 0, 9, 0, 0, 0, 0, 7, 0, 0, 15],
                    [0, 16, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 14, 10, 0, 13],
                    [1, 0, 0, 0, 0, 9, 0, 0, 0, 10, 0, 2, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 6, 0, 0, 7]
                    ]
    grp5_16x16_3 = [[0, 3, 0, 0, 0, 4, 0, 7, 0, 0, 0, 0, 8, 11, 0, 0],
                    [0, 0, 12, 15, 0, 0, 0, 0, 16, 0, 0, 0, 13, 0, 0, 0],
                    [9, 0, 0, 14, 0, 0, 2, 0, 0, 0, 0, 0, 16, 1, 12, 10],
                    [0, 11, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 14, 0],
                    [0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 0, 0, 0, 0],
                    [0, 7, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 13, 8, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 1, 2, 0, 0, 0],
                    [0, 0, 0, 11, 0, 0, 0, 0, 0, 1, 0, 0, 0, 13, 0, 0],
                    [0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 8, 0, 0, 0, 14, 13, 11, 3, 7, 0, 0, 1, 0, 0, 0],
                    [5, 0, 0, 0, 4, 3, 0, 9, 0, 0, 0, 0, 7, 0, 0, 15],
                    [0, 16, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 14, 10, 0, 13],
                    [1, 0, 0, 0, 0, 9, 0, 0, 0, 10, 0, 2, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 6, 0, 0, 7]]
    all_16x16 = [grp1_16x16_1, grp1_16x16_2, grp1_16x16_3, grp2_16x16_1, grp2_16x16_2, grp2_16x16_3, grp3_16x16_1,
                 grp3_16x16_2, grp3_16x16_3, grp4_16x16_1, grp4_16x16_2, grp4_16x16_3, grp5_16x16_1, grp5_16x16_2,
                 grp5_16x16_3]

    # 12x12
    grp1_12x12_1 = [[0, 0, 0, 5, 0, 0, 6, 0, 9, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                    [8, 0, 0, 4, 0, 0, 7, 0, 0, 0, 0, 0],
                    [1, 4, 0, 0, 0, 0, 0, 12, 2, 0, 0, 0],
                    [0, 2, 0, 0, 10, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 12, 0, 0, 11, 0, 0],
                    [0, 0, 0, 2, 0, 4, 0, 0, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 2],
                    [0, 0, 0, 0, 0, 8, 3, 0, 0, 0, 0, 9],
                    [7, 0, 0, 0, 6, 0, 0, 0, 5, 12, 0, 8],
                    [4, 3, 9, 0, 12, 0, 0, 0, 11, 0, 0, 0]]
    grp1_12x12_2 = [[0, 0, 0, 0, 4, 10, 0, 12, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0],
                    [11, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 9, 0, 0, 4, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 7],
                    [0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 5, 0],
                    [0, 0, 0, 0, 0, 2, 0, 9, 0, 0, 0, 0],
                    [0, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    [0, 0, 0, 0, 0, 0, 3, 0, 0, 11, 7, 0],
                    [0, 0, 0, 0, 7, 0, 6, 0, 12, 0, 1, 0],
                    [0, 11, 0, 0, 2, 0, 0, 1, 10, 0, 3, 6]]
    grp1_12x12_3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
                    [3, 0, 5, 0, 0, 11, 0, 6, 0, 8, 1, 12],
                    [0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 11, 0],
                    [1, 0, 11, 0, 6, 2, 0, 0, 5, 0, 9, 8],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 9, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 3, 7, 0, 0, 0, 1, 8, 0, 0, 0],
                    [8, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 1],
                    [0, 10, 0, 0, 0, 1, 0, 4, 0, 0, 3, 9],
                    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    grp2_12x12_1 = [[0, 10, 0, 0, 0, 6, 4, 0, 8, 0, 0, 0],
                    [0, 8, 9, 0, 0, 3, 0, 11, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 8, 10, 0, 0, 3, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 8, 0, 0],
                    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 5, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0],
                    [1, 4, 0, 0, 8, 7, 0, 0, 0, 11, 0, 0],
                    [0, 0, 10, 8, 0, 5, 0, 0, 0, 0, 0, 0],
                    [0, 7, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
                    [0, 2, 3, 0, 0, 0, 0, 0, 0, 10, 1, 6]]
    grp2_12x12_2 = [[0, 12, 0, 0, 2, 0, 3, 0, 5, 10, 0, 0],
                    [0, 0, 11, 0, 0, 0, 0, 0, 12, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0],
                    [0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 12, 0, 0, 6, 0, 2, 4, 8, 0],
                    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 9, 12, 5, 0, 8, 0, 2, 4],
                    [0, 9, 12, 0, 1, 10, 0, 0, 0, 0, 0, 0],
                    [8, 0, 0, 0, 0, 0, 0, 7, 10, 0, 0, 0]]
    grp2_12x12_3 = [[9, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 12, 0, 7, 0, 0, 0, 0, 0, 8, 0, 5],
                    [0, 3, 0, 11, 8, 0, 0, 0, 10, 0, 7, 2],
                    [5, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0],
                    [0, 0, 0, 9, 0, 0, 0, 3, 1, 0, 0, 0],
                    [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 6, 0, 0, 4, 7, 0, 0, 0, 0],
                    [4, 1, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0],
                    [0, 9, 0, 0, 11, 0, 0, 0, 0, 0, 0, 10],
                    [0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 7, 6, 0, 0, 3, 0, 0, 1],
                    [11, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    grp3_12x12_1 = [[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [6, 0, 0, 11, 0, 0, 4, 0, 3, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0],
                    [11, 0, 12, 0, 3, 5, 0, 0, 0, 0, 10, 8],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5],
                    [0, 0, 0, 0, 0, 6, 0, 5, 0, 11, 0, 9],
                    [0, 9, 0, 0, 11, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 6, 0, 10, 0, 2, 8, 0, 0, 0],
                    [0, 8, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0],
                    [0, 0, 10, 0, 0, 0, 0, 0, 0, 7, 0, 0],
                    [9, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 1]]
    grp3_12x12_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 6, 11, 0, 0, 3, 0, 0, 0],
                    [3, 6, 0, 0, 1, 0, 5, 10, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 11, 5],
                    [0, 12, 5, 0, 0, 0, 0, 0, 0, 0, 9, 10],
                    [5, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 3],
                    [0, 0, 6, 0, 0, 10, 0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 11, 2, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 2, 5, 8, 0, 0, 0, 0, 9, 3, 0],
                    [0, 3, 0, 0, 12, 0, 0, 0, 0, 11, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2]]
    grp3_12x12_3 = [[0, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0, 12],
                    [3, 0, 0, 0, 8, 0, 7, 0, 0, 0, 2, 0],
                    [2, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 11],
                    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 3, 5, 0, 0, 0, 6, 0, 0, 0, 0],
                    [10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 12, 0],
                    [1, 0, 0, 7, 0, 0, 4, 10, 2, 6, 0, 0],
                    [0, 6, 0, 9, 0, 0, 0, 0, 0, 0, 0, 7],
                    [0, 0, 0, 0, 12, 6, 0, 11, 1, 0, 9, 3],
                    [0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    grp4_12x12_1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 3, 0],
                    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 2, 4],
                    [0, 0, 0, 0, 2, 0, 0, 0, 8, 10, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 4, 9, 0, 0],
                    [5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6, 0],
                    [0, 0, 0, 6, 0, 8, 0, 0, 0, 0, 0, 12],
                    [0, 1, 0, 0, 0, 12, 11, 0, 7, 0, 0, 0],
                    [0, 12, 2, 0, 0, 0, 0, 7, 3, 0, 0, 0],
                    [0, 6, 0, 9, 0, 0, 0, 0, 0, 0, 0, 11],
                    [4, 0, 0, 0, 0, 0, 2, 0, 12, 0, 0, 3],
                    [8, 0, 0, 7, 5, 11, 0, 0, 0, 0, 0, 0]]
    grp4_12x12_2 = [[10, 0, 7, 11, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 9, 8, 10, 2, 0, 0, 0],
                    [0, 0, 4, 0, 7, 0, 3, 1, 0, 0, 8, 0],
                    [0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 10, 0],
                    [1, 9, 12, 0, 0, 4, 0, 0, 0, 0, 0, 7],
                    [0, 0, 0, 0, 11, 3, 12, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 6, 10, 0, 0, 0, 0, 0, 12],
                    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 9, 12, 0, 0, 0, 2]]
    grp4_12x12_3 = [[0, 0, 0, 0, 3, 8, 0, 0, 0, 0, 0, 0],
                    [9, 0, 10, 0, 0, 0, 11, 0, 0, 4, 0, 2],
                    [11, 0, 0, 12, 10, 0, 0, 0, 0, 0, 0, 6],
                    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                    [12, 0, 0, 9, 0, 0, 0, 3, 5, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0, 8, 0, 11, 0, 0, 0],
                    [0, 0, 6, 0, 0, 0, 5, 0, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [8, 7, 0, 0, 0, 10, 2, 0, 1, 0, 4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
                    [0, 0, 0, 0, 0, 12, 0, 6, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0, 2, 0, 0, 10, 0, 0, 5]]
    grp5_12x12_1 = [[0, 0, 0, 11, 8, 0, 0, 0, 0, 0, 0, 6],
                    [10, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 11],
                    [0, 7, 0, 0, 0, 0, 2, 6, 0, 0, 1, 0],
                    [7, 0, 12, 0, 0, 0, 9, 0, 0, 0, 0, 0],
                    [0, 0, 0, 10, 6, 2, 0, 0, 5, 0, 0, 0],
                    [0, 9, 0, 8, 0, 10, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 5, 0, 7, 8, 0, 0, 6, 3],
                    [0, 0, 0, 7, 2, 0, 10, 0, 8, 0, 12, 5],
                    [0, 6, 0, 5, 0, 0, 12, 0, 0, 1, 0, 0],
                    [6, 5, 7, 12, 0, 3, 8, 2, 0, 11, 0, 0],
                    [0, 0, 8, 0, 0, 1, 5, 0, 0, 7, 3, 9],
                    [9, 3, 4, 1, 0, 0, 6, 10, 12, 0, 0, 0]]
    grp5_12x12_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 3, 0],
                    [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 2, 4],
                    [0, 0, 0, 0, 2, 0, 0, 0, 8, 10, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 4, 9, 0, 0],
                    [5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6, 0],
                    [0, 0, 0, 6, 0, 8, 0, 0, 0, 0, 0, 12],
                    [0, 1, 0, 0, 0, 12, 11, 0, 7, 0, 0, 0],
                    [0, 12, 2, 0, 0, 0, 0, 7, 3, 0, 0, 0],
                    [0, 6, 0, 9, 0, 0, 0, 0, 0, 0, 0, 11],
                    [4, 0, 0, 0, 0, 0, 2, 0, 12, 0, 0, 3],
                    [8, 0, 0, 7, 5, 11, 0, 0, 0, 0, 0, 0]]
    grp5_12x12_3 = [[10, 0, 7, 11, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 9, 8, 10, 2, 0, 0, 0],
                    [0, 0, 4, 0, 7, 0, 3, 1, 0, 0, 8, 0],
                    [0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 10, 0],
                    [1, 9, 12, 0, 0, 4, 0, 0, 0, 0, 0, 7],
                    [0, 0, 0, 0, 11, 3, 12, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 6, 10, 0, 0, 0, 0, 0, 12],
                    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 9, 12, 0, 0, 0, 2]]
    all_12x12 = [grp1_12x12_1, grp1_12x12_2, grp1_12x12_3, grp2_12x12_1, grp2_12x12_2, grp2_12x12_3, grp3_12x12_1,
                 grp3_12x12_2, grp3_12x12_3, grp4_12x12_1, grp4_12x12_2, grp4_12x12_3, grp5_12x12_1, grp5_12x12_2,
                 grp5_12x12_3]

    # 9x9
    grp1_9x9_1 = [[0, 0, 0, 1, 0, 0, 0, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 8, 0, 2, 0, 0, 0],
                  [0, 6, 0, 0, 0, 5, 7, 9, 0],
                  [0, 0, 0, 2, 0, 0, 0, 0, 0],
                  [0, 0, 0, 9, 7, 1, 0, 0, 6],
                  [0, 0, 0, 0, 1, 0, 4, 0, 0],
                  [0, 2, 0, 0, 0, 0, 0, 0, 5],
                  [0, 0, 0, 5, 0, 0, 0, 6, 7]]
    grp1_9x9_2 = [[0, 0, 0, 2, 0, 8, 0, 0, 0],
                  [2, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 8, 0, 0],
                  [0, 0, 0, 5, 0, 0, 0, 0, 0],
                  [0, 0, 0, 3, 0, 1, 0, 0, 0],
                  [0, 1, 9, 0, 0, 0, 0, 0, 6],
                  [0, 0, 0, 0, 8, 0, 0, 0, 0],
                  [0, 0, 4, 1, 2, 0, 0, 0, 0],
                  [8, 0, 6, 9, 3, 0, 1, 0, 5]]
    grp1_9x9_3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [2, 0, 0, 4, 7, 9, 0, 0, 0],
                  [0, 0, 0, 1, 0, 3, 0, 5, 0],
                  [0, 0, 7, 2, 0, 0, 0, 8, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 6, 0, 0, 9, 4, 5, 2, 0],
                  [0, 0, 3, 0, 0, 7, 0, 0, 0],
                  [0, 0, 9, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 5, 0, 2, 0, 0]]
    grp2_9x9_1 = [[7, 2, 0, 0, 0, 4, 0, 1, 9],
                  [0, 0, 0, 0, 0, 0, 0, 0, 8],
                  [0, 8, 9, 1, 0, 7, 0, 0, 0],
                  [5, 3, 0, 0, 0, 0, 7, 4, 0],
                  [0, 4, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 7, 0, 0, 8, 0, 5],
                  [0, 7, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 7]]
    grp2_9x9_2 = [[5, 6, 0, 4, 9, 0, 0, 3, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 4, 0, 0],
                  [0, 0, 0, 0, 0, 4, 0, 0, 0],
                  [0, 3, 0, 0, 0, 6, 1, 4, 8],
                  [0, 0, 0, 0, 0, 7, 0, 2, 3],
                  [0, 0, 0, 8, 0, 9, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 9, 7, 0, 0, 8, 0, 0]]
    grp2_9x9_3 = [[5, 0, 0, 0, 0, 4, 0, 0, 0],
                  [0, 0, 0, 9, 0, 0, 0, 0, 2],
                  [0, 0, 3, 0, 1, 0, 4, 0, 0],
                  [0, 2, 0, 0, 7, 0, 0, 0, 0],
                  [0, 0, 0, 6, 9, 0, 0, 0, 0],
                  [9, 0, 0, 1, 0, 0, 0, 0, 0],
                  [4, 0, 0, 0, 5, 0, 9, 0, 0],
                  [0, 0, 5, 0, 4, 9, 0, 0, 3],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    grp3_9x9_1 = [[0, 0, 0, 0, 0, 0, 2, 0, 0],
                  [0, 5, 0, 0, 0, 0, 6, 0, 0],
                  [2, 0, 0, 0, 9, 0, 1, 4, 5],
                  [0, 0, 1, 0, 0, 0, 5, 0, 0],
                  [0, 0, 0, 9, 0, 0, 7, 0, 4],
                  [0, 0, 7, 0, 0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 1, 0, 0, 5, 0],
                  [0, 0, 0, 0, 0, 3, 0, 8, 0],
                  [5, 0, 3, 0, 0, 0, 0, 0, 0]]
    grp3_9x9_2 = [[0, 6, 8, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 6, 0, 0, 4, 0, 0],
                  [0, 0, 0, 7, 0, 0, 5, 0, 0],
                  [0, 0, 9, 0, 6, 0, 0, 0, 4],
                  [0, 0, 5, 0, 0, 0, 0, 2, 9],
                  [0, 0, 0, 0, 2, 0, 0, 0, 0],
                  [0, 8, 4, 0, 7, 6, 2, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 8, 0, 0, 4, 0]]
    grp3_9x9_3 = [[0, 2, 0, 0, 9, 0, 6, 0, 5],
                  [3, 0, 0, 2, 0, 0, 7, 0, 8],
                  [0, 0, 6, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 3, 0, 0, 0, 4],
                  [0, 4, 0, 0, 0, 0, 2, 0, 7],
                  [0, 0, 5, 0, 0, 0, 1, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 8, 4, 0, 2, 0, 0, 0, 0],
                  [0, 3, 0, 0, 0, 0, 0, 0, 0]]
    grp4_9x9_1 = [[4, 0, 0, 0, 0, 0, 8, 0, 5],
                  [0, 3, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 7, 0, 0, 0, 0, 0],
                  [0, 2, 0, 0, 0, 0, 0, 6, 0],
                  [0, 0, 0, 0, 8, 0, 4, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 6, 0, 3, 0, 7, 0],
                  [5, 0, 0, 2, 0, 0, 0, 0, 0],
                  [1, 0, 4, 0, 0, 0, 0, 0, 0]]
    grp4_9x9_2 = [[0, 0, 0, 0, 2, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 7, 0],
                  [0, 0, 0, 0, 0, 0, 8, 0, 2],
                  [3, 6, 0, 0, 0, 0, 1, 9, 4],
                  [0, 0, 0, 6, 0, 0, 7, 0, 0],
                  [0, 0, 0, 8, 0, 0, 0, 0, 0],
                  [0, 0, 0, 4, 0, 2, 0, 8, 0],
                  [5, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 9, 0, 6, 0, 3, 0]]
    grp4_9x9_3 = [[0, 0, 0, 5, 0, 3, 0, 0, 2],
                  [0, 0, 1, 0, 6, 8, 0, 4, 3],
                  [0, 0, 0, 0, 4, 0, 0, 0, 0],
                  [0, 0, 0, 0, 2, 0, 0, 0, 5],
                  [9, 0, 0, 0, 0, 0, 7, 0, 1],
                  [0, 8, 0, 9, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 5, 0],
                  [0, 0, 9, 0, 0, 0, 0, 8, 0],
                  [0, 0, 6, 0, 0, 0, 0, 3, 9]]
    grp5_9x9_1 = [[4, 0, 0, 0, 0, 0, 8, 0, 5],
                  [0, 3, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 7, 0, 0, 0, 0, 0],
                  [0, 2, 0, 0, 0, 0, 0, 6, 0],
                  [0, 0, 0, 0, 8, 0, 4, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 6, 0, 3, 0, 7, 0],
                  [5, 0, 0, 2, 0, 0, 0, 0, 0],
                  [1, 0, 4, 0, 0, 0, 0, 0, 0]]
    grp5_9x9_2 = [[0, 9, 2, 3, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 8, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 7, 0, 4, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 6, 5],
                  [8, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 6, 0, 5, 0, 2, 0, 0, 0],
                  [4, 0, 0, 0, 0, 0, 7, 0, 0],
                  [0, 0, 0, 9, 0, 0, 0, 0, 0]]
    grp5_9x9_3 = [[0, 2, 3, 7, 0, 0, 0, 0, 6],
                  [8, 0, 0, 0, 6, 0, 5, 9, 0],
                  [9, 0, 0, 0, 0, 0, 7, 0, 0],
                  [0, 0, 0, 0, 4, 0, 9, 7, 0],
                  [3, 0, 7, 0, 9, 6, 0, 0, 2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [5, 0, 0, 4, 7, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 2, 0, 0, 0],
                  [0, 8, 0, 0, 0, 0, 0, 0, 0]]
    all_9x9 = [grp1_9x9_1, grp1_9x9_2, grp1_9x9_3, grp2_9x9_1, grp2_9x9_2, grp2_9x9_3, grp3_9x9_1, grp3_9x9_2,
               grp3_9x9_3, grp4_9x9_1, grp4_9x9_2, grp4_9x9_3, grp5_9x9_1, grp5_9x9_2, grp5_9x9_3]

    restart_16 = all_25x25
    # for sets in all_puzzles:
    for puzzle in restart_16:
        csp = CSP(puzzle)

        csp.init_board()
        csp.init_binary_constraints()
        # print("Initial Board")
        # for i in csp.board:
        #     for j in i:
        #         print(j.row, j.col, j.domain)

        # preprocessing using ac3
        arcs = csp.init_constraints()
        csp.ac3(arcs)

        # print("After initial AC3 with all arcs")
        # for i in csp.board:
        #     for j in i:
        #         print(j.row, j.col, j.domain)
        start_time = time.time()
        print(f"{puzzle}")
        if csp.solve_csp_multiprocess():
            print(f"Solved in {time.time() - start_time} seconds")
        else:
            print(f"Failed to solve in {time.time() - start_time} seconds")
        # print("\nSolved Board")
        # for i in csp.board:
        #     for j in i:
        #         print(j.row, j.col, j.domain)


# def test_generate_sets(puzzle_data):
#     puzzle_size = len(puzzle_data)
#     row_set = [set() for _ in range(puzzle_size)]
#     col_set = [set() for _ in range(puzzle_size)]
#     sub_grid_set = [set() for _ in range(puzzle_size)]
#     for i, j in itertools.product(range(puzzle_size), range(puzzle_size)):
#         num = puzzle_data[i][j]
#         row_set[i].add(num)
#         col_set[j].add(num)
#         sub_grid_set[(i // int(math.sqrt(puzzle_size)))
#                      * int(math.sqrt(puzzle_size))
#                      + (j // int(math.ceil(math.sqrt(puzzle_size))))].add(num)

#     return row_set, col_set, sub_grid_set


if __name__ == "__main__":
    test()
