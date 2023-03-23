import math
import copy
import itertools
import time


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
        self.neighbors = None
        # TODO: add a subgrid store

    def __str__(self):
        return f"Square {self.row} {self.col} domain: {self.domain} length of neighbors: {len(self.neighbors)}"


class Arc:
    """
    Represent an arc between two squares.
    Each arc has a start square and an end square.
    """

    def __init__(self, square1, square2):
        self.square1 = square1
        self.square2 = square2


class CSP:
    def __init__(self, puzzle_data):
        self.puzzle_data = copy.deepcopy(puzzle_data)
        # self.row_set = copy.deepcopy(row_set)
        # self.col_set = copy.deepcopy(col_set)
        # self.sub_grid_set = copy.deepcopy(sub_grid_set)
        self.size_data = len(puzzle_data)
        self.board = None
        self.unassigned = set()
        # self.arcs = set()

    def init_board(self):
        """ Initialize a sudoku board as a 2D array of Squares, and the domain of each square. """
        size = self.size_data
        puzzle = self.puzzle_data
        board = [[None for _ in range(size)] for _ in range(size)]
        for i, j in itertools.product(range(size), range(size)):
            value = puzzle[i][j]
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
        board = self.board
        size = self.size_data
        for i, j in itertools.product(range(size), range(size)):
            square = board[i][j]
            for arc in self._get_arcs(square):
                if arc.square1 != arc.square2:
                    arcs.add(arc)
        return arcs

    def _get_neighbors(self, square):
        return square.neighbors

    def _get_arcs(self, square):
        neighbors = self._get_neighbors(square)
        arcs = [Arc(neighbor, square) for neighbor in neighbors]
        return arcs

    # def _get_consistent_values(self, row, col):
    #     """
    #     Get values for a squares that are consistent with existing assignments.
    #     :param row: a square's row number, int
    #     :param col: a square's column number, int
    #     :return: a set of consistent values
    #     """
    #     size = self.size_data
    #     sg_row_total = int(math.sqrt(size))
    #     sg_col_total = int(math.ceil(math.sqrt(size)))
    #     set_index = (row // sg_row_total) * \
    #         sg_col_total + (col // sg_col_total)
    #     return set(range(1, size + 1)) - set(
    #         list(self.row_set[row])
    #         + list(self.col_set[col])
    #         + list(self.sub_grid_set[set_index])
    #     )

    def init_binary_constraints(self):
        """ Populate the neighbors field of every square on the current board. """
        board = self.board
        size = self.size_data
        sg_row_total = int(math.sqrt(size))
        sg_col_total = int(math.ceil(math.sqrt(size)))

        for i, j in itertools.product(range(size), range(size)):
            curr_square = board[i][j]
            # for curr_square in self.unassigned:
            row = curr_square.row
            col = curr_square.col
            neighbors = set()

            for n in range(size):
                square = board[row][n]
                if n != col:
                    neighbors.add(square)

            for m in range(size):
                square = board[m][col]
                if m != row:
                    neighbors.add(square)

            shift_row = row // sg_row_total * sg_row_total
            shift_col = col // sg_col_total * sg_col_total
            for m, n in itertools.product(range(sg_row_total), range(sg_col_total)):
                square = board[m + shift_row][n + shift_col]
                if m + shift_row != row and n + shift_col != col:
                    neighbors.add(square)
            curr_square.neighbors = neighbors

    def solve_csp(self):
        """ CSP algorithms. """

        # Return true if all squares have been assigned a value
        if len(self.unassigned) == 0:
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
                if is_inference:
                    if self.solve_csp():
                        return True
                for square, value in revised_list:
                    square.domain.append(value)

                # Remove the value from assignment
                next_empty.domain = saved_values
                next_empty.assigned = False
                self.unassigned.add(next_empty)

        return False

    def is_consistent(self, next_empty, v):
        return not any(
            len(neighbor.domain) == 1 and v in neighbor.domain
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
                [neighbor for neighbor in square.neighbors if not neighbor.assigned])
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
        arcs = set()
        # We start with only the arcs (Xj, Xi) for all Xj that are unassigned variables that are neighbors of Xi
        for neighbor in square.neighbors:
            if not neighbor.assigned:
                arcs.add(Arc(neighbor, square))

        return self.ac3(arcs)

    def ac3(self, arcs):
        """
        Check if the domain of a square is consistent with its neighbors.
        :param square: a Square
        :return: a boolean
        """
        revised_list = []
        while len(arcs) > 0:
            arc = arcs.pop()
            # is_revised, revised_list = self.revise(arc, revised_list)

            if self.revise(arc, revised_list):
                if len(arc.square1.domain) == 0:
                    # print("Removed", len(revised_list), "values")
                    # print("failure from", arc.square1.row, arc.square1.col)
                    return False, revised_list
                for neighbor in arc.square1.neighbors:
                    if neighbor is not arc.square2:
                        arcs.add(Arc(neighbor, arc.square1))
        return True, revised_list

    def revise(self, arc, revised_list):
        """
        Check if the domain of a square is consistent with its neighbors.
        :param arc: an Arc
        :return: a boolean
        """
        if len(arc.square2.domain) > 1:
            return False

        for x in arc.square1.domain:
            # Xi's domain = {1 2 3}, Xj's domain = {1}
            if x in arc.square2.domain:
                # print("X", x, "from square", arc.square1.row, arc.square1.col, "because square",
                #       arc.square2.row, arc.square2.col, "has domain", arc.square2.domain)
                arc.square1.domain.remove(x)
                revised_list.append((arc.square1, x))
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
            for neighbour in [neighbor for neighbor in square.neighbors if not neighbor.assigned]:
                if val in neighbour.domain:
                    neighbour_frequency[val] += 1

        # min_val = None
        # for key, value in neighbour_frequency.items():
        #     if min_val is None or value < neighbour_frequency[min_val]:
        #         min_val = key
        return sorted(neighbour_frequency, key=neighbour_frequency.get)


def test():
    # puzzle = [
    #     [0, 0, 0, 5, 0, 3, 0, 0, 2],
    #     [0, 0, 1, 0, 6, 8, 0, 4, 3],
    #     [0, 0, 0, 0, 4, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 2, 0, 0, 0, 5],
    #     [9, 0, 0, 0, 0, 0, 7, 0, 1],
    #     [0, 8, 0, 9, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 5, 0],
    #     [0, 0, 9, 0, 0, 0, 0, 8, 0],
    #     [0, 0, 6, 0, 0, 0, 0, 3, 9]
    # ]
    puzzle = [
        [4, 0, 1, 0, 0, 0, 6, 0, 0],
        [0, 9, 0, 3, 0, 6, 0, 5, 0],
        [0, 0, 0, 0, 9, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 0, 1, 0, 9, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 8, 0, 5, 0, 7, 0, 6, 0],
        [1, 0, 3, 0, 0, 0, 7, 0, 2]
    ]
    # puzzle = [
    #     [4, 1, 7, 3, 6, 9, 8, 0, 5],
    #     [0, 3, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 7, 0, 0, 0, 0, 0],
    #     [0, 2, 0, 0, 0, 0, 0, 6, 0],
    #     [0, 0, 0, 0, 8, 0, 4, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 6, 0, 3, 0, 7, 0],
    #     [5, 0, 0, 2, 0, 0, 0, 0, 0],
    #     [1, 0, 4, 0, 7, 5, 2, 9, 3],
    # ]
    # puzzle = [
    #     [1, 4, 3, 7, 2, 8, 9, 5, 0],
    #     [9, 0, 0, 3, 0, 5, 0, 0, 1],
    #     [0, 0, 1, 8, 0, 6, 4, 0, 0],
    #     [0, 0, 8, 1, 0, 2, 9, 0, 0],
    #     [7, 0, 0, 0, 0, 0, 0, 0, 8],
    #     [0, 0, 6, 7, 0, 8, 2, 0, 0],
    #     [0, 0, 2, 6, 0, 9, 5, 0, 0],
    #     [8, 0, 0, 2, 0, 3, 0, 0, 9],
    #     [0, 0, 5, 0, 1, 0, 3, 0, 0]
    # ]

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

    test_puzzle = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # row_set, col_set, sub_grid_set = test_generate_sets(puzzle)
    import time
    now = time.time()
    # csp = CSP(puzzle, row_set, col_set, sub_grid_set)
    csp = CSP(test_puzzle)
    csp.init_board()
    csp.init_binary_constraints()
    print("Initial Board")
    for i in csp.board:
        for j in i:
            print(j.row, j.col, j.domain)

    # preprocessing using ac3
    arcs = csp.init_constraints()
    csp.ac3(arcs)

    # print("After initial AC3 with all arcs")
    # for i in csp.board:
    #     for j in i:
    #         print(j.row, j.col, j.domain)
    start_time = time.time()
    if final_result := csp.solve_csp():
        print(f"Solved in {time.time() - start_time} seconds")
    else:
        print(f"Failed to solve in {time.time() - start_time} seconds")
    print("\nSolved Board")
    for i in csp.board:
        for j in i:
            print(j.row, j.col, j.domain)


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
