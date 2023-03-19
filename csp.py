import math


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

    def __str__(self):
        return f"{self.row} {self.col} {self.domain} {self.neighbors}"


class CSP:

    def __init__(self, puzzle_data):
        self.puzzle_data = puzzle_data
        self.size_data = len(puzzle_data)
        self.board = None

    def init_board(self):
        """ Initialize a sudoku board as a 2D array of Squares. """
        size = self.size_data
        puzzle = self.puzzle_data
        board = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                value = puzzle[i][j]
                if value == 0:
                    board[i][j] = Square(i, j, set(range(1, size + 1)))
                else:
                    board[i][j] = Square(i, j, {value}, True)

        self.board = board

    def init_binary_constraints(self):
        """ Populate the neighbors field of every square on the current board. """
        board = self.board
        size = self.size_data
        for i in range(size):
            for j in range(size):
                curr_square = board[i][j]
                if not curr_square.assigned:
                    neighbors = set()

                    for n in range(size):
                        square = board[i][n]
                        if not square.assigned and n != j:
                            neighbors.add(square)

                    for m in range(size):
                        square = board[m][j]
                        if not square.assigned and m != i:
                            neighbors.add(square)

                    sg_row_total = int(math.sqrt(size))
                    sg_col_total = int(math.ceil(math.sqrt(size)))
                    shift_row = i // sg_row_total * sg_row_total
                    shift_col = j // sg_col_total * sg_col_total
                    for m in range(sg_row_total):
                        for n in range(sg_col_total):
                            square = board[m + shift_row][n + shift_col]
                            if not square.assigned and m != i and n != j:
                                neighbors.add(square)

                    curr_square.neighbors = neighbors

    def solve_csp(self):
        """ CSP algorithms. """
        if self.check_complete():
            return True

        # TODO: check MAC at beginning

        next_empty = self.select_unassigned()

        # TODO: order domain values: Least constraining value heuristic() -> values: []
        values = next_empty.domain

        for v in values:
            #   if is_consistent(v)
            #       assign v to next_empty
            #           inference = MAC()
            pass

    def check_complete(self):
        """
        Check if assignments are complete.
        :return: bool
        """
        complete = True
        for row in self.board:
            for square in row:
                if len(square.domain) != 1:
                    complete = False
                    break

        return complete

    def select_unassigned(self):
        """
        Select the best square to assign next using MRV and Degree heuristics.
        :return: a Square
        """
        mrv_squares = self.find_mrv()
        mrv_md_squares = self.find_max_degree(mrv_squares)
        return mrv_md_squares[0]

    def find_mrv(self):
        """
        Find the Squares with the Minimum Remaining Values
        :return: a list of Square
        """
        min_size = self.size_data
        mrv = []
        for row in self.board:
            for square in row:
                if not square.assigned:
                    length = len(square.domain)
                    if length < min_size:
                        mrv = [square]
                        min_size = length
                    elif len(square.domain) == min_size:
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
            degree = len(square.neighbors)
            if degree > max_degree:
                md = [square]
                max_degree = degree
            elif len(square.neighbors) == max_degree:
                md.append(square)
        return md
