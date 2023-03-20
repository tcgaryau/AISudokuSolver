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
        self.unassigned = set()

    def init_board(self):
        """ Initialize a sudoku board as a 2D array of Squares, and the domain of each square. """
        size = self.size_data
        puzzle = self.puzzle_data
        board = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                value = puzzle[i][j]
                if value == 0:
                    square = Square(i, j, set(range(1, size + 1)))
                    board[i][j] = square
                    self.unassigned.add(square)
                else:
                    board[i][j] = Square(i, j, {value}, True)

        self.board = board

    def init_binary_constraints(self):
        """ Populate the neighbors field of every square on the current board. """
        board = self.board
        size = self.size_data
        sg_row_total = int(math.sqrt(size))
        sg_col_total = int(math.ceil(math.sqrt(size)))

        for curr_square in self.unassigned:
            row = curr_square.row
            col = curr_square.col
            neighbors = set()

            for n in range(size):
                square = board[row][n]
                if not square.assigned and n != col:
                    neighbors.add(square)

            for m in range(size):
                square = board[m][col]
                if not square.assigned and m != row:
                    neighbors.add(square)

            shift_row = row // sg_row_total * sg_row_total
            shift_col = col // sg_col_total * sg_col_total
            for m in range(sg_row_total):
                for n in range(sg_col_total):
                    square = board[m + shift_row][n + shift_col]
                    if not square.assigned and m != row and n != col:
                        neighbors.add(square)

            curr_square.neighbors = neighbors

    def solve_csp(self):
        """ CSP algorithms. """

        # Return true if all squares have been assigned a value
        if len(self.unassigned) == 0:
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


def test():
    puzzle = [
        [5, 0, 1, 0, 0, 0, 6, 0, 4],
        [0, 9, 0, 3, 0, 6, 0, 5, 0],
        [0, 0, 0, 0, 9, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 0, 1, 0, 9, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 6],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 8, 0, 5, 0, 7, 0, 6, 0],
        [1, 0, 3, 0, 0, 0, 7, 0, 2]
    ]
    csp = CSP(puzzle)
    csp.init_board()
    csp.init_binary_constraints()
    csp.solve_csp()


if __name__ == "__main__":
    test()
