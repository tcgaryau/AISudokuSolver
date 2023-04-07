import math
import copy
import itertools


class BruteForce:
    """
    Brute force depth - first searching algorithm.
    """

    def __init__(self, puzzle_data, size_data, row_set, col_set, sub_grid_set):
        self.puzzle_data = copy.deepcopy(puzzle_data)
        self.row_set = copy.deepcopy(row_set)
        self.col_set = copy.deepcopy(col_set)
        self.sub_grid_set = copy.deepcopy(sub_grid_set)
        self.sg_row_total = 0
        self.sg_col_total = 0
        self.size_data = size_data
        self.current_limit = 0

        self.num_branch_fail = 0
        self.max_fail = size_data * size_data * 1000

    def solve(self, is_first=True) -> bool:
        """
        Brute force depth - first searching algorithm.
        In each node, find a valid number to empty square with the least number of options.
        A solution is found if every square on the board is filled.
        :param is_first: bool: if it is the first time to call the function
        :return: if a solution is found
        """

        self.sg_row_total = int(math.sqrt(self.size_data))
        self.sg_col_total = int(math.ceil(math.sqrt(self.size_data)))

        if empty_tuple := self.find_best_empty():
            row, col = empty_tuple
        else:
            return True

        if self.num_branch_fail >= self.max_fail:
            return False
        set_index = (row // self.sg_row_total) * self.sg_row_total + (col // self.sg_col_total)
        for num in self.get_available_numbers(row, col, set_index):

            # if we've reached the max_depth, reset the counter and continue
            if is_first and self.num_branch_fail >= self.max_fail:
                self.current_limit = self.num_branch_fail
                self.num_branch_fail = 0
                continue

            # add the number to the board and its associated sets
            self.puzzle_data[row][col] = num
            self.row_set[row].add(num)
            self.col_set[col].add(num)
            self.sub_grid_set[set_index].add(num)

            if self.solve(False):
                return True

            # remove the number from its associated sets
            self.row_set[row].remove(num)
            self.col_set[col].remove(num)
            self.sub_grid_set[set_index].remove(num)

            self.num_branch_fail += 1

        self.puzzle_data[row][col] = 0
        return False

    def find_next_empty(self):
        """
        Get the row and col number of the next empty square on the board.
        :return: a tuple
        """
        return next(
            (
                (row, col)
                for row, col in itertools.product(
                    range(self.size_data), range(self.size_data)
                )
                if self.puzzle_data[row][col] == 0
            ),
            None,
        )

    def find_best_empty(self):
        """
        Get the row and col number of the empty square with the least number of options.
        :return: a tuple or None
        """

        min_options = self.size_data + 1
        best_row = None
        best_col = None
        for row in range(self.size_data):
            for col in range(self.size_data):
                if self.puzzle_data[row][col] == 0:
                    set_index = (row // self.sg_row_total) * self.sg_row_total + (
                            col // self.sg_col_total)

                    num_options = len(self.get_available_numbers(row, col, set_index))
                    if num_options < min_options:
                        min_options = num_options
                        best_row = row
                        best_col = col
        return None if best_row is None else (best_row, best_col)

    def get_available_numbers(self, row, col, set_index):
        """
        Get a list of numbers that can be used to assign an empty square. An available number must
        not have been used by other squares in the same row, column, and sub grid of the current
        square.
        :param row: row number, int
        :param col: column number, int
        :param set_index: sub grid number, int
        :return: a list of int
        """
        used_nums = set(
            list(self.row_set[row]) + list(self.col_set[col]) + list(self.sub_grid_set[set_index]))
        all_possible_options = list(range(1, self.size_data + 1))
        return [num for num in all_possible_options if num not in used_nums]

    def return_board(self):
        """
        Returns solved puzzle data.
        :return: list of lists
        """
        return self.puzzle_data

    def increase_max_depth(self):
        """ Increase DFS search depth before fail. """
        self.max_fail = self.max_fail * 1.05
