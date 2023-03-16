import math


class BruteForce:

    def __init__(self, puzzle_data, size_data, row_set, col_set, sub_grid_set):
        self.puzzle_data = puzzle_data
        self.row_set = row_set
        self.col_set = col_set
        self.sub_grid_set = sub_grid_set
        self.sg_row_total = 0
        self.sg_col_total = 0
        self.size_data = size_data

    def solve_brute_force(self) -> bool:
        """
        Brute force depth-first searching algorithm.
        In each node, find a valid number to fill the most top-left empty square.
        A solution is found if every square on the board is filled.
        :return: if a solution is found
        """
        self.sg_row_total = int(math.sqrt(self.size_data))
        self.sg_col_total = int(math.ceil(math.sqrt(self.size_data)))
        if empty_tuple := self.find_next_empty():
            row, col = empty_tuple

        else:
            return True
        set_index = (row // self.sg_row_total) * self.sg_col_total + (col // self.sg_col_total)
        for num in self.get_available_numbers(row, col, set_index):
            # if check_valid(num, row, col):
            self.puzzle_data[row][col] = num
            self.row_set[row].add(num)
            self.col_set[col].add(num)

            # print("index", set_index, "adding", num, "to", sub_grid_set[set_index])
            self.sub_grid_set[set_index].add(num)

            if self.solve_brute_force():
                return True
            self.row_set[row].remove(num)
            self.col_set[col].remove(num)
            # print("Backtracking from set #", set_index, ": ", sub_grid_set[set_index])
            self.sub_grid_set[set_index].remove(num)
        self.puzzle_data[row][col] = 0
        return False

    def find_next_empty(self):
        """
        Get the row and col number of the next empty square on the board.
        :return: a tuple
        """
        for row in range(self.size_data):
            for col in range(self.size_data):
                if self.puzzle_data[row][col] == 0:
                    return row, col
        return None

    def get_available_numbers(self, row, col, set_index):
        used_nums = set(
            list(self.row_set[row]) + list(self.col_set[col]) + list(self.sub_grid_set[set_index]))
        all_possible_options = list(range(1, self.size_data + 1))
        return [num for num in all_possible_options if num not in used_nums]

    def check_valid(self, num, row, col) -> bool:
        """
        Check if num can be assigned at (row, col) on the board.
        :return: if the assignment is valid
        """

        if num in self.row_set[row] or num in self.col_set[col]:
            return False
        if num in self.sub_grid_set[
            ((row // self.sg_row_total) * self.sg_row_total + (col // self.sg_col_total))]:
            return False

        return True

    def return_board(self):
        # print(self.puzzle_data)
        return self.puzzle_data
