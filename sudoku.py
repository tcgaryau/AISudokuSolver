import random
import time
from tkinter import *
from tkinter import filedialog
import os
import math
from brute_force import BruteForce

cells = {}


class SudokuBoard:

    def __init__(self):
        self.puzzle_data = []
        self.size_data = 0
        self.row_set = None
        self.col_set = None
        self.sub_grid_set = None

        self.main_frame = None
        self.main_height = None

        self.clicked = None

    def clear(self):
        for widgets in self.main_frame.winfo_children():
            widgets.destroy()
        cells.clear()

    def change_colour(self, colour):
        return "#ffffd0" if colour == "#D0ffff" else "#D0ffff"

    def draw_sub_grid(self, row_num, col_num, sub_row_num, sub_col_num, bgcolour, parent_frame):
        frame = Frame(parent_frame)

        # segment: use a canvas inside subframe
        # each square is at least of dimension 20
        sqr_w = max(self.main_height // sub_col_num / sub_row_num, 20)
        width = int(sqr_w * sub_col_num)
        height = int(sqr_w * sub_row_num)
        canvas = Canvas(frame, height=height, width=width,
                        bg=bgcolour, bd=0, highlightthickness=0)
        canvas.pack(fill=BOTH, expand=False)

        for i in range(0, sub_row_num):
            for j in range(0, sub_col_num):
                sqrW = width / sub_col_num
                sqrH = height / sub_row_num
                x = j * sqrW
                y = i * sqrH
                canvas.create_rectangle(
                    x, y, x + sqrW, y + sqrH, outline='black')
                if self.puzzle_data:
                    entry = self.puzzle_data[i + row_num][j + col_num] if \
                        self.puzzle_data[i + row_num][
                            j + col_num] != 0 else ""
                    canvas.create_text(x + sqrW / 2, y + sqrH / 2,
                                       text=f"{entry}",
                                       font=(None, f'{width // sub_col_num // 2}'), fill="black")
                else:
                    canvas.create_text(x + sqrW / 2, y + sqrH / 2,
                                       text=f"{int((j + col_num + 1))}",
                                       font=(None, f'{width // sub_col_num // 2}'), fill="red")
        # end of segment

        # for i in range(sub_row_num):
        #     for j in range(sub_col_num):
        #         label = Label(frame, width=2, text=col_num+j+1,
        #                       bg=bgcolour, justify="center", borderwidth=1, relief="solid")
        #         label.grid(row=i+1, column=j+1,
        #                    sticky="nsew", ipady=2)
        frame.grid(row=row_num + 1, column=col_num + 1)

    def draw_whole_grid(self, row, col, sub_row_num, sub_col_num):
        self.clear()
        colour = "#D0ffff"

        # add scroll bars to main frame when it shows a grid
        canvas = Canvas(self.main_frame, height=self.main_height, width=self.main_height)
        scrollbar_h = Scrollbar(self.main_frame, orient=HORIZONTAL, command=canvas.xview)
        scrollbar_v = Scrollbar(self.main_frame, orient=VERTICAL, command=canvas.yview)
        scrollbar_h.pack(side=BOTTOM, fill=X)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar_v.pack(side=RIGHT, fill=Y)
        canvas.configure(xscrollcommand=scrollbar_h.set)
        canvas.configure(yscrollcommand=scrollbar_v.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        inner_frame = Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor='nw')

        for row_num in range(0, row, sub_row_num):
            for col_num in range(0, col, sub_col_num):
                self.draw_sub_grid(row_num, col_num, sub_row_num, sub_col_num, colour, inner_frame)
                colour = self.change_colour(colour)
            if sub_row_num % 2 == 0:
                colour = self.change_colour(colour)

    def submit(self):
        match self.clicked.get():
            case "9x9":
                self.generate_board(9)
                self.draw_whole_grid(9, 9, 3, 3)
            case "12x12":
                self.generate_board(12)
                self.draw_whole_grid(12, 12, 3, 4)
            case "16x16":
                self.generate_board(16)
                self.draw_whole_grid(16, 16, 4, 4)
            case "25x25":
                self.generate_board(25)
                self.draw_whole_grid(25, 25, 5, 5)
            case "100x100":
                self.generate_board(100)
                self.draw_whole_grid(100, 100, 10, 10)
            case "Select Options":
                self.draw_whole_grid(self.size_data, self.size_data, int(
                    math.sqrt(self.size_data)), math.ceil(math.sqrt(self.size_data)))

    def browse_files(self):
        label_file_explorer = Label(self.main_frame,
                                    text="File Explorer using Tkinter",
                                    width=100, height=4,
                                    fg="blue")
        label_file_explorer.pack()
        filename = filedialog.askopenfilename(initialdir=os.getcwd,
                                              title="Select a File",
                                              filetypes=(("Text files",
                                                          "*.txt"),
                                                         ("all files",
                                                          "*.*")))

        # Change label contents
        label_file_explorer.configure(text=f"File Opened: {filename}")
        # try:
        #     with open(filename, "r") as f:
        #         file_data = f.read().strip()
        #         global puzzle_data, size_data
        #         puzzle_data, size_data = parse_input_file(file_data)
        # except Exception as e:
        #     label_file_explorer.destroy()
        #     print(e)
        with open(filename, "r") as f:
            file_data = f.read().strip()
            self.puzzle_data, self.size_data = self.parse_input_file(file_data)

    def generate_board(self, size):
        board = [[0 for _ in range(size)] for _ in range(size)]
        board_tiles = size * size
        required_tiles = board_tiles * 0.25
        tiles_placed = 0
        subgrid_size = int(math.sqrt(size))
        self.row_set = [set() for _ in range(size)]
        self.col_set = [set() for _ in range(size)]
        self.sub_grid_set = [set() for _ in range(size)]
        while tiles_placed < required_tiles:
            row = random.randint(0, size - 1)
            col = random.randint(0, size - 1)
            value_to_insert = random.randint(1, size)

            can_put_in_row = value_to_insert not in self.row_set[row]
            can_put_in_col = value_to_insert not in self.col_set[col]
            can_put_in_subgrid = value_to_insert not in \
                                 self.sub_grid_set[(row // subgrid_size) * subgrid_size +
                                                   (col // math.ceil(math.sqrt(size)))]
            if board[row][col] == 0 and can_put_in_row and can_put_in_col and can_put_in_subgrid:
                board[row][col] = value_to_insert
                self.row_set[row].add(value_to_insert)
                self.col_set[col].add(value_to_insert)
                self.sub_grid_set[
                    (row // subgrid_size) * subgrid_size + (col // math.ceil(math.sqrt(size)))].add(
                    value_to_insert)
                tiles_placed += 1
        self.puzzle_data = board
        self.size_data = len(self.puzzle_data)
        print(['' if num == 0 else num for row in self.puzzle_data for num in row])

    def parse_input_file(self, data):
        data = data.split(',')
        puzzle_size = int(math.sqrt(len(data)))
        data = [0 if num == "" else int(num) for num in data]
        rows_of_input = [
            data[row * puzzle_size: row * puzzle_size + puzzle_size]
            for row in range(puzzle_size)
        ]
        input_array = [[0 for _ in range(puzzle_size)]
                       for _ in range(puzzle_size)]
        self.row_set = [set() for _ in range(puzzle_size)]
        self.col_set = [set() for _ in range(puzzle_size)]
        self.sub_grid_set = [set() for _ in range(puzzle_size)]
        for row_num, row in enumerate(rows_of_input):
            for col_num, number in enumerate(row):
                input_array[row_num][col_num] = int(number)
                self.row_set[row_num].add(int(number))
                self.col_set[col_num].add(int(number))
                self.sub_grid_set[(row_num // int(math.sqrt(puzzle_size)))
                                  * int(math.sqrt(len(data)))
                                  + (col_num // int(math.ceil(math.sqrt(puzzle_size))))].add(
                    int(number))

            data = input_array
        return data, puzzle_size

    def drop_down_menu(self):
        options = [
            "9x9",
            "12x12",
            "16x16",
            "25x25",
            "100x100"
        ]
        btn_input_file = Button(self.main_frame, text="Choose File",
                                command=self.browse_files, width=15)
        btn_input_file.pack()
        self.clicked.set("Select Options")
        drop = OptionMenu(self.main_frame, self.clicked, *options)
        drop.pack()
        button = Button(self.main_frame, text="Submit", command=self.submit, width=15)
        button.pack()

    def display_message(self, msg):
        Label(self.main_frame, text=msg).grid(
            row=0, column=0, columnspan=5)

    def create_sudoku(self):
        self.clear()
        self.drop_down_menu()

    def on_click_solve_brute_force(self):
        self.clear()
        brute_force = BruteForce(self.puzzle_data, self.size_data, self.row_set, self.col_set,
                                 self.sub_grid_set)
        start = time.time()
        max_time = start + 300
        while time.time() < max_time:
            if brute_force.solve_brute_force():
                self.puzzle_data = brute_force.return_board()
                print(f"Brute Force took {time.time() - start} s")
                self.draw_whole_grid(self.size_data, self.size_data, int(
                    math.sqrt(self.size_data)), math.ceil(math.sqrt(self.size_data)))
                break
            elif brute_force.num_branch_fail < brute_force.max_fail:
                self.display_message("Invalid board. No solution found.")
                break
            brute_force.increase_max_depth()
        else:
            self.display_message("Timer ran out. No solution found.")

        # if brute_force.solve_brute_force():
        #     self.puzzle_data = brute_force.return_board()
        #     print(f"Brute Force took {time.time() - start} s")
        #     self.draw_whole_grid(self.size_data, self.size_data, int(
        #         math.sqrt(self.size_data)), math.ceil(math.sqrt(self.size_data)))
        # else:
        #     self.display_message("No solution found.")

    def solve_csp(self):
        if len(cells) == 0:
            return

    def main(self):
        root = Tk()
        root.geometry("900x900")
        # root.attributes('-fullscreen', True)
        root.title("AI Sudoku Solver")

        self.clicked = StringVar()

        self.main_frame = Frame(root)
        self.main_height = root.winfo_screenheight() - root.winfo_screenheight() * 0.25
        Label(self.main_frame, text="Welcome User").grid(
            row=0, column=0, columnspan=5)

        self.main_frame.pack(side=TOP)

        bottomFrame = Frame(root)

        bottomFrame.pack(side=BOTTOM, pady=20)
        btn_create = Button(bottomFrame, text="Create Sudoku",
                            command=self.create_sudoku, width=15)
        btn_create.grid(row=0, column=0)

        btn_solve_heuristic = Button(
            bottomFrame, text="Solve (heuristic)", command=self.on_click_solve_brute_force,
            width=15)
        btn_solve_heuristic.grid(row=0, column=4)

        btn_solve_csp = Button(bottomFrame, text="Solve (CSP)",
                               command=self.solve_csp, width=15)
        btn_solve_csp.grid(row=0, column=8)

        btn_clear = Button(bottomFrame, text="Clear", command=self.clear, width=15)
        btn_clear.grid(row=0, column=12)

        btn_exit = Button(bottomFrame, text="Exit", command=exit, width=15)
        btn_exit.grid(row=0, column=16)

        root.mainloop()


if __name__ == "__main__":
    sudoku = SudokuBoard()
    sudoku.main()
