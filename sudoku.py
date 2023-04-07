import random
import time
from tkinter import *
from tkinter import filedialog
import os
import math
from enum import Enum
from brute_force import BruteForce
from csp import CSP

cells = {}
btnFont = ("Californian FB", 15)


class SudokuBoard:
    """ Contains sudoku board's data and UI. """

    def __init__(self):
        self.timer_frame_2 = None
        self.root = None
        self.puzzle_data = []
        self.size_data = 0
        self.row_set = None
        self.col_set = None
        self.sub_grid_set = None
        self.selected_100x100 = False
        self.main_frame = None
        self.bottom_frame = None
        self.timer_frame = None
        self.main_height = None
        self.clicked = None

        self.puzzle_solution_1 = None
        self.puzzle_solution_2 = None

    def clear(self):
        """ Clear all widgets. """
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            cells.clear()

        if self.timer_frame is not None:
            for widget in self.timer_frame.winfo_children():
                widget.destroy()
                cells.clear()
            self.timer_frame = None

        if self.timer_frame_2 is not None:
            for widget in self.timer_frame_2.winfo_children():
                widget.destroy()
                cells.clear()
            self.timer_frame_2 = None

    def clear_data(self):
        """ Clear board data. """
        self.puzzle_data = []
        self.puzzle_solution_1 = None
        self.puzzle_solution_2 = None

    def change_colour(self, colour):
        """ Change colour of the board. """
        return "#ffffd0" if colour == "#D0ffff" else "#D0ffff"

    def draw_sub_grid(self, row_num, col_num, sub_row_num, sub_col_num, bgcolour, draw_list):
        """
        Draw one of the sub-grid of a sudoku board.
        :param row_num: number of rows in the board
        :param col_num: number of columns in the board
        :param sub_row_num: the row number of the subgrid
        :param sub_col_num: the column number of the subgrid
        :param bgcolour: background colour of the subgrid
        :param draw_list: a list of tuples, each tuple contains a frame and a solution display
        """
        for item in draw_list:
            frame = Frame(item[0])
            solution_display = item[1]
            puzzle_data = solution_display.puzzle_data

            # segment:use a canvas inside subframe
            sqr_w = max(self.main_height // sub_col_num / sub_row_num, 20)
            width = int(sqr_w * sub_col_num)
            height = int(sqr_w * sub_row_num)
            canvas = Canvas(frame, height=height, width=width,
                            bg=bgcolour, bd=0, highlightthickness=0)
            canvas.pack(fill=BOTH, expand=False)

            for i in range(sub_row_num):
                for j in range(sub_col_num):
                    sqrW = width / sub_col_num
                    sqrH = height / sub_row_num
                    x = j * sqrW
                    y = i * sqrH
                    canvas.create_rectangle(
                        x, y, x + sqrW, y + sqrH, outline='black')
                    if puzzle_data:
                        entry = puzzle_data[i + row_num][j + col_num] if \
                                puzzle_data[i + row_num][
                                j + col_num] != 0 else ""
                        canvas.create_text(x + sqrW / 2, y + sqrH / 2,
                                           text=f"{entry}",
                                           font=(None, f'{width // sub_col_num // 2}'),
                                           fill="black")
                    else:
                        canvas.create_text(x + sqrW / 2, y + sqrH / 2,
                                           text=f"{int((j + col_num + 1))}",
                                           font=(None, f'{width // sub_col_num // 2}'),
                                           fill="red")

            frame.grid(row=row_num + 1, column=col_num + 1)

    def draw_canvas(self, duo=False):
        """ Set up Canvas for each sudoku board. """
        if duo:
            canvas1 = Canvas(
                self.main_frame, height=self.main_height, width=self.main_height)
            canvas2 = Canvas(
                self.main_frame, height=self.main_height, width=self.main_height)

            def h_scroll(*args):
                """ Set up horizontal scrollbar. """
                canvas1.xview(*args)
                canvas2.xview(*args)

            def v_scroll(*args):
                """ Set up vertical scrollbar. """
                canvas1.yview(*args)
                canvas2.yview(*args)

            scrollbar_h = Scrollbar(
                self.main_frame, orient=HORIZONTAL, command=h_scroll)
            scrollbar_v = Scrollbar(
                self.main_frame, orient=VERTICAL, command=v_scroll)
            scrollbar_h.pack(side=BOTTOM, fill=X, expand=True)
            canvas1.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
            canvas2.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
            scrollbar_v.pack(side=RIGHT, fill=Y, expand=True)
            canvas1.configure(xscrollcommand=scrollbar_h.set)
            canvas1.configure(yscrollcommand=scrollbar_v.set)
            canvas2.configure(xscrollcommand=scrollbar_h.set)
            canvas2.configure(yscrollcommand=scrollbar_v.set)
            canvas1.bind('<Configure>', lambda e: canvas1.configure(
                scrollregion=canvas1.bbox('all')))
            canvas2.bind('<Configure>', lambda e: canvas2.configure(
                scrollregion=canvas2.bbox('all')))
            inner_frame1 = Frame(canvas1)
            inner_frame2 = Frame(canvas2)
            canvas1.create_window((0, 0), window=inner_frame1, anchor='nw')
            canvas2.create_window((0, 0), window=inner_frame2, anchor='nw')

            return inner_frame1, inner_frame2
        else:
            canvas = Canvas(self.main_frame,
                            height=self.main_height, width=self.main_height)
            scrollbar_h = Scrollbar(
                self.main_frame, orient=HORIZONTAL, command=canvas.xview)
            scrollbar_v = Scrollbar(
                self.main_frame, orient=VERTICAL, command=canvas.yview)
            scrollbar_h.pack(side=BOTTOM, fill=X)
            canvas.pack(side=LEFT, fill=BOTH, expand=True)
            scrollbar_v.pack(side=RIGHT, fill=Y)
            canvas.configure(xscrollcommand=scrollbar_h.set)
            canvas.configure(yscrollcommand=scrollbar_v.set)
            canvas.bind('<Configure>', lambda e: canvas.configure(
                scrollregion=canvas.bbox('all')))
            inner_frame = Frame(canvas)
            canvas.create_window((0, 0), window=inner_frame, anchor='nw')

            return inner_frame

    def draw_whole_grid(self, row, col, sub_row_num, sub_col_num):
        """
        Draw the entire sudoku board.
        :param row: number of rows
        :param col: number of columns
        :param sub_row_num: number of sub rows
        :param sub_col_num: number of sub columns
        """
        self.clear()
        colour = "#D0ffff"
        if self.puzzle_solution_1 and self.puzzle_solution_2:
            inner_frames = self.draw_canvas(True)
            draw_list = [[inner_frames[0], self.puzzle_solution_1], [
                inner_frames[1], self.puzzle_solution_2]]
        else:
            inner_frame = self.draw_canvas()
            solution = self.puzzle_solution_1 or self.puzzle_solution_2 or SolutionDisplay(
                self.puzzle_data, None, None)
            draw_list = [[inner_frame, solution]]

        for row_num in range(0, row, sub_row_num):
            for col_num in range(0, col, sub_col_num):
                self.draw_sub_grid(row_num, col_num, sub_row_num,
                                   sub_col_num, colour, draw_list)
                colour = self.change_colour(colour)
            if sub_row_num % 2 == 0:
                colour = self.change_colour(colour)

        if getattr(self.puzzle_solution_1, "time_cost", None) and \
                getattr(self.puzzle_solution_2, "time_cost", None):
            self.display_timer(self.puzzle_solution_1, position=1)
            self.display_timer(self.puzzle_solution_2, position=2)
        elif getattr(self.puzzle_solution_1, "time_cost", None):
            self.display_timer(self.puzzle_solution_1)
        elif getattr(self.puzzle_solution_2, "time_cost", None):
            self.display_timer(self.puzzle_solution_2)

    def display_timer(self, solution, position=0):
        """
        Display the time cost for the corresponding sudoku solver.
        :param solution: SolutionDisplay
        :param position: 0 to display in the center, 1 left, 2 right
        """
        timer_text = f"{solution.solver_type.value} took {solution.time_cost:.5f} seconds."
        timer_frame = Frame(self.root)
        if position == 1:
            self.timer_frame = timer_frame
            relx = 0.3
        elif position == 2:
            self.timer_frame_2 = timer_frame
            relx = 0.7
        else:
            self.timer_frame = timer_frame
            relx = 0.5

        timer_label = Label(timer_frame, text=timer_text, font=("Arial", 24))
        timer_label.pack(side=BOTTOM)
        timer_frame.place(relx=relx, y=self.main_height + 100, anchor='s')

    def submit(self):
        """
        Display initial sudoku board according to user's selection.
        """
        self.toggle_button('!button2', True)
        self.toggle_button('!button3', True)
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
                self.toggle_button('!button2', False)
                self.draw_whole_grid(100, 100, 10, 10)
            case "Select Options" | "From File":
                self.draw_whole_grid(self.size_data, self.size_data, int(
                    math.sqrt(self.size_data)), math.ceil(math.sqrt(self.size_data)))

    def browse_files(self):
        """
        Accept and read a .txt file from user input with sudoku board data;
        Use the data to initialize the board.
        """
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
        file_extension = filename.split(".")[-1]
        if file_extension != "txt":
            self.clear()
            self.display_message("Your file extension must be txt. Please try again.")
            return

        with open(filename, "r") as f:
            file_data = f.read().strip()
            if self.validate_file_contents(file_data):
                self.puzzle_data, self.size_data = self.parse_input_file(file_data)
            self.validate_file_contents(file_data)
            self.puzzle_data, self.size_data = self.parse_input_file(file_data)

    def validate_file_contents(self, file_data):
        """
        Validate the contents of the file to ensure that it is a valid sudoku board.
        """
        puzzle_size = len(file_data.split('\n'))
        for row in file_data.split('\n'):
            nums_in_row = sum(1 for char in row.split(',') if char.isdigit())
            if nums_in_row != puzzle_size:
                self.clear()
                self.display_message("Invalid file contents.")
                return False
        return True

    def generate_board(self, size):
        """
        Generate an initial sudoku board randomly that is 25% filled.
        :param size: int representing the edge size of the board
        """
        board = [[0 for _ in range(size)] for _ in range(size)]
        board_tiles = size * size
        required_tiles = board_tiles * 0.25
        tiles_placed = 0
        subgrid_size = int(math.sqrt(size))
        self.row_set = [set() for _ in range(size)]
        self.col_set = [set() for _ in range(size)]
        self.sub_grid_set = [set() for _ in range(size)]

        # Add numbers into the board until we hit the limit of 25% filled
        while tiles_placed < required_tiles:
            row = random.randint(0, size - 1)
            col = random.randint(0, size - 1)
            value_to_insert = random.randint(1, size)
            sub_grid_index = (row // subgrid_size) * subgrid_size + \
                             (col // math.ceil(math.sqrt(size)))

            can_put_in_row = value_to_insert not in self.row_set[row]
            can_put_in_col = value_to_insert not in self.col_set[col]
            can_put_in_subgrid = value_to_insert not in self.sub_grid_set[sub_grid_index]
            if board[row][col] == 0 and can_put_in_row and can_put_in_col and can_put_in_subgrid:
                board[row][col] = value_to_insert
                self.row_set[row].add(value_to_insert)
                self.col_set[col].add(value_to_insert)
                self.sub_grid_set[sub_grid_index].add(value_to_insert)
                tiles_placed += 1
        self.puzzle_data = board
        self.size_data = len(self.puzzle_data)

    def parse_input_file(self, data):
        """
        Parse sudoku board data read from a file.
        param data: string of puzzle data
        """
        data = data.split("\n")
        puzzle_size = len(data)

        data = [int(num) for row in data for num in row.split(',')]

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
                sg_index = (row_num // int(math.sqrt(puzzle_size))) * int(math.sqrt(len(data))) + \
                           (col_num // int(math.ceil(math.sqrt(puzzle_size))))
                self.sub_grid_set[sg_index].add(int(number))
            data = input_array
        return data, puzzle_size

    def drop_down_menu(self):
        """ Set up dropdown menu for sudoku size. """
        options = [
            "From File",
            "9x9",
            "12x12",
            "16x16",
            "25x25",
            "100x100"
        ]
        btn_input_file = Button(self.main_frame, text="Choose File", font=btnFont,
                                command=self.browse_files, width=15, height=2)
        btn_input_file.pack()
        self.clicked.set("Select Options")
        drop = OptionMenu(self.main_frame, self.clicked, *options)
        drop.config(font=btnFont, width=13, height=2)
        drop.pack()
        button = Button(self.main_frame, text="Submit",
                        font=btnFont, command=self.submit, width=15, height=2)
        button.pack()

    def display_message(self, msg):
        """
        Display a message on the GUI.
        :param msg: string to display
        """
        Label(self.main_frame, text=msg).grid(row=0, column=0, columnspan=5)

    def create_sudoku(self):
        """ On click 'Create Sudoku' button. """
        self.clear_data()
        self.clear()
        self.drop_down_menu()
        self.toggle_button("!button2", False)
        self.toggle_button("!button3", False)

    def on_click_solve_brute_force(self):
        """ Solve sudoku using brute force algorithm. """
        self.clear()
        self.toggle_button("!button2", False)
        brute_force = BruteForce(self.puzzle_data, self.size_data,
                                 self.row_set, self.col_set, self.sub_grid_set)
        self.solve_puzzle(brute_force, SolverType.BF)

    def on_click_solve_csp(self):
        """ Solve sudoku using CSP algorithm. """
        self.clear()
        self.toggle_button("!button3", False)
        csp = CSP(self.puzzle_data)
        self.solve_puzzle(csp, SolverType.CSP)

    def on_click_clear(self):
        """ Reset data and UI. """
        self.clear()
        self.clear_data()
        self.toggle_button('!button2', False)
        self.toggle_button('!button3', False)

    def toggle_button(self, button_id, enable):
        """
        Toggles the button state between active and disabled
        param: str button_id
        param: boolean enable
        """
        state = "active" if enable else "disabled"
        self.bottom_frame.children[button_id].configure(state=state)

    def solve_puzzle(self, solver, mode):
        """
        Solve sudoku puzzle with the given solver.
        param solver: BruteForce or CSP object
        mode: BF or CSP enum
        """
        start = time.perf_counter()

        while time.perf_counter() < start + 300:
            if solver.solve():
                solved_puzzle = solver.return_board()
                solution = SolutionDisplay(solved_puzzle, time.perf_counter() - start, mode)

                if self.puzzle_solution_1:
                    self.puzzle_solution_2 = solution
                else:
                    self.puzzle_solution_1 = solution

                self.draw_whole_grid(self.size_data, self.size_data, int(
                    math.sqrt(self.size_data)), math.ceil(math.sqrt(self.size_data)))
                break
            elif mode is SolverType.BF and solver.current_limit < solver.max_fail:
                self.display_message(
                    "This is an invalid board that has no solution.")
                break
            elif mode is SolverType.CSP:
                self.display_message(
                    "This is an invalid board that has no solution.")
                break
            if mode is SolverType.BF and solver.max_fail < solver.current_limit:
                print("Increasing the depth from", solver.max_fail)
                solver.increase_max_depth()
        else:
            self.display_message("Timer ran out. No solution found.")

    def gui(self):
        """ Creates the GUI. """
        self.root = Tk()
        self.root.geometry("900x900")
        self.root.attributes('-fullscreen', True)
        self.root.title("AI Sudoku Solver")

        self.clicked = StringVar()

        self.main_frame = Frame(self.root)
        self.main_height = min(self.root.winfo_screenheight(
        ) * 0.75, self.root.winfo_screenwidth() * 0.375)
        Label(self.main_frame, text="Welcome User", font=("Arial", 24)).grid(
            row=0, column=0, columnspan=5, ipady=10)

        self.main_frame.pack(side=TOP, pady=20)

        self.bottom_frame = Frame(self.root)

        self.bottom_frame.pack(side=BOTTOM, pady=20)
        btn_create = Button(self.bottom_frame, text="Create Sudoku", font=btnFont,
                            command=self.create_sudoku, width=15, height=2)
        btn_create.grid(row=0, column=0)

        btn_solve_heuristic = Button(
            self.bottom_frame, text="Solve (heuristic)", font=btnFont,
            command=self.on_click_solve_brute_force,
            width=15, height=2, state="disabled")
        btn_solve_heuristic.grid(row=0, column=4)

        btn_solve_csp = Button(self.bottom_frame, text="Solve (CSP)", font=btnFont,
                               command=self.on_click_solve_csp, width=15, height=2,
                               state="disabled")
        btn_solve_csp.grid(row=0, column=8)

        btn_clear = Button(self.bottom_frame, text="Clear", font=btnFont,
                           command=self.on_click_clear, width=15,
                           height=2)
        btn_clear.grid(row=0, column=12)

        btn_exit = Button(self.bottom_frame, text="Exit",
                          font=btnFont, command=exit, width=15, height=2)
        btn_exit.grid(row=0, column=16)

        self.root.mainloop()


class SolverType(Enum):
    """
    Type of a sudoku solver.
    """
    BF = "Brute Force"
    CSP = "CSP"


class SolutionDisplay:
    """ Contains the solution, time cost and solver used for a solved puzzle. """

    def __init__(self, puzzle_data, time_cost, solver_type):
        self.puzzle_data = puzzle_data
        self.time_cost = time_cost
        self.solver_type = solver_type


if __name__ == "__main__":
    sudoku = SudokuBoard()
    sudoku.gui()
