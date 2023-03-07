import random
import time
from tkinter import *
from tkinter import filedialog
import os
import math

root = Tk()
root.geometry("900x900")
# root.attributes('-fullscreen', True)
root.title("AI Sudoku Solver")

main_frame = Frame(root)
main_height = root.winfo_screenheight() - root.winfo_screenheight() * 0.25
Label(main_frame, text="Welcome User").grid(
    row=0, column=0, columnspan=5)

cells = {}

options = [
    "9x9",
    "12x12",
    "16x16",
    "25x25",
    "100x100"
]

clicked = StringVar()
puzzle_data = None
size_data = 0
row_set = None
col_set = None
sub_grid_set = None
sg_row_total = 0
sg_col_total = 0


def clear():
    for widgets in main_frame.winfo_children():
        widgets.destroy()
    cells.clear()


def change_colour(colour):
    if colour == "#D0ffff":
        return "#ffffd0"
    return "#D0ffff"


def draw_sub_grid(row_num, col_num, sub_row_num, sub_col_num, bgcolour, parent_frame):
    frame = Frame(parent_frame)

    # segment: use a canvas inside subframe
    # each square is at least of dimension 20
    sqr_w = max(main_height // sub_col_num / sub_row_num, 20)
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
            if puzzle_data:
                entry = puzzle_data[i + row_num][j + col_num] if puzzle_data[i + row_num][
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


def draw_whole_grid(row, col, sub_row_num, sub_col_num):
    clear()
    colour = "#D0ffff"

    # add scroll bars to main frame when it shows a grid
    canvas = Canvas(main_frame, height=main_height, width=main_height)
    scrollbar_h = Scrollbar(main_frame, orient=HORIZONTAL, command=canvas.xview)
    scrollbar_v = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
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
            draw_sub_grid(row_num, col_num, sub_row_num, sub_col_num, colour, inner_frame)
            colour = change_colour(colour)
        if sub_row_num % 2 == 0:
            colour = change_colour(colour)


def submit():
    match clicked.get():
        case "9x9":
            generate_board(9)
            draw_whole_grid(9, 9, 3, 3)
        case "12x12":
            generate_board(12)
            draw_whole_grid(12, 12, 3, 4)
        case "16x16":
            generate_board(16)
            draw_whole_grid(16, 16, 4, 4)
        case "25x25":
            generate_board(25)
            draw_whole_grid(25, 25, 5, 5)
        case "100x100":
            generate_board(100)
            draw_whole_grid(100, 100, 10, 10)
        case "Select Options":
            draw_whole_grid(size_data, size_data, int(
                math.sqrt(size_data)), math.ceil(math.sqrt(size_data)))


def browse_files():
    label_file_explorer = Label(main_frame,
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
    label_file_explorer.configure(text="File Opened: " + filename)
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
        global puzzle_data, size_data
        puzzle_data, size_data = parse_input_file(file_data)


def generate_board(size):
    global row_set, col_set, sub_grid_set
    board = [[0 for _ in range(size)] for _ in range(size)]
    board_tiles = size * size
    required_tiles = board_tiles * 0.25
    tiles_placed = 0
    subgrid_size = int(math.sqrt(size))
    row_set = [set() for _ in range(size)]
    col_set = [set() for _ in range(size)]
    sub_grid_set = [set() for _ in range(size)]
    while tiles_placed < required_tiles:
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        value_to_insert = random.randint(1, size)

        can_put_in_row = value_to_insert not in row_set[row]
        can_put_in_col = value_to_insert not in col_set[col]
        can_put_in_subgrid = value_to_insert not in \
                             sub_grid_set[(row // subgrid_size) * subgrid_size +
                                          (col // math.ceil(math.sqrt(size)))]
        if board[row][col] == 0 and can_put_in_row and can_put_in_col and can_put_in_subgrid:
            board[row][col] = value_to_insert
            row_set[row].add(value_to_insert)
            col_set[col].add(value_to_insert)
            sub_grid_set[
                (row // subgrid_size) * subgrid_size + (col // math.ceil(math.sqrt(size)))].add(
                value_to_insert)
            tiles_placed += 1
    global puzzle_data
    puzzle_data = board
    print(['' if num == 0 else num for row in puzzle_data for num in row])


def solve_brute_force() -> bool:
    """
    Brute force depth-first searching algorithm.
    In each node, find a valid number to fill the most top-left empty square.
    A solution is found if every square on the board is filled.
    :return: if a solution is found
    """
    global puzzle_data, row_set, col_set, sg_row_total, sg_col_total, sub_grid_set
    sg_row_total = int(math.sqrt(size_data))
    sg_col_total = int(math.ceil(math.sqrt(size_data)))
    empty = find_next_empty()
    if not empty:
        return True
    else:
        row, col = empty

    set_index = (row // sg_row_total) * sg_col_total + (col // sg_col_total)
    for num in get_available_numbers(row, col, set_index):
        # if check_valid(num, row, col):
        puzzle_data[row][col] = num
        row_set[row].add(num)
        col_set[col].add(num)

        # print("index", set_index, "adding", num, "to", sub_grid_set[set_index])
        sub_grid_set[set_index].add(num)

        if solve_brute_force():
            return True
        else:
            row_set[row].remove(num)
            col_set[col].remove(num)
            # print("Backtracking from set #", set_index, ": ", sub_grid_set[set_index])
            sub_grid_set[set_index].remove(num)
            # print("result", sub_grid_set[set_index])

    puzzle_data[row][col] = 0
    return False


def get_available_numbers(row, col, set_index):
    global row_set, col_set, sg_row_total, sg_col_total, sub_grid_set
    used_nums = set(list(row_set[row]) + list(col_set[col]) + list(sub_grid_set[set_index]))
    all_possible_options = list(range(1, size_data+1))
    remaining_options = []
    for num in all_possible_options:
        if num not in used_nums:
            remaining_options.append(num)
    return remaining_options


def find_next_empty():
    """
    Get the row and col number of the next empty square on the board.
    :return: a tuple
    """
    global size_data, puzzle_data
    for row in range(size_data):
        for col in range(size_data):
            if puzzle_data[row][col] == 0:
                return row, col
    return None


def check_valid(num, row, col) -> bool:
    """
    Check if num can be assigned at (row, col) on the board.
    :return: if the assignment is valid
    """
    # for i in range(size_data):
    #     if puzzle_data[row][i] == num:
    #         return False
    #
    # for i in range(size_data):
    #     if puzzle_data[i][col] == num:
    #         return False
    # print("row set", row, row_set[row])
    # print("col set", col, col_set[col])
    # print("sub grid set", (row // row_total) * row_total + (col // col_total),
    #       sub_grid_set[(row // row_total) * row_total + (col // col_total)])

    if num in row_set[row] or num in col_set[col]:
        return False

    # sg_row_total = int(math.sqrt(size_data))
    # sg_col_total = int(math.ceil(math.sqrt(size_data)))
    # shift_row = row // sg_row_total * sg_row_total
    # shift_col = col // sg_col_total * sg_col_total
    # for i in range(sg_row_total):
    #     for j in range(sg_col_total):
    #         if num == puzzle_data[i + shift_row][j + shift_col]:
    #             return False
    if num in sub_grid_set[((row // sg_row_total) * sg_row_total + (col // sg_col_total))]:
        return False

    return True

#
# def convert_from_dot_to_number(data):
#     row_len = int(math.sqrt(len(data)))
#     input_array = [[int(data[(i * 9 + j)]) if data[(i * 9 + j)]
#                                               != "." else 0 for j in range(row_len)] for i in
#                    range(row_len)]
#     return input_array, row_len

#
# def parse_input_file(data):
#     global row_set, col_set, sub_grid_set
#
#     if "." in data:
#         data, puzzle_size = convert_from_dot_to_number(data)
#     else:
#         data = data.split('\n')
#     puzzle_size = len(data)
#     input_array = [[0 for _ in range(puzzle_size)]
#                    for _ in range(puzzle_size)]
#     row_set = [set() for _ in range(puzzle_size)]
#     col_set = [set() for _ in range(puzzle_size)]
#     sub_grid_set = [set() for _ in range(puzzle_size)]
#     for row_num, row in enumerate(data):
#         for col_num, number in enumerate(row):
#             input_array[row_num][col_num] = int(number)
#             row_set[row_num].add(int(number))
#             col_set[col_num].add(int(number))
#             sub_grid_set[(row_num // int(math.sqrt(puzzle_size)))
#                          * int(math.sqrt(len(data)))
#                          + (col_num // int(math.ceil(math.sqrt(puzzle_size))))].add(int(number))
#
#         data = input_array
#     return data, puzzle_size

def parse_input_file(data):
    global row_set, col_set, sub_grid_set

    data = data.split(',')
    puzzle_size = int(math.sqrt(len(data)))
    data = [0 if num == "" else int(num) for num in data]
    rows_of_input = []
    for row in range(puzzle_size):
        rows_of_input.append(data[row*puzzle_size:row*puzzle_size+puzzle_size])

    input_array = [[0 for _ in range(puzzle_size)]
                   for _ in range(puzzle_size)]
    row_set = [set() for _ in range(puzzle_size)]
    col_set = [set() for _ in range(puzzle_size)]
    sub_grid_set = [set() for _ in range(puzzle_size)]
    for row_num, row in enumerate(rows_of_input):
        for col_num, number in enumerate(row):
            input_array[row_num][col_num] = int(number)
            row_set[row_num].add(int(number))
            col_set[col_num].add(int(number))
            sub_grid_set[(row_num // int(math.sqrt(puzzle_size)))
                         * int(math.sqrt(len(data)))
                         + (col_num // int(math.ceil(math.sqrt(puzzle_size))))].add(int(number))

        data = input_array
    return data, puzzle_size


def drop_down_menu():
    btn_input_file = Button(main_frame, text="Choose File",
                            command=browse_files, width=15)
    btn_input_file.pack()
    clicked.set("Select Options")
    drop = OptionMenu(main_frame, clicked, *options)
    drop.pack()
    button = Button(main_frame, text="Submit", command=submit, width=15)
    button.pack()


def display_message(msg):
    Label(main_frame, text=msg).grid(
        row=0, column=0, columnspan=5)


def create_sudoku():
    clear()
    drop_down_menu()


def on_click_solve_brute_force():
    clear()
    global size_data, puzzle_data
    size_data = len(puzzle_data)
    start = time.time()
    if solve_brute_force():
        print(f"Brute Force took {time.time() - start} s")
        draw_whole_grid(size_data, size_data, int(
            math.sqrt(size_data)), math.ceil(math.sqrt(size_data)))
    else:
        display_message("No solution found.")


def solve_heruistic():
    if len(cells) == 0:
        return
    pass


def solve_csp():
    if len(cells) == 0:
        return
    pass


def main():
    main_frame.pack(side=TOP)

    bottomFrame = Frame(root)

    bottomFrame.pack(side=BOTTOM, pady=20)
    btn_create = Button(bottomFrame, text="Create Sudoku",
                        command=create_sudoku, width=15)
    btn_create.grid(row=0, column=0)

    btn_solve_heuristic = Button(
        bottomFrame, text="Solve (heuristic)", command=on_click_solve_brute_force, width=15)
    btn_solve_heuristic.grid(row=0, column=4)

    btn_solve_csp = Button(bottomFrame, text="Solve (CSP)",
                           command=solve_csp, width=15)
    btn_solve_csp.grid(row=0, column=8)

    btn_clear = Button(bottomFrame, text="Clear", command=clear, width=15)
    btn_clear.grid(row=0, column=12)

    btn_exit = Button(bottomFrame, text="Exit", command=exit, width=15)
    btn_exit.grid(row=0, column=16)

    root.mainloop()


if __name__ == "__main__":
    main()
