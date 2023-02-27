from tkinter import *


root = Tk()
root.title("AI Sudoku Solver")
root.geometry("900x900")

main_frame = Frame(root)
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


def clear():
    for widgets in main_frame.winfo_children():
        widgets.destroy()
    cells.clear()


def change_colour(colour):
    if colour == "#D0ffff":
        return "#ffffd0"
    return "#D0ffff"


def draw_sub_grid(row_num, col_num, sub_row_num, sub_col_num, bgcolour):
    frame = Frame(main_frame)

    # segment: use a canvas inside subframe
    width = 80
    height = 80
    canvas = Canvas(frame, height=height, width=width,
                    bg=bgcolour, bd=0, highlightthickness=0)
    canvas.pack(fill=BOTH, expand=False)

    for i in range(0, sub_row_num):
        for j in range(0, sub_col_num):
            x = j * width/sub_col_num
            y = i * height/sub_row_num
            canvas.create_rectangle(
                x, y, x + width/sub_col_num, y + height/sub_row_num, outline='black')
            canvas.create_text(x + width/sub_col_num/2, y + height/sub_row_num/2,
                               text=f"{int(col_num+j+1)}", font=(None, f'{width//sub_col_num//2}'))
    # end of segment

    # for i in range(sub_row_num):
    #     for j in range(sub_col_num):
    #         label = Label(frame, width=2, text=col_num+j+1,
    #                       bg=bgcolour, justify="center", borderwidth=1, relief="solid")
    #         label.grid(row=i+1, column=j+1,
    #                    sticky="nsew", ipady=2)
    frame.grid(row=row_num+1, column=col_num+1)


def draw_whole_grid(row, col, sub_row_num, sub_col_num):
    clear()
    colour = "#D0ffff"
    for row_num in range(0, row, sub_row_num):
        for col_num in range(0, col, sub_col_num):
            draw_sub_grid(row_num, col_num, sub_row_num, sub_col_num, colour)
            colour = change_colour(colour)
        if sub_row_num % 2 == 0:
            colour = change_colour(colour)


def submit():
    match clicked.get():
        case "9x9":
            draw_whole_grid(9, 9, 3, 3)
        case "12x12":
            draw_whole_grid(12, 12, 3, 4)
        case "16x16":
            draw_whole_grid(16, 16, 4, 4)
        case "25x25":
            draw_whole_grid(25, 25, 5, 5)
        case "100x100":
            draw_whole_grid(100, 100, 10, 10)


def drop_down_menu():
    btn_input_file = Button(main_frame, text="Choose File", width=15)
    clicked.set("Select Options")
    drop = OptionMenu(main_frame, clicked, *options)
    drop.pack()
    button = Button(main_frame, text="Submit", command=submit, width=15)
    button.pack()


def create_sudoku():
    clear()
    drop_down_menu()


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
        bottomFrame, text="Solve (heuristic)", command=solve_heruistic, width=15)
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
