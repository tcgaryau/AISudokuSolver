from tkinter import *

root = Tk()
root.title("AI Sudoku Solver")
# Width x Height


cells = {}

main_frame = Frame(root)
main_frame.pack(side=TOP)

label = Label(main_frame, text="Welcome User").grid(
    row=0, column=0, columnspan=5)


def drawSubGrid(row_num, col_num, sub_row_num, sub_col_num, bgcolour):
    frame = Frame(main_frame)
    for i in range(sub_row_num):
        for j in range(sub_col_num):
            label = Label(frame, width=2, text=col_num+j + 1,
                          bg=bgcolour, justify="center")
            label.grid(row=i+1, column=j+1,
                       sticky="nsew", padx=1, pady=1, ipady=2)
            cells[(row_num+i+1, col_num+j+1)] = label
    frame.grid(row=row_num+1, column=col_num+1)


def drawWholeGrid(row, col, sub_row_num, sub_col_num):
    colour = "#D0ffff"
    for row_num in range(1, row + 1, sub_row_num):
        for col_num in range(0, col, sub_col_num):
            drawSubGrid(row_num, col_num, sub_row_num, sub_col_num, colour)
            if colour == "#D0ffff":
                colour = "#ffffd0"
            else:
                colour = "#D0ffff"
        if sub_row_num % 2 == 0:
            if colour == "#D0ffff":
                colour = "#ffffd0"
            else:
                colour = "#D0ffff"


bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM, pady=20)
btn_create = Button(bottomFrame, text="Create Sudoku", width=15)
btn_create.grid(row=0, column=0)

btn_solve_heuristic = Button(bottomFrame, text="Solve (heuristic)", width=15)
btn_solve_heuristic.grid(row=0, column=4)

btn_solve_csp = Button(bottomFrame, text="Solve (CSP)", width=15)
btn_solve_csp.grid(row=0, column=8)

btn_clear = Button(bottomFrame, text="Solve (CSP)", width=15)
btn_clear.grid(row=0, column=12)

btn_exit = Button(bottomFrame, text="Exit", width=15)
btn_exit.grid(row=0, column=16)


drawWholeGrid(16, 16, 4, 4)
root.mainloop()
