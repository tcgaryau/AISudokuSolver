#AISudokuSolver
Team4 for COMP3981
Ramil Garipov, Felix Ng, Gary Au, Kirin Tang

Wireframe:
https://www.figma.com/file/oHEESjudYPdYPDR0O3w2S9/Wireframing-(Copy)?node-id=651378-93


# How to run:
To use the GUI, run the sudoku.py file.
Then the user can click "Create Sudoku" and subsequently "Choose File" to select a '.txt' file or the dropdown menu to
select the size of the Sudoku board.
The user now clicks "Submit" to generate the Sudoku board.
Now the user can click "Solve (Heuristic)" or "Solve (CSP)" to solve the Sudoku board with the corresponding algorithm.

# Terminal test
To see CSP results directly in the terminal, run test_csp.py.
Navigate to the bottom of the file and run the main function, passing in the parameter for the size of puzzles that you
would like to test. We have pre-built the following lists of puzzles: [all_9x9, all_12x12, all_16x16, all_25x25, 
all_100x100]. Currently, the program is set to run all_9x9 puzzles. The results of the board and the computational time 
will be printed in the terminal.

# File validation:
For puzzle input files, the program is able to detect the following errors:
* Too many rows
* Too few rows
* Missing a number in a row
* Extra number in a row
* A word/letter instead of a number

# Functionalities:
The user can select both "Solve (Heuristic)" and "Solve (CSP)" to solve the same board with both algorithms and the 
result will be displayed side by side.
The user can click "Clear" to clear the UI and start over.
The user can click "Exit" to exit the program.

# Brute Force:
We were able to solve all the 9x9 and 12x12 puzzles. For 16x16, we were able to solve 12/15 puzzles. 
For the 25x25, they were all timed out at 5 minutes, so the rest of those times were not tracked. 

Instead of a naive brute force, we added in some heuristics. To calculate the domain of a selected cell, we combined the
domains of all of its neighbours (in the row, column, and subgrid), and determined values were not used by its 
neighbours. Then these values can be attempted to be assigned to that cell.

We also used MRV to determine the next value to pick on our brute force.

## Limits:
We restricted the Brute Force execution down to 5 minutes. 
We also created a counter for the number of values attempted in each solution. If this number went over the limit of
(puzzle_size^2 * 1000), that solution would be cut short, and the next value at the starting square would be tried next.

In order to potentially improve the calculation time, we defined a limit of the number of branches of each tree that is 
allowed. That limit is defined as (puzzle_size ^ 2 * 1000). 
If that branch reaches the limit before the time limit is reached, the next value at the starting square would be 
tried next. Otherwise we conclude that the puzzle will not reach a solution in our defined time limit.

# CSP: 
We are able to solve all the 9x9, 12x12 and 16x16. For the 25x25 we were able to solve 7/15 of the puzzles.

## Multiprocessing:
All the best values of the best cell (determined by  MRV and degree heuristic) get their own CPU core to be processed
for a CSP solution. This only happens for the first call as we were unable to apply multiprocessing to the recursive
calls to backtrack. If any process finds a solution to the puzzle, it will kill all the other processes.

## MAC and AC3:
Maintain Arc consistency and AC3 are used to propagate the constraints of the Sudoku board. This is done once before 
the CSP algorithm begins in order to reduce the domain space. Then it is done during each recursive call in order to 
propagate the constraints across the rest of the board based on a new assignment.

## Backtrack:
We used the backtrack algorithm while using these heuristic:
* AC3 to pre-preprocess to reduce search space before entering backtrack
* Enter backtrack
* MRV and degree heuristic
* Least constraining value heuristic
* Naked Pair Heuristic
* MAC using ac3 is then run

## MRV and Degree heuristic:
MRV is used to select the best cell to assign a value to. The best cell is the cell with the smallest domain.
In order to rank the cell's domain values, we use the degree heuristic. The degree heuristic ranks the domain values
such that the value that is most constrained is assigned first. This is done by counting the frequency of the domain 
values of the unassigned neighbours.

## Least constraining value heuristic:
On the domain list from that was determined from MRV and Degree Heuristic we order the domain list where the variable
chosen would rule out the fewest choices for the remaining domains of its neighbors.

## Naked Pair Heuristic:
This is where we look at all the unassigned cells on the sudoku board.
For each of these cells we would look at the row, column, and subgrid. For each of these group if there are
2 cells that are domain size 2 and have the same domain, we can remove these values from the domain on the other cells
in the corresponding row, column or subgrid.

## Uniqueness Constraint:
If a cell only has one value in the domain, we can effectively assign the value. Then that value can be removed from
its neighbour's domain.

## Hidden Pair Heuristic: (Not implemented)
If exactly two neighbouring squares are sharing two numbers in their domains, that means that their domains can be 
reduced to just those two numbers. Additionally, these two numbers can be removed from the domains of all their 
neighbors.

## Naked Subsets Heuristic: (Not implemented)
Similar to naked pairs, we can look further into naked triplets, naked quads, etc. which would help speed up larger 
board sizes.

# Resources:
Solving Sudoku by Heuristic Search - David Carmel
https://medium.com/@davidcarmel/solving-sudoku-by-heuristic-search-b0c2b2c5346e

GUI Sudoku Solver | Python Tutorial - Sharnav's Tech
https://www.youtube.com/watch?v=xAXmfZmC2SI

Adding a Full Screen ScrollBar - Python Tkinter GUI Tutorial - Codemy.com
https://www.youtube.com/watch?v=0WafQCaok6g