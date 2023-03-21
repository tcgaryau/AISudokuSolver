def main():
    with open("test_16x16_board_1.txt", "r") as f:
        puzzle_data = [line.strip() for line in f.readlines()]

    pline = ""
    for line in puzzle_data:
        for idx, char in enumerate(line.split(",")):
            if idx % 16 == 0:
                print(pline[:-1])
                pline = ""
            pline += char + ","


    # for idx, char in enumerate(puzzle_data):


        # print(char)
        # line += char
        # if idx % 16 == 0:
        #     print(line)
        #     line = ""



if __name__ == "__main__":
    main()