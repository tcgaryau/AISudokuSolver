def main():
    with open("test_12x12_board_3.txt", "r") as f:
        puzzle_data = [line.strip() for line in f.readlines()]

    pline = ""
    for line in puzzle_data:
        for idx, char in enumerate(line.split(",")):
            if idx % 12 == 0:
                print(pline[:-1])
                pline = ""
            pline += char + ","
        print(pline[:-1])


if __name__ == "__main__":
    main()