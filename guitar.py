import os

with open("wordlist.10000.txt", "r", encoding="utf-8") as f:
    wordlist = [line.strip() for line in f]

while True:
    word = input("Enter a word (or type 'exit' to quit): ")

    if word.lower() == "exit":
        print("Bye!")
        break

    os.system("cls" if os.name == "nt" else "clear")

    found = False
    for line in wordlist:
        if word.lower() in line.lower():
            print(line)
            found = True

    if not found:
        print("No match found.")
