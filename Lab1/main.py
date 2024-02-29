def count_words(file_paths):
    combined_word_counts = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                text = file.read()
                words = text.split()
                for word in words:
                    combined_word_counts[word] = combined_word_counts.get(word, 0) + 1
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
    
    return combined_word_counts

def write_word_counts_to_file(word_counts, output_file):
    try:
        with open(output_file, 'w') as file:
            for word, count in word_counts.items():
                file.write(f"{word}: {count}\n")
        print(f"Word counts written to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    file_paths = []
    for i in range(3):
        file_path = input(f"Enter the path of file {i+1}: ")
        file_paths.append(file_path)
    
    output_file = input("Enter the path of the output file: ")

    word_counts = count_words(file_paths)
    if word_counts:
        write_word_counts_to_file(word_counts, output_file)

