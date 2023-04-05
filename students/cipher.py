def caesar_cipher(input_path, shift_value, output_path):
    # Read the input file
    with open(input_path, 'r') as file:
        input_text = file.read()

    # Convert the shift value to an integer
    shift_value = int(shift_value)

    # Define the alphabet (including spaces and punctuation)
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?:;-_'

    # Define a function to shift a single character
    def shift_char(char, shift_value):
        if char in alphabet:
            old_index = alphabet.index(char)
            new_index = (old_index + shift_value) % len(alphabet)
            return alphabet[new_index]
        else:
            return char

    # Define a function to apply the shift to a string
    def shift_string(string, shift_value):
        shifted_chars = [shift_char(char, shift_value) for char in string]
        return ''.join(shifted_chars)

    # Apply the shift to the input text
    output_text = shift_string(input_text, shift_value)

    # Write the output to a file
    with open(output_path, 'w') as file:
        file.write(output_text)


caesar_cipher('/Users/kjames/Desktop/kiki/students/book3.txt',8,'/Users/kjames/Desktop/kiki/students/b3_coded.txt')