def caesar_cipher(text, shift):
    """
    Encrypts a string using the Caesar Cipher algorithm with a specified shift.

    :param text: The string to be encrypted.
    :param shift: The number of positions to shift each letter in the string.
    :return: The encrypted string.
    """

    # Convert the text to uppercase for consistency
    text = text.upper()

    # Initialize the encrypted string
    encrypted_text = ""

    # Loop through each character in the string
    for char in text:
        if char.isalpha():
            # Shift the character by the specified amount
            shifted_char = chr((ord(char) - 65 + shift) % 26 + 65)
            encrypted_text += shifted_char
        else:
            # Non-alphabetic characters are unchanged
            encrypted_text += char

    return encrypted_text



