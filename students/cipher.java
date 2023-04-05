import java.util.Scanner;
// Main Program
class Main {
  public static void main(String[] args) {

    Scanner input = new Scanner(System.in);

    System.out.print("Enter the plaintext: ");
    String plaintext = input.nextLine();
    System.out.print("Enter the key (shift amount): ");
    int key = input.nextInt();

    String ciphertext = encrypt(plaintext, key);
    System.out.println("Encrypted text: " + ciphertext);

  }
  // Encrypt Method
  public static String encrypt(String plaintext, int key) {
    String ciphertext = "";
    for (int i = 0; i < plaintext.length(); i++) {
      char ch = plaintext.charAt(i);
      if (Character.isLetter(ch)) {
        ch = (char) ((ch + key - 'a') % 26 + 'a');
      }
      ciphertext += ch;
    }
    return ciphertext;
  }
}