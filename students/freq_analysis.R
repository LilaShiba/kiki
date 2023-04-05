library(stringr)
library(ggplot2)


freq_count <- function(path) {
  # Read file, know file size
  book <- readChar(path, file.info(path)$size)
  # Convert to lowercase
  book <- tolower(book)
  # Count out non-alphabetic characters
  char_freq <- table(strsplit(book, "")[[1]])
  # Sort by frequency
  char_freq <- sort(char_freq, decreasing = TRUE)
  # Plot character frequencies
  df <- as.data.frame.table(char_freq)
  # subset df
  #df <- subset(df, Freq > 10001)
  #df <- subset(df, Freq < 70001)
  # Scatter Plot
  ggplot(df, aes(x = Var1, y = Freq)) +
    geom_point() +
    labs(x = "Char", y = "Freq", title = "Scatter Plot")

}

freq_count("b3.txt")
freq_count("b3_coded.txt")


x <- c(1, 2, 3, 4, 5)
y <- c(3, 5, 7, 9, 11)

plot(x, y, main = "Scatter Plot of x and y", xlab = "x", ylab = "y")
abline(lm(y ~ x), col = "red")

