import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import collections


def freq_count(path):
    with open(path, 'r') as file:
        book = file.read()
        file.close()

    char_freq = collections.Counter(book.lower())
    filtered_keys = sorted( 
        [(key,val) for key,val in char_freq.items() if key.isalpha()]
        , key=lambda x:x[1]
        )

    x,y = zip(*filtered_keys)

    n = np.sum(y)
    probVector = zip(x, [cnt/n for cnt in y])
    px,py = zip(*probVector)
    
    mu = np.mean(y)

    plt.xlabel('Char')
    plt.ylabel('Frequency')
    plt.title('Frequency of Char')
    plt.scatter(x, y)
    plt.show()


    # plt.xlabel('K')
    # plt.ylabel('P(K)')
    # plt.title('Prob Vector')
    # plt.scatter(px, py)
    # plt.show()



    return probVector, mu, filtered_keys







freq_count('/Users/kjames/Desktop/kiki/students/book.txt')
freq_count('/Users/kjames/Desktop/kiki/students/b3_coded.txt')



