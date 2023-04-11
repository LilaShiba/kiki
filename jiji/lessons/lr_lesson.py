import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x, y):
    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the difference between each x value and the mean of x
    x_diff = x - x_mean

    # Calculate the difference between each y value and the mean of y
    y_diff = y - y_mean

    # Calculate the slope of the line
    slope = np.sum(x_diff * y_diff) / np.sum(x_diff ** 2)

    # Calculate the intercept of the line
    intercept = y_mean - slope * x_mean

    return slope, intercept


if __name__ == "__main__":
    # Generate some example data
    # study_time = np.array([2, 4, 6, 8, 10, 12])
    # exam_score = np.array([65, 70, 75, 80, 85, 90])

    # # Plot the data
    # plt.scatter(study_time, exam_score)
    # plt.xlabel('Study Time')
    # plt.ylabel('Exam Score')
    # plt.show()

    # slope, intercept = linear_regression(study_time, exam_score)
    # print('Slope:', slope)
    # print('Intercept:', intercept)

    # # Plot the data
    # plt.scatter(study_time, exam_score)
    # plt.xlabel('Study Time')
    # plt.ylabel('Exam Score')

    # # Plot the line of best fit
    # line_x = np.array([0, 14])
    # line_y = slope * line_x + intercept
    # plt.plot(line_x, line_y, color='red')

    # plt.show()

    # Example 2
    # Define the age and income arrays
    age = np.array([22, 25, 30, 35, 40, 45, 50, 55, 60])
    income = np.array([25000, 32000, 39000, 48000, 56000, 67000, 78000, 89000, 98000])

    # Plot the data points
    plt.scatter(age, income)
    plt.show()
    slope, intercept = linear_regression(age, income)
    print('Slope:', slope)
    print('Intercept:', intercept)

    # Plot the data
    plt.scatter(age, income)
    plt.xlabel('Study Time')
    plt.ylabel('Exam Score')

    
    line_x = np.array([20, 65])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red')
    plt.show()

def rate_of_change(vector):
    # Define the initial and final position of the object (in meters)
    initial_position = vector[0]
    final_position = vector[-1]

    # Define the initial and final time (in seconds)
    initial_time = 0.0
    final_time = 5.0

    # Calculate the change in position and time
    change_in_position = final_position - initial_position
    change_in_time = final_time - initial_time

    # Calculate the rate of change (velocity) of the object
    velocity = change_in_position / change_in_time

    # Print the result
    print("The rate of change of the position of the object is:", velocity, "m/s")
