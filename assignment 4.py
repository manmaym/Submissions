#q1
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate 50 random data points for x and y coordinates
np.random.seed(42)  # Set seed for reproducibility
x = np.random.rand(50) * 100  # Random x-coordinates between 0 and 100
y = np.random.rand(50) * 100  # Random y-coordinates between 0 and 100

# Step 2: Calculate the average values of x and y coordinates
avg_x = np.mean(x)
avg_y = np.mean(y)

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the average point
plt.scatter(avg_x, avg_y, color='red', s=100, label='Average Point')
plt.text(avg_x, avg_y, f'({avg_x:.2f}, {avg_y:.2f})', color='red', fontsize=12, ha='right')

# Add labels and title
plt.title('Scatter Plot of Random Data Points with Average Point')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')

# Display the legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#q2
import numpy as np
np.random.seed(42)
temperature_data = np.random.uniform(15, 35, (7, 24))
average_daily_temperatures = np.mean(temperature_data, axis=1)
max_temperature = np.max(temperature_data)
min_temperature = np.min(temperature_data)
print("Average daily temperatures (°C):\n", average_daily_temperatures)
print("Maximum temperature of the week (°C):\n", max_temperature)
print("Minimum temperature of the week (°C):\n", min_temperature)

#q3
import matplotlib.pyplot as plt
import numpy as np

# Sample sales data for each month
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales = [150, 200, 180, 220, 300, 250, 400, 350, 320, 310, 280, 260]

# Identify the month with the highest sales
max_sales = max(sales)
max_sales_index = sales.index(max_sales)

# Create the bar chart
plt.figure(figsize=(12, 6))
bar_colors = ['lightblue' if i != max_sales_index else 'orange' for i in range(len(sales))]

plt.bar(months, sales, color=bar_colors)

# Highlight the month with the highest sales
plt.text(months[max_sales_index], max_sales, f'Highest: {max_sales}', ha='center', va='bottom', color='black', fontsize=12)

# Add labels and title
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales (in units)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Show the plot
plt.show()


#q4
import numpy as np

# Create a random NumPy array
np.random.seed(42)
array = np.random.randint(1, 101, size=20)  # Array of 20 random integers between 1 and 100
print("Array:", array)

# Function to search for an element in the array
def search_element(arr, element):
    index = np.where(arr == element)
    if index[0].size > 0:
        return index[0][0]  # Return the first index found
    else:
        return -1  # Element not found

# Take the number to search from the user
element_to_search = int(input("Enter the number to search: "))
index = search_element(array, element_to_search)

if index != -1:
    print(f"Element {element_to_search} found at index {index}.")
else:
    print(f"Element {element_to_search} not found in the array.")

#q5
import matplotlib.pyplot as plt

# Sales data for each month
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales = [12000, 15000, 13000, 17000, 14000, 17000, 18000, 17000, 17700, 18900, 20000, 21000]

# Create a line chart
plt.figure(figsize=(12, 6))
plt.plot(months, sales, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Monthly Sales')

# Add a grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add labels and title
plt.title('Trend of Monthly Sales Over the Year')
plt.xlabel('Month')
plt.ylabel('Sales (in USD)')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#q6
import numpy as np

# Create an array in the range 1 to 20 with values 1.25 apart
array1 = np.arange(1, 20, 1.25)

# Create another array containing the natural logarithm of the elements in array1
log_array = np.log(array1)

# Display the arrays
print("Array 1 (Original values):")
print(array1)
print("\nArray 2 (Logarithm of the original values):")
print(log_array)


#q7
import mypackage.math_operations as mp
a= int(input("Enter first number:",))
b= int(input("Enter second number:",))
print("Sum of two numbers is:",mp.add(a,b))

#q8

import matplotlib.pyplot as plt

# Data
products = ["A", "B", "C", "D", "E"]
sales = [15, 30, 25, 10, 20]

# Subplot configuration
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart (left)
axs[0].bar(products, sales, color='skyblue')
axs[0].set_title('Number of Sales by Product')
axs[0].set_xlabel('Products')
axs[0].set_ylabel('Number of Sales')

# Pie chart (right)
axs[1].pie(sales, labels=products, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'pink', 'yellow', 'lightcoral'])
axs[1].set_title('Market Share by Product')

plt.tight_layout()
plt.show()

##9
import matplotlib.pyplot as plt

# Data for the plots
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]
hist_data = [22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27]
categories = ['A', 'B', 'C', 'D']
values = [5, 7, 3, 8]

# Creating a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Line chart
axs[0, 0].plot(x, y, marker='o', color='blue')
axs[0, 0].set_title('Line Chart')
axs[0, 0].set_xlabel('X-axis')
axs[0, 0].set_ylabel('Y-axis')

# Scatter plot
axs[0, 1].scatter(x, y, color='red')
axs[0, 1].set_title('Scatter Plot')
axs[0, 1].set_xlabel('X-axis')
axs[0, 1].set_ylabel('Y-axis')

# Histogram
axs[1, 0].hist(hist_data, bins=5, color='green', edgecolor='black')
axs[1, 0].set_title('Histogram')
axs[1, 0].set_xlabel('Bins')
axs[1, 0].set_ylabel('Frequency')

# Bar chart
axs[1, 1].bar(categories, values, color='purple')
axs[1, 1].set_title('Bar Chart')
axs[1, 1].set_xlabel('Categories')
axs[1, 1].set_ylabel('Values')

# Adjusting the layout
plt.tight_layout()
plt.show()


#q10
import numpy as np

# Generate a range of values from 0 to 2π with an interval of 0.1
x = np.arange(0, 2 * np.pi, 0.1)

# Compute the sine and cosine values for each x value
sine_values = np.sin(x)
cosine_values = np.cos(x)

# Print the generated data
print("x values:", x)
print("Sine values:", sine_values)
print("Cosine values:", cosine_values)

#11
import matplotlib.pyplot as plt
import numpy as np

# Sample data for the plot
months = np.arange(1, 13)
average_temperatures = [30, 32, 35, 40, 45, 50, 55, 55, 50, 45, 35, 30]  # Average monthly temperatures
rainfall = [80, 70, 60, 50, 40, 30, 20, 30, 40, 60, 70, 80]  # Monthly rainfall

# Create subplots that share the same x-axis
fig, ax1 = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# First subplot (Bar chart for average temperatures)
ax1[0].bar(months, average_temperatures, color='orange')
ax1[0].set_title("Average Monthly Temperatures")
ax1[0].set_ylabel("Temperature (°C)")

# Second subplot (Line chart for rainfall)
ax1[1].plot(months, rainfall, color='blue', marker='o')
ax1[1].set_title("Monthly Rainfall")
ax1[1].set_ylabel("Rainfall (mm)")
ax1[1].set_xlabel("Month")

# Show the plots
plt.tight_layout()
plt.show()


#q12
import numpy as np
# Create a 2D NumPy array of shape (4, 5) containing integers from 1 to 20
array_2d = np.arange(1, 21).reshape(4, 5)
# Reshape the array into a new shape of (5, 4)
reshaped_array = array_2d.reshape(5, 4)
# Print the original and reshaped arrays
print("Original array (4, 5):")
print(array_2d)
print("\nReshaped array (5, 4):")
print(reshaped_array)
# Create a NumPy array with random integers between 1 and 100
np.random.seed(42)  # For reproducibility
random_array = np.random.randint(1, 101, size=20)
# Sort the array in ascending order
sorted_array = np.sort(random_array)
# Slice the array to get the top 5 largest elements
top_5_largest = sorted_array[-5:]
# Print the sorted array and the sliced array
print("\nSorted array:")
print(sorted_array)
print("\nTop 5 largest elements:")
print(top_5_largest)

#13
import matplotlib.pyplot as plt
import numpy as np

# Sample data
means = [0.2474, 0.1235, 0.1737, 0.1824]  # Mean velocity values
std_devs = [0.3314, 0.2278, 0.2836, 0.2645]  # Standard deviation of velocity
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']  # Categories for the bars

# Create the bar plot with error bars
plt.bar(categories, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')

# Set the labels and title
plt.xlabel('Categories')
plt.ylabel('Mean Velocity')
plt.title('Mean Velocity with Error Bars')

# Show the plot
plt.tight_layout()
plt.show()


#14
import matplotlib.pyplot as plt

# Data for the pie chart
flavors = ['Strawberry', 'Vanilla', 'Chocolate', 'Butterscotch', 'Raspberry', 'Mint', 'Blueberry']
frequencies = [44, 76, 30, 78, 39, 11, 22]

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(frequencies, labels=flavors, autopct='%1.1f%%', startangle=140, colors=['pink', 'lightblue', 'brown', 'yellow', 'red', 'green', 'blue'])

# Set the title
plt.title('Favorite Ice Cream Flavors of 300 Children')

# Display the pie chart
plt.show()


#15
import numpy as np
from scipy.stats import pearsonr

# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Generate an array of prime numbers between 2 and 1000
primes_2_1000 = [num for num in range(2, 1001) if is_prime(num)]

# Generate an array of prime numbers between 2000 and 4000
primes_2000_4000 = [num for num in range(2000, 4001) if is_prime(num)]

# Truncate the larger array to make it the same size as the smaller array
min_size = min(len(primes_2_1000), len(primes_2000_4000))
primes_2_1000_truncated = primes_2_1000[:min_size]
primes_2000_4000_truncated = primes_2000_4000[:min_size]

# Find the correlation between the two arrays
correlation, _ = pearsonr(primes_2_1000_truncated, primes_2000_4000_truncated)

# Output the results
print("Prime numbers between 2 and 1000:", primes_2_1000_truncated)
print("Prime numbers between 2000 and 4000:", primes_2000_4000_truncated)
print("Correlation between the two arrays:", correlation)


#q16
import numpy as np

# Create a NumPy array with the names of five students
students = np.array(["Alice", "Bob", "Charlie", "David", "Eve"])

# Create a NumPy array with their corresponding scores in Mathematics
scores = np.array([85, 78, 92, 67, 88])

# 1. Print both arrays
print("Student names:", students)
print("Mathematics scores:", scores)

# 2. Calculate the average score and print it
average_score = np.mean(scores)
print("\nAverage score in Mathematics:", average_score)

# 3. Find students who scored above 75 using boolean indexing
above_75 = students[scores > 75]
above_75_scores = scores[scores > 75]
print("\nStudents who scored above 75:")
for student, score in zip(above_75, above_75_scores):
    print(f"{student}: {score}")

# 4. Sort the scores in descending order and display with corresponding student names
sorted_indices = np.argsort(scores)[::-1]  # Get indices for sorting in descending order
sorted_students = students[sorted_indices]
sorted_scores = scores[sorted_indices]

print("\nSorted scores in descending order with corresponding students:")
for student, score in zip(sorted_students, sorted_scores):
    print(f"{student}: {score}")

# 5. Find and print the highest and lowest scores along with student names
highest_score = np.max(scores)
lowest_score = np.min(scores)

highest_scorer = students[scores == highest_score][0]  # Get the student with the highest score
lowest_scorer = students[scores == lowest_score][0]    # Get the student with the lowest score

print("\nHighest score:")
print(f"{highest_scorer}: {highest_score}")

print("\nLowest score:")
print(f"{lowest_scorer}: {lowest_score}")

#17
import matplotlib.pyplot as plt
import numpy as np

# Data for the closing values of Alphabet Inc. from October 3, 2016, to October 7, 2016
dates = ['2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']
closing_values = [776.86, 774.39, 777.60, 774.96, 776.43]  # Sample closing values for Alphabet Inc.

# Create a line chart
plt.plot(dates, closing_values, marker='o', color='b', label='Closing Value')

# Customizing the grid
plt.grid(which='major', linestyle='-', linewidth=1, color='black')  # Major grid with black color
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')  # Minor grid with gray color and dotted lines
plt.minorticks_on()  # Turn on minor ticks
plt.tick_params(axis='both', which='both', length=0)  # Turn off the ticks

# Labels and title
plt.xlabel('Date')
plt.ylabel('Closing Value (USD)')
plt.title('Alphabet Inc. Closing Value from October 3 to October 7, 2016')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


#q18
import numpy as np

# Create sample arrays
array1 = np.array([2, 4, 6, 8, 10])
array2 = np.array([1, 3, 5, 7, 9])

# Add the arrays
array_sum = array1 + array2
print("Sum of the arrays:", array_sum)

# Subtract the arrays
array_difference = array1 - array2
print("Difference of the arrays:", array_difference)

# Retrieve and display elements between 4 and 9 (excluding 4 and 9) from the first array
filtered_elements = array1[(array1 > 4) & (array1 < 9)]
print("\nElements in array1 between 4 and 9 (excluding 4 and 9):", filtered_elements)

#q18
import numpy as np

# Create sample arrays
array1 = np.array([2, 4, 6, 8, 10])
array2 = np.array([1, 3, 5, 7, 9])

# Add the arrays
array_sum = array1 + array2
print("Sum of the arrays:", array_sum)

# Subtract the arrays
array_difference = array1 - array2
print("Difference of the arrays:", array_difference)

# Retrieve and display elements between 4 and 9 (excluding 4 and 9) from the first array
filtered_elements = array1[(array1 > 4) & (array1 < 9)]
print("\nElements in array1 between 4 and 9 (excluding 4 and 9):", filtered_elements)


#q19
import numpy as np

# Create two 3x3 NumPy arrays with random integers between 1 and 10
np.random.seed(42)  # For reproducibility
array1 = np.random.randint(1, 11, size=(3, 3))
array2 = np.random.randint(1, 11, size=(3, 3))

# Perform element-wise addition
addition_result = array1 + array2

# Perform element-wise subtraction
subtraction_result = array1 - array2

# Perform element-wise multiplication
multiplication_result = array1 * array2

# Perform element-wise division (use np.divide to handle division safely)
division_result = np.divide(array1, array2)

# Print the arrays and results
print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)

print("\nElement-wise Addition:")
print(addition_result)

print("\nElement-wise Subtraction:")
print(subtraction_result)

print("\nElement-wise Multiplication:")
print(multiplication_result)

print("\nElement-wise Division:")
print(division_result)

#20
import numpy as np
from scipy import stats

# Data represented as a list of dictionaries
data = [
    {'Rollno': 'S1001', 'Name': 'Arun', 'Age': 18, 'Marks': 68},
    {'Rollno': 'S1002', 'Name': 'Mohit', 'Age': 14, 'Marks': 47},
    {'Rollno': 'S1003', 'Name': 'Karan', 'Age': 13, 'Marks': 78},
    {'Rollno': 'S1004', 'Name': 'Lalit', 'Age': 16, 'Marks': 87},
    {'Rollno': 'S1005', 'Name': 'Ravi', 'Age': 14, 'Marks': 60}
]

# Extracting relevant columns
ages = [student['Age'] for student in data]
marks = [student['Marks'] for student in data]

# a) Maximum marks and minimum marks
max_marks = max(marks)
min_marks = min(marks)

# b) Sum of all the marks
sum_marks = sum(marks)

# c) Mean and Mode of Age of the students
mean_age = np.mean(ages)
mode_age = stats.mode(ages)[0][0]  # Mode returns an array, so we take the first element

# d) Count the number of rows in the list
row_count = len(data)

# Displaying the results
print(f"Maximum Marks: {max_marks}")
print(f"Minimum Marks: {min_marks}")
print(f"Sum of Marks: {sum_marks}")
print(f"Mean Age: {mean_age}")
print(f"Mode of Age: {mode_age}")
print(f"Number of Rows: {row_count}")
 

#q21
array11=np.random.randint(1,11,(6,3))
reshapearray11=np.reshape(array11,(2,9))
filter=np.where(array11%2==0,1,0)
print("og array is:",array11)
print("reshaped arrray is :",reshapearray11)
print("filtered array showing even numbers as 1 is:",filter)


#22
import matplotlib.pyplot as plt
import numpy as np

# Example data: heatwave days in the summer months (March, April, May) for 5 years
years = ['2019', '2020', '2021', '2022', '2023']
march_days = [5, 8, 3, 6, 7]  # Number of heatwave days in March for each year
april_days = [10, 12, 9, 13, 11]  # Number of heatwave days in April for each year
may_days = [15, 18, 14, 20, 19]  # Number of heatwave days in May for each year

# X-axis positions for the bars
x = np.arange(len(years))

# Width of each bar
bar_width = 0.25

# Creating the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar1 = ax.bar(x - bar_width, march_days, bar_width, label='March', color='orange')
bar2 = ax.bar(x, april_days, bar_width, label='April', color='red')
bar3 = ax.bar(x + bar_width, may_days, bar_width, label='May', color='yellow')

# Adding labels and title
ax.set_xlabel('Years')
ax.set_ylabel('Heatwave Days')
ax.set_title('Heatwave Days During the Summer Season (March-May) for 5 Years')
ax.set_xticks(x)
ax.set_xticklabels(years)

# Adding a legend
ax.legend()

# Displaying the plot
plt.tight_layout()
plt.show()

#23
















#q24
heightarray=np.random.randint(160,181,(1,11))
weightarray=np.random.randint(65,85,(1,11))
BMIarray=(weightarray/heightarray**2)
print("height array is:",heightarray)
print("weoght array is:",weightarray)
print("BMI array is:",BMIarray)

#q25
array12=np.random.randint(1,21,(1,11))
evenarray=array12[array12%2==0]
print("og array:",array12)
print("even array:",evenarray)


#26
import matplotlib.pyplot as plt

# Data for the months and total profit
months = [1, 4, 7, 10, 12]  # Month numbers
profits = [1200, 200, 800, 1500, 1700]  # Corresponding total profit in INR

# Create a line plot
plt.plot(months, profits, marker='o', color='r', markersize=8)  # Circle marker with red line

# Label the axes
plt.xlabel('Month Number')
plt.ylabel('Total Profit in INR')

# Add title to the plot
plt.title('Total Profit per Month')

# Show the plot
plt.tight_layout()
plt.show()

#27
import matplotlib.pyplot as plt

# Data for programming languages and their popularity (in arbitrary units)
languages = ['Python', 'Java', 'JavaScript', 'C++', 'Ruby', 'PHP']
popularity = [85, 75, 80, 60, 45, 50]  # Example popularity scores

# Define different colors for each bar
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F6', '#33FFF4']

# Create a bar chart
plt.bar(languages, popularity, color=colors)

# Label the axes
plt.xlabel('Programming Languages')
plt.ylabel('Popularity')

# Add a title
plt.title('Popularity of Programming Languages')

# Display the plot
plt.tight_layout()
plt.show()


#q28
array13=np.random.randint(10,51,(6,6))
evenarray13=np.where(array13%2==0,-1,array13)
print("og array:",array13)
print("even array:",evenarray13)

#q29
array14=np.random.randint(1,101,(3,3))
determinant=np.linalg.det(array14)
trasnpose=np.matrix.transpose(array14)
print("orignal matrix is:",array14)
print("determnant of matrix is:",determinant)
print("tranpose of matrix is :",trasnpose)
try:
    if det!=0:
        inverse=np.linalg.inv(array14)
        print("the inverse of matrix is:",inverse)
    else:
        print("inverse of matrix does not exist")
except:
    print("an error occured, matrix may not have inverse")

#30
import matplotlib.pyplot as plt
import numpy as np

# Data for the months and temperatures
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
temperatures = [5, 7, 10, 15, 20, 25, 30, 29, 22, 16, 10, 6]

# Define seasons and corresponding months
seasons = {
    'Winter': ['December', 'January', 'February'],
    'Spring': ['March', 'April', 'May'],
    'Summer': ['June', 'July', 'August'],
    'Autumn': ['September', 'October', 'November']
}

# Calculate average temperatures for each season
season_averages = {}
for season, months_in_season in seasons.items():
    season_temps = [temperatures[months.index(month)] for month in months_in_season]
    season_averages[season] = np.mean(season_temps)

# Create a figure and a set of subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# First subplot: Line chart for monthly temperatures
ax[0].plot(months, temperatures, marker='o', color='b', label='Temperature (°C)')
ax[0].set_title('City Monthly Temperatures')
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Temperature (°C)')
ax[0].grid(True)
ax[0].legend()

# Second subplot: Bar chart for average temperatures per season
ax[1].bar(season_averages.keys(), season_averages.values(), color=['#FF5733', '#33FF57', '#3357FF', '#F3FF33'])
ax[1].set_title('Average Temperature by Season')
ax[1].set_xlabel('Season')
ax[1].set_ylabel('Average Temperature (°C)')

# Rotate x-axis labels for better readability
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)

# Adjust layout for better fit
plt.tight_layout()

# Display the plots
plt.show()

#31
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Data for study hours and exam scores
study_hours = [2, 3, 4.5, 1, 6, 7.5, 8, 5, 9, 2.5, 3.5, 7, 6.5, 4, 8.5]
exam_scores = [55, 60, 65, 50, 75, 85, 90, 70, 95, 58, 63, 88, 80, 68, 92]

# Scatter plot to show the relationship between study hours and exam scores
plt.figure(figsize=(12, 6))

# Subplot 1: Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(study_hours, exam_scores, color='b', label='Data points')
plt.title('Study Hours vs Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')

# Add a trendline (linear regression)
slope, intercept, r_value, p_value, std_err = stats.linregress(study_hours, exam_scores)
line_x = np.linspace(min(study_hours), max(study_hours), 100)
line_y = slope * line_x + intercept
plt.plot(line_x, line_y, color='r', label='Trendline')
plt.legend()

# Subplot 2: Histogram of exam scores
plt.subplot(1, 2, 2)
plt.hist(exam_scores, bins=10, color='g', edgecolor='black', alpha=0.7)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Scores')
plt.ylabel('Frequency')

# Adjust layout for better fit
plt.tight_layout()

# Show the plots
plt.show()

#32
import matplotlib.pyplot as plt

# Data: Categories and their corresponding percentages
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
percentages = [15, 25, 35, 10, 15]

# Create a pie chart
plt.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700'])

# Add title
plt.title('Distribution of Categories')

# Display the pie chart
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()

#33
import matplotlib.pyplot as plt
import numpy as np

# Generate x values from -10 to 10
x_values = np.linspace(-10, 10, 400)

# Calculate y values using the function y = x^2
y_values = x_values ** 2

# Create the line plot
plt.plot(x_values, y_values, label='y = x^2', color='b')

# Label the axes
plt.xlabel('X Values')
plt.ylabel('Y Values')

# Add a title to the plot
plt.title('Plot of y = x^2')

# Display the plot
plt.grid(True)  # Add gridlines for better readability
plt.legend()    # Show the legend
plt.show()

#34
import numpy as np

# Number of students and subjects
num_students = 5
num_subjects = 3

# Random scores between 0 and 100 for each student and subject
scores = np.random.randint(0, 101, size=(num_students, num_subjects))

# Create an array with students' scores for Mathematics, Science, and English
subjects = ['Mathematics', 'Science', 'English']

# Print the complete dataset
print("Student Scores Dataset:")
print("Subjects: ", subjects)
print(scores)

#35
import matplotlib.pyplot as plt

# Data for the smartphone brands and their respective market shares
brands = ['Samsung', 'Apple', 'Xiaomi', 'vivo', 'OPPO', 'Others']
market_share = [18.4, 15.6, 14.5, 8.8, 8.8, 33.8]

# Create a pie chart
plt.pie(market_share, labels=brands, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700', '#D3D3D3'])

# Add a title
plt.title('Smartphone Brand Market Share (Q2 2024)')

# Display the pie chart
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular
plt.show()

#36
import numpy as np

# Given array of ages
ages = np.array([18, 22, 21, 19, 22, 24, 20, 25, 30, 32, 21, 20, 18, 19, 23])

# 1. Filter the array to find the ages that fall between 20 and 30, inclusive.
filtered_ages = ages[(ages >= 20) & (ages <= 30)]

# 2. Calculate the mean and standard deviation of filtered_ages.
mean_filtered_ages = np.mean(filtered_ages)
std_filtered_ages = np.std(filtered_ages)

# 3. Create a new array adjusted_ages by subtracting the mean of filtered_ages from each element of the original ages array.
adjusted_ages = ages - mean_filtered_ages

# 4. Find the indices of the elements in adjusted_ages that are negative.
negative_indices = np.where(adjusted_ages < 0)[0]

# Display results
print("Filtered Ages (20 to 30):", filtered_ages)
print("Mean of Filtered Ages:", mean_filtered_ages)
print("Standard Deviation of Filtered Ages:", std_filtered_ages)
print("Adjusted Ages:", adjusted_ages)
print("Indices of ages below the mean of the filtered group:", negative_indices)


#q37
array15=np.array([[9,8,7],[6,5,4],[1,2,3]])
sortedrow=np.sort(array15,axis=1)
sortedcolumn=np.sort(sortedrow,axis=0)
print("orginal array:",array15)
print("sorted row<",sortedrow)
print("sorted row and column:",sortedcolumn)
