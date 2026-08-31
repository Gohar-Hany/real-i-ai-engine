## Digilians


## Math Basics

Session 1: (Statistics Fundamentals)


## Session Agenda

- Introduction to Statistics
- Descriptive Statistics


## Introduction to Statistics


## Importance of Statistics in Al and Machine Learning

## Data Analysis

Statistics provide the tools to analyze and interpret data, which is the backbone of A/. Understanding data distributions, variability, and patterns is crucial for building robust models.

## Decision Making

Al models often make decisions based on probabilities. Understanding probability and statistical inference is key to trusting and explaining these decisions:

## Model Evaluation

In AI, especially in machine learning, evaluating the performance of models (e.g., accuracy, precision, recall) relies on statistical measures:

## Real-World Applications

From predicting customer behavior to diagnosing diseases in healthcare, statistics provide the foundation for analyzing data and making informed decisions.


## Population

The entire group of individuals or items that you're interested in studying:

Example: If you want to study the average height of adult males in a country, the population would be all adult males in that country.

## Basic Statistical Definitions

Population vS: Sample

## Sample

A subset of the population that is used to represent the whole.

Example: Measuring the height of 1,000 adult males from different regions could be a sample representing the entire population:


## Basic Statistical Definitions

## Population vs: Sample

Population: The entire group of individuals or items that you're interested in studying: For example, if you want to study the average height of adult males in a country, the population would be all adult males in that country.

Sample: A subset of the population that is used to represent the whole. For instance, measuring the height of 1,000 adult males from different regions could be a sample representing the entire population: Sampling is often done because it is impractical or impossible to study the whole population.


## Types of Variables

## Categorical (Qualitative) Variables

These variables represent categories or groups and can be either:

- Nominal: No inherent order among categories Examples include colors (red, blue, green) , gender (male, female):
- Ordinal: There is an inherent order, but the difference between levels is not quantifiable. Examples include customer satisfaction ratings (poor, fair, good, excellent) , education levels (high school, bachelors, master's):


## Types of Variables

## Numerical (Quantitative) Variables

These variables represent measurable quantities and can be either:

- Discrete: Countable, finite values. Examples include the number of students in a class, the number of cars in a parking lot:
- Continuous: Infinite possibilities within a range. Examples include height, weight, temperature.


## Real-World Relevance and Applications

## Examples of Statistics in Al Applications

- Fraud Detection: Analyzing transaction data to identify patterns that indicate fraudulent activity. Here, statistics help to detect anomalies that deviate from normal behavior.
- Recommendation Systems: E-commerce platforms Iike Amazon use statistics to analyze customer preferences and recommend products based on what similar customers have purchased:
- Healthcare Analytics: Predicting the likelihood of diseases based on patient data. Statistical models can assess risk factors and provide personalized healthcare recommendations:


---


## Descriptive Statistics


## Descriptive Statistics

Descriptive statistics help provide a clear understanding of data through numerical calculations, graphs, and tables. This is a crucial step before conducting any further statistical analysis or building machine learning models.


## Cases VS. Variables

Cases: A case represents an individual entity or subject in a dataset on which measurements or observations are made. It can be a person, a country, an event; or any other subject of study:

Variables: A variable is a characteristic or attribute that can be measured or observed for each case. Variables are features or properties that describe the cases.


|    |   Variables | Variables   | Variables   |
|----|-------------|-------------|-------------|
|    |             | Customers   | Refunds     |
|  1 |          14 | 9           | 3           |
|  2 |          19 | 13          | 4           |
|  3 |          22 | 19          | 4           |
|  4 |          24 | 20          | 3           |
|  5 |          29 | 26          | 8           |
|  6 |          40 | 34          | 6           |


## Data Matrix vs. Frequency Table

## Data Matrix

A data matrix is a structured table where each row represents a single case (individual data point) , and each column represents a variable (characteristic or feature of the cases)

| Variables   | Variables         | Variables   | Variables           | Variables           | Variables   | Variables   | Variables   |
|-------------|-------------------|-------------|---------------------|---------------------|-------------|-------------|-------------|
|             | sepal Length epal | width petal | length Ipetal width | class               |             |             |             |
|             | 5.1               | 3.5         | 1.4                 | 0.2/Iris-setosa     |             |             |             |
|             | 4.9               | 3           | 1.4                 | 0.2/Iris-setosa     |             |             |             |
|             | 6 .5              | 3.2         | 5.1                 | 2uris-Virginica     |             |             |             |
|             | 6 . 4             | 2.7         | 5.3                 | L9uris-Virginica    |             |             |             |
|             | 6.8               | 3           | 5.5                 | 21uris-virginica    |             |             |             |
|             | 6 .7              | 3.1         | 44                  | 1.4 Iris-versicolor |             |             |             |
|             | 5.6               | 3           | 4.5                 | 1.5 Iris-versicolor |             |             |             |
|             | 5.3               | 2.7         | 41                  | Iris-versicolor     |             |             |             |


## Data Matrix vs. Frequency Table

## Data Matrix

- Purpose: The data matrix is useful for raw data representation where each case's specific details are important: It's a comprehensive way to store all the information but not necessarily the best for summarizing data.
- When to Use: Data matrices are used in scenarios where you need to keep all details about each individual case, such as during data collection or when you need to perform case-by-case analysis:


## Data Matrix vs. Frequency Table

## Frequency Table

A frequency table is a summary of the data that shows how often each value of a variable occurs: It can display frequencies, percentages, and cumulative percentages:

| Score Frequency   |
|-------------------|
| 50-59 2           |
| 60-69 2           |
| 70-79 6           |
| 80-89 7           |
| 90-99 3           |


## Data Matrix vs. Frequency Table

## Frequency Table

- Purpose: Frequency tables are used to summarize and visualize the distribution of a variable, making it easier to see patterns and understand the data's structure.
- When to Use: When you want to quickly grasp how data is distributed across different categories or ranges, especially when dealing with categorical or quantitative variables:


## Descriptive Statistics

## Measures of Central Tendency

Understand how to describe the center of a dataset using different measures and learn how to calculate and interpret these values.


## Measures of Central Tendency

## Mean (Average)

The mean is the sum of all values divided by the number of values. It is a commonly used measure of central tendency:

## Formula

## Example

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The mean is sensitive to outliers. If the dataset has extreme values (very high or very low) , the mean can be misleading


## Median

The median is the middle value of a dataset when it is ordered from smallest to largest. If the dataset has an even number of values, the median is the average of the two middle values:

## Example

For the dataset [4,8,6,5,3,4], first sort the data:[3,4,4,5,6,8]. The median is the average of the two middle numbers, 4 and 5, which is 4+5 4.5

The median is not affected by outliers and provides a better measure of central tendency for skewed distributions.

## Measures of Central Tendency


---


## Mode

The mode is the value that appears most frequently in a dataset:

## Example

In the dataset [4,8,6,5,3,4], the mode is 4 because it appears twice, more than other number. any

The mode is useful for categorical data and for identifying the most common item in a dataset. A dataset can have more than one mode (bimodal or multimodal) or no mode at all:

## Measures of Central Tendency


## Measures of Dispersion

Learn how to describe the spread or variability of a dataset using different measures.


## Range

The range is the difference between the maximum and minimum values in a dataset:

## Formula

Range = Max value Min value.

## Example

For the dataset [4,8,6,5,3,4], the range is 8-3-5.

The range gives a quick sense of the spread of the data but is highly sensitive to outliers.

## Measures of Dispersion


## Measures of Dispersion

## Interquartile Range (IQR)

The IQR is the range between the first quartile (Q1) and the third quartile (Q3). It measures the spread of the middle 50% of the data, effectively capturing the central tendency without being influenced by extreme values or outliers. The IQR is calculated as:

<!-- formula-not-decoded -->

- Q1 (First Quartile): The value below which 25% of the data falls.
- Q3 (Third Quartile): The value below which 75% of the data falls:


## Interquartile Range (IQR)

## Why Use IQR?

- Robustness Against Outliers: Unlike the range, which considers only the minimum and maximum values, the IQR focuses on the middle 50% of the data: This makes it less sensitive to outliers and extreme values, providing a more reliable measure of dispersion for skewed distributions.
- Understanding Data Spread: The IQR helps understand the variability of the central portion of the data: A larger IQR indicates more spread in the middle 50% of the dataset, while a smaller IQR suggests that the data points are closer to the median.


## Interquartile Range (IQR)

## Example Calculation of IQR

Let's use a small dataset to illustrate: Dataset: [2,4,4,5,6,8,9]

1. Arrange the data in ascending order (already sorted in this case):
2. Find Q1 and Q3:
3. Q1 (First Quartile): The median of the first half of the dataset (excluding the median if the number of data points is odd): For [2,4,4], Q1 =4
4. Q3 (Third Quartile): The median of the second half of the dataset: For [6,8,9], Q3 = 8
3. Calculate IQR:

<!-- formula-not-decoded -->


## Variance

Variance measures the average squared deviation of each number from the mean. It provides insight into the spread of all data points around the mean:

## Formula

<!-- formula-not-decoded -->

## Example

For the dataset [4,8,6,5,3,4], calculate the mean (0=5), then compute the variance:

<!-- formula-not-decoded -->

Variance is in squared units, making it less interpretable in the original scale. However, it is fundamental in statistical theorv.

## Measures of Dispersion


## Measures of Dispersion

## Standard Deviation

The standard deviation is the square root of the variance and provides a measure of the average distance from the mean: It is in the same units as the data, making it more interpretable.

## Formula

## Example

For the dataset [4,8,6,5,3,4], the standard deviation is:

<!-- formula-not-decoded -->

A small standard deviation indicates that the values are close to the mean, while a large standard deviation indicates that the values are spread out over a wider range.


## Descriptive Statistics

Data Distribution and Visualization

Learn how to visualize data distribution and understand its shape using graphical representations.


## Descriptive Statistics

## Data Distribution and Visualization

Histograms: A histogram is a graphical representation that organizes a group of data points into user-specified ranges. It shows the frequency distribution of a dataset

Example: Create a histogram for a dataset representing the number of books read by students in a class.


Histograms help visualize the shape of the data distribution (e.g , normal, skewed) and identify patterns like bimodality or skewness.


---


## Descriptive Statistics

## Data Distribution and Visualization

Box Plots (Box-and-Whisker Plots): A box plot displays the distribution of data based on a five-number summary: minimum, first quartile (Q1) , median (Q2) , third quartile (Q3) , and maximum:

Example: Create a box for a dataset of test scores to identify the median score, quartiles, and potential outliers. plot

## Quartiles:


First Quartile (Q1, 25th percentile): The left edge of the box represents the 25th percentile, indicating that 25% of the students scored below this value.

Third Quartile (Q3, 75th percentile): The right edge of the box represents the 75th percentile, showing that 75% of the students scored below this value.
