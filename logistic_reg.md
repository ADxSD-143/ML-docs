
# Logistic Regression — Lesson 1: Classification

## 1. What is Classification?

Classification is a supervised machine learning task in which a model predicts a discrete class or category from input features.

Examples:
- Spam or Not Spam
- Fraud or Not Fraud
- Pass or Fail

## 2. Regression vs Classification

**Regression:** Predicts continuous numerical values, such as house prices or salaries.

**Classification:** Predicts discrete classes, such as 0 or 1.

## 3. Binary Classification

Binary classification involves two possible classes.

Example:
- 0 = Fail
- 1 = Pass

Here, `X` represents the input features, and `y` represents the target variable.

## 4. Why Logistic Regression?

Linear Regression can produce predictions outside the range [0, 1]. Therefore, its output cannot always be interpreted as a probability.

Logistic Regression uses the sigmoid function to map a linear combination of features to a value between 0 and 1.

This value can be interpreted as the estimated probability of the positive class.

## 5. Key Takeaways

- Classification predicts categories.
- Logistic Regression is commonly used for binary classification.
- Input features are represented by X.
- The target variable is represented by y.
- The sigmoid function converts the model's linear score into a probability.

## 6. Python Practice

Created a sample dataset using Pandas with:
- `study_hours` as the input feature.
- `result` as the target variable.

The target uses 0 for Fail and 1 for Pass.

**Next:** Understanding the difference between Linear Regression and Logistic Regression.


## Lesson 2: Linear Regression vs Logistic Regression

### Linear Regression

Linear Regression predicts continuous numerical values.

Equation:

\[
\hat{y}=b_0+b_1x
\]

Example: Predicting a student's marks from study hours.

The output can be any real number.

### Logistic Regression

Logistic Regression estimates the probability of a class, commonly for binary classification.

First, it calculates a linear score:

\[
z=b_0+b_1x
\]

Then, the sigmoid function transforms the score into a probability:

\[
p=\frac{1}{1+e^{-z}}
\]

The probability lies between 0 and 1.

A classification threshold, commonly 0.5, converts the probability into a predicted class.

### Key Differences

- Linear Regression predicts numerical values.
- Logistic Regression estimates class probabilities.
- Logistic Regression applies the sigmoid function to a linear score.
- Logistic Regression commonly uses Log Loss, whereas Linear Regression commonly uses Mean Squared Error.

### Python Practice

Calculated predicted marks using a linear equation and estimated a binary class probability using the sigmoid function.

Note: The coefficients were manually specified for demonstration; no model was trained.

**Next lesson:** The Sigmoid Function — intuition, mathematical properties, and Python implementation.

# Lesson 3: Sigmoid Function

## 1. What is the Sigmoid Function?

The sigmoid function converts a real-valued input into an output between 0 and 1. Logistic Regression uses it to estimate the probability of the positive class.

## 2. Mathematical Formula

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Where:
- \(z\) = linear score calculated by the model.
- \(e\) = Euler's number, approximately 2.71828.
- \(\sigma(z)\) = sigmoid output.

## 3. Important Properties

- Output lies strictly between 0 and 1 for every finite real input.
- If \(z=0\), then \(\sigma(z)=0.5\).
- For large positive inputs, the output approaches 1.
- For large negative inputs, the output approaches 0.
- The graph is S-shaped and monotonically increasing.

## 4. Numerical Example

For \(z=2\):

\[
\sigma(2)=\frac{1}{1+e^{-2}}\approx 0.8808
\]

The estimated positive-class probability is approximately 88.08%.

With a classification threshold of 0.5, the predicted class is 1.

## 5. Python Implementation

Implemented the sigmoid function using Python's `math.exp()` and evaluated it for different input scores.

## 6. Key Takeaway

Logistic Regression calculates a linear score and applies the sigmoid function to obtain a probability estimate. A separate decision threshold can convert that probability into a class prediction.

**Next:** Understanding probability, odds and log-odds in Logistic Regression.
