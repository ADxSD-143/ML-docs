# Logistic Regression — Learning Notes

> A progressive guide from fundamentals to advanced concepts. Each lesson contains intuition, mathematics, examples, and Python practice.

## Learning roadmap

1. [Classification fundamentals](#lesson-1-classification-fundamentals)
2. [Linear Regression vs Logistic Regression](#lesson-2-linear-regression-vs-logistic-regression)
3. [Sigmoid function](#lesson-3-sigmoid-function)
4. [Probability, odds, and log-odds](#lesson-4-probability-odds-and-log-odds)
5. Decision boundary and classification thresholds *(next)*
6. Cost function / Log Loss
7. Maximum Likelihood Estimation (MLE)
8. Gradient Descent and coefficient learning
9. Training Logistic Regression with scikit-learn
10. Evaluation: confusion matrix, precision, recall, F1, ROC-AUC
11. Multiclass Logistic Regression
12. Regularization: L1 and L2
13. Imbalanced classes, threshold tuning, calibration, and practical considerations

---

# Lesson 1: Classification Fundamentals

## 1. What is classification?

Classification is a supervised machine-learning task where a model predicts a discrete class or category from input features.

Examples include spam/not spam, fraud/not fraud, and pass/fail.

### Regression vs classification

- **Regression** predicts a continuous numerical value, such as house price or marks.
- **Classification** predicts a category, such as class `0` or class `1`.

## 2. Binary classification

Binary classification has two possible classes. For example:

- `0` = Fail
- `1` = Pass

`X` represents input features; `y` represents the target.

## 3. Why Logistic Regression?

Linear Regression produces an unrestricted real-valued prediction, so its output is not generally a valid probability. Logistic Regression transforms a linear score using the sigmoid function to estimate the probability of the positive class.

## 4. Python practice: create a dataset

```python
import pandas as pd

data = {
    "study_hours": [1, 2, 3, 5, 6, 8],
    "result":      [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[["study_hours"]]
y = df["result"]

print(df)
print("\nInput features (X):")
print(X)
print("\nTarget (y):")
print(y)
```

Here, `study_hours` is the input feature and `result` is the target. This lesson only creates a dataset; it does not train a model.

## Key takeaways

- Classification predicts discrete classes.
- Binary classification has two target classes.
- Logistic Regression is commonly used for binary classification.

---

# Lesson 2: Linear Regression vs Logistic Regression

## 1. Linear Regression

Linear Regression predicts a continuous numerical value:

$$
\hat{y}=b_0+b_1x
$$

For example, a model might predict marks from study hours. If the equation is:

$$
\widehat{\text{marks}}=30+8x
$$

and $x=5$, then the prediction is $30+8(5)=70$ marks.

## 2. Logistic Regression

Logistic Regression estimates the probability of a class. It first calculates a linear score:

$$
z=b_0+b_1x
$$

It then converts the score to a probability using the sigmoid function:

$$
p=\frac{1}{1+e^{-z}}
$$

For example, let $z=-4+1.2x$. At $x=5$, $z=2$ and the estimated probability is approximately $0.8808$ (88.08%).

A classification threshold, often 0.5 by default, can convert the probability into a class label. The threshold is a decision rule, not part of the sigmoid formula itself.

## 3. Key differences

| Property | Linear Regression | Logistic Regression |
|---|---|---|
| Typical task | Predict a numerical value | Estimate class probability |
| Model output | Any real number | Probability strictly between 0 and 1 for finite scores |
| Common objective | Mean Squared Error | Log Loss / Maximum Likelihood |
| Example | Predict marks or price | Predict pass/fail or spam/not spam |

## 4. Python demonstration

The coefficients below are deliberately specified by hand to illustrate the difference; this is not a trained model.

```python
import math

study_hours = 5

# A hand-set linear model for a numerical prediction
predicted_marks = 30 + 8 * study_hours

# A hand-set logistic score and sigmoid probability
z = -4 + 1.2 * study_hours
probability = 1 / (1 + math.exp(-z))
predicted_class = 1 if probability >= 0.5 else 0

print("Predicted marks:", predicted_marks)
print("Linear score:", z)
print("Estimated positive-class probability:", round(probability, 4))
print("Predicted class:", predicted_class)
```

Expected probability is approximately 0.8808; the class at threshold 0.5 is 1.

## Key takeaway

Linear Regression models a numerical target. Logistic Regression models class probability through a sigmoid transformation of a linear score.

---

# Lesson 3: Sigmoid Function

## 1. Why do we need it?

The linear score $z=b_0+b_1x$ can be any real number, including negative values or values greater than 1. It cannot directly serve as a probability. The sigmoid function maps the score to the open interval $(0,1)$.

## 2. Formula

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Where:
- $z$ is the model's linear score.
- $e\approx 2.71828$ is Euler's number.
- $\sigma(z)$ is the sigmoid output.

## 3. Values and intuition

| Score $z$ | Sigmoid $\sigma(z)$ | Approximate percentage |
|---:|---:|---:|
| -10 | 0.000045 | 0.0045% |
| -3 | 0.0474 | 4.74% |
| -1 | 0.2689 | 26.89% |
| 0 | 0.5000 | 50% |
| 1 | 0.7311 | 73.11% |
| 2 | 0.8808 | 88.08% |
| 3 | 0.9526 | 95.26% |
| 10 | 0.999955 | 99.9955% |

Important properties:
- The output is strictly between 0 and 1 for every finite real input.
- $\sigma(0)=0.5$.
- Large positive scores approach 1; large negative scores approach 0.
- The graph is S-shaped and monotonically increasing.
- A sigmoid output is a model estimate; it is not a guarantee and may not be well-calibrated.

## 4. Numerical example

For $z=2$:

$$
\begin{aligned}
\sigma(2)&=\frac{1}{1+e^{-2}}\\
&\approx\frac{1}{1+0.1353}\\
&\approx 0.8808
\end{aligned}
$$

So the estimated positive-class probability is about 88.08%. With a 0.5 threshold, the predicted class is 1.

For $z=1$, $\sigma(1)\approx0.7311$, **not** 1. Positive scores usually yield probabilities greater than 0.5, but finite scores do not produce an exact probability of 1.

## 5. Python implementation

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

scores = [-10, -3, -1, 0, 1, 2, 3, 10]

for z in scores:
    print(f"Score: {z:>3} | Probability: {sigmoid(z):.4f}")
```

This simple implementation is suitable for learning. For extreme scores, `math.exp(-z)` can overflow; numerically stable implementations should be used in robust production code.

## Key takeaway

Logistic Regression applies sigmoid to a linear score to obtain an estimated positive-class probability. A separate threshold turns that probability into a class prediction.

---

# Lesson 4: Probability, Odds, and Log-Odds

## 1. Probability

Let $p=P(y=1\mid X)$ be the model's estimated probability of the positive class. In binary classification, the probability of the other class is $1-p$.

For example, if $p=0.8$, the positive-class probability is 80% and the negative-class probability is 20%.

## 2. Odds

Odds compare the probability of an event with the probability of it not occurring:

$$
\text{Odds}=\frac{p}{1-p}
$$

For $p=0.8$:

$$
\text{Odds}=\frac{0.8}{1-0.8}=\frac{0.8}{0.2}=4
$$

Odds are 4:1 in favour of the event. Probability and odds are related but are not the same quantity.

- If $p=0.5$, odds are 1:1.
- If $p<0.5$, odds are less than 1.
- If $p>0.5$, odds are greater than 1.

## 3. Log-odds (logit)

Log-odds are the natural logarithm of odds:

$$
\mathrm{logit}(p)=\ln\left(\frac{p}{1-p}\right)
$$

For $p=0.8$, odds are 4, so:

$$
\mathrm{logit}(0.8)=\ln(4)\approx1.3863
$$

For probabilities strictly between 0 and 1, log-odds can take any real value.

| Probability $p$ | Odds $p/(1-p)$ | Log-odds |
|---:|---:|---:|
| 0.1 | 0.1111 | -2.1972 |
| 0.5 | 1 | 0 |
| 0.8 | 4 | 1.3863 |
| 0.9 | 9 | 2.1972 |

At $p=0$ or $p=1$, log-odds are undefined as finite values.

## 4. Connection to Logistic Regression

Binary Logistic Regression models the log-odds as a linear function of its input features:

$$
\ln\left(\frac{p}{1-p}\right)=b_0+b_1x
$$

With multiple features:

$$
\ln\left(\frac{p}{1-p}\right)
=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

Let the right-hand side be the linear score $z$. Solving for $p$ gives:

$$
p=\frac{1}{1+e^{-z}}
$$

This is the sigmoid function from Lesson 3. The coefficients are learned from training data in a fitted model.

## 5. Python practice

```python
import math

probability = 0.8

if not 0 < probability < 1:
    raise ValueError("Probability must be strictly between 0 and 1.")

odds = probability / (1 - probability)
log_odds = math.log(odds)
recovered_probability = 1 / (1 + math.exp(-log_odds))

print("Probability:", probability)
print("Odds:", odds)
print("Log-odds:", round(log_odds, 4))
print("Recovered probability:", round(recovered_probability, 4))
```

Expected results: odds = 4.0, log-odds $\approx1.3863$, recovered probability = 0.8.

## Key takeaway

Logistic Regression models log-odds linearly and uses the sigmoid function to obtain a probability estimate. Odds are a probability ratio; log-odds are the natural logarithm of that ratio.

---

## Progress tracker

- [x] Lesson 1 — Classification fundamentals
- [x] Lesson 2 — Linear vs Logistic Regression
- [x] Lesson 3 — Sigmoid function
- [x] Lesson 4 — Probability, odds, and log-odds
- [ ] Lesson 5 — Decision boundary and classification thresholds
- [ ] Lesson 6 — Log Loss / cost function
- [ ] Lesson 7 — Maximum Likelihood Estimation
- [ ] Lesson 8 — Gradient Descent and coefficient learning
- [ ] Lesson 9 — scikit-learn implementation
- [ ] Lesson 10 — Evaluation metrics
- [ ] Lesson 11 — Multiclass Logistic Regression
- [ ] Lesson 12 — L1/L2 regularization
- [ ] Lesson 13 — Imbalance, threshold tuning, calibration, and practical considerations
