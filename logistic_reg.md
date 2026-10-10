# Logistic Regression — Learning Notes

> A progressive guide from fundamentals to advanced concepts. Each lesson contains intuition, mathematics, examples, and Python practice.

## Learning roadmap

1. [Classification fundamentals](#lesson-1-classification-fundamentals)
2. [Linear Regression vs Logistic Regression](#lesson-2-linear-regression-vs-logistic-regression)
3. [Sigmoid function](#lesson-3-sigmoid-function)
4. [Probability, odds, and log-odds](#lesson-4-probability-odds-and-log-odds)
5. [Decision boundary and classification thresholds](#lesson-5-decision-boundary-and-classification-thresholds)
6. [Log Loss / Binary Cross-Entropy](#lesson-6-log-loss--binary-cross-entropy)
7. [Maximum Likelihood Estimation (MLE)](#lesson-7-maximum-likelihood-estimation-mle)
8. [Gradient Descent and coefficient learning](#lesson-8-gradient-descent-and-coefficient-learning)
9. [scikit-learn Logistic Regression training](#lesson-9-scikit-learn-logistic-regression--training-and-evaluation)
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


```text
ŷ = b₀ + b₁x
```


For example, a model might predict marks from study hours. If the equation is:


```text
predicted marks = 30 + 8x
```


and x = 5, then the prediction is 30 + 8(5) = 70 marks.

## 2. Logistic Regression

Logistic Regression estimates the probability of a class. It first calculates a linear score:


```text
z = b₀ + b₁x
```


It then converts the score to a probability using the sigmoid function:


```text
p = 1 / (1 + e^(−z))
```


For example, let z = −4 + 1.2x. At x = 5, z = 2 and the estimated probability is approximately 0.8808 (88.08%).

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

The linear score z = b₀ + b₁x can be any real number, including negative values or values greater than 1. It cannot directly serve as a probability. The sigmoid function maps the score to the open interval (0,1).

## 2. Formula


```text
σ(z) = 1 / (1 + e^(−z))
```


Where:
- z is the model's linear score.
- e ≈ 2.71828 is Euler's number.
- σ(z) is the sigmoid output.

## 3. Values and intuition

| Score z | Sigmoid σ(z) | Approximate percentage |
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
- σ(0) = 0.5.
- Large positive scores approach 1; large negative scores approach 0.
- The graph is S-shaped and monotonically increasing.
- A sigmoid output is a model estimate; it is not a guarantee and may not be well-calibrated.

## 4. Numerical example

For z = 2:


```text
σ(2) = 1 / (1 + e^(−2)) ≈ 1 / (1 + 0.1353) ≈ 0.8808
```


So the estimated positive-class probability is about 88.08%. With a 0.5 threshold, the predicted class is 1.

For z = 1, σ(1) ≈ 0.7311, **not** 1. Positive scores usually yield probabilities greater than 0.5, but finite scores do not produce an exact probability of 1.

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

Let p = P(y = 1| X) be the model's estimated probability of the positive class. In binary classification, the probability of the other class is 1 − p.

For example, if p = 0.8, the positive-class probability is 80% and the negative-class probability is 20%.

## 2. Odds

Odds compare the probability of an event with the probability of it not occurring:


```text
Odds = p / (1 − p)
```


For p = 0.8:


```text
Odds = 0.8 / 0.2 = 4
```


Odds are 4:1 in favour of the event. Probability and odds are related but are not the same quantity.

- If p = 0.5, odds are 1:1.
- If p < 0.5, odds are less than 1.
- If p > 0.5, odds are greater than 1.

## 3. Log-odds (logit)

Log-odds are the natural logarithm of odds:


```text
logit(p) = ln(p / (1 − p))
```


For p = 0.8, odds are 4, so:


```text
logit(0.8) = ln(4) ≈ 1.3863
```


For probabilities strictly between 0 and 1, log-odds can take any real value.

| Probability p | Odds p / (1 − p) | Log-odds |
|---:|---:|---:|
| 0.1 | 0.1111 | -2.1972 |
| 0.5 | 1 | 0 |
| 0.8 | 4 | 1.3863 |
| 0.9 | 9 | 2.1972 |

At p = 0 or p = 1, log-odds are undefined as finite values.

## 4. Connection to Logistic Regression

Binary Logistic Regression models the log-odds as a linear function of its input features:


```text
ln(p / (1 − p)) = b₀ + b₁x
```


With multiple features:


```text
ln(p / (1 − p)) = b₀ + b₁x₁ + b₂x₂ + … + bₙxₙ
```


Let the right-hand side be the linear score z. Solving for p gives:


```text
p = 1 / (1 + e^(−z))
```


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

Expected results: odds = 4.0, log-odds ≈ 1.3863, recovered probability = 0.8.

## Key takeaway

Logistic Regression models log-odds linearly and uses the sigmoid function to obtain a probability estimate. Odds are a probability ratio; log-odds are the natural logarithm of that ratio.


---

# Lesson 5: Decision Boundary and Classification Thresholds

## 1. Probability is not the same as a class prediction

Logistic Regression first estimates the probability of the positive class:


```text
p = 1 / (1 + e^(−z))
```


It then uses a **decision threshold** to convert that probability into a class label. A common default threshold is 0.5:

- If p ≥ 0.5, predict class 1.
- If p < 0.5, predict class 0.

The threshold is a decision rule applied after probability estimation. Changing it does not, by itself, retrain the model or change its predicted probabilities.

## 2. What is a decision boundary?

A decision boundary is the point, line, or surface where a model switches from predicting one class to the other.

For a single-feature Logistic Regression model:


```text
z = b₀ + b₁x
```


At threshold 0.5, the boundary occurs where p = 0.5. Since the sigmoid function equals 0.5 when z = 0, the boundary equation is:


```text
b₀ + b₁x = 0
```


Solving for x (when b₁ ≠ 0):


```text
x = −b₀ / b₁
```


For multiple features, the 0.5-threshold boundary is:


```text
b₀ + b₁x₁ + b₂x₂ + … + bₙxₙ = 0
```


This is a hyperplane in the feature space. With one feature it is a point; with two features it is a line; with three features it is a plane. If nonlinear features are engineered, the boundary may look nonlinear when plotted against the original features.

## 3. Numerical example

Suppose the model's score is:


```text
z = −4 + 1.2x
```


Here, x is study hours. At threshold 0.5, set z = 0:


```text
−4 + 1.2x = 0 → 1.2x = 4 → x = 4 / 1.2 ≈ 3.333
```


So the 0.5-threshold boundary is approximately 3.33 study hours for this illustrative model.

- Below 3.33 hours, the model predicts class 0.
- Above 3.33 hours, it predicts class 1.
- At the boundary, the estimated probability is exactly 0.5; using the rule p ≥ 0.5 predicts class 1 at equality.

This is an illustration with manually chosen coefficients, not a real or validated model of exam outcomes.

## 4. What happens when the threshold changes?

For a general threshold t where 0<t<1, the boundary satisfies p = t. Inverting the sigmoid gives:


```text
z = ln(t / (1 − t))
```


For z = b₀ + b₁x, the boundary is therefore:


```text
x = [ln(t / (1 − t)) − b₀] / b₁
```


Using z = −4 + 1.2x:

**Threshold t = 0.5**


```text
z = ln(0.5 / 0.5) = ln(1) = 0
```


The boundary is x ≈ 3.333.

**Threshold t = 0.8**


```text
z = ln(0.8 / 0.2) = ln(4) ≈ 1.3863
```



```text
x = (1.3863 + 4) / 1.2 ≈ 4.489
```


A higher threshold makes the model require a higher estimated probability before predicting class 1. For this model, the boundary moves from about 3.33 to 4.49 study hours.

## 5. Python practice: compare thresholds

The coefficients are specified by hand to demonstrate the mechanics. This code does not train a model.

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict_probability(study_hours):
    z = -4 + 1.2 * study_hours
    return sigmoid(z)

thresholds = [0.5, 0.8]
study_hours_values = [2, 3, 10 / 3, 4, 5, 6]

for hours in study_hours_values:
    probability = predict_probability(hours)

    predictions = [
        int(probability >= threshold)
        for threshold in thresholds
    ]

    print(
        f"Hours: {hours:5.2f} | "
        f"Probability: {probability:.4f} | "
        f"Class at 0.5: {predictions[0]} | "
        f"Class at 0.8: {predictions[1]}"
    )
```

Notice that the probability stays the same for each study-hours value. Only the class label can change when the threshold changes.

## 6. How to choose a threshold

A threshold of 0.5 is common, but it is not always the best choice. The appropriate threshold depends on the problem and the cost of different errors.

- **Lower threshold:** more observations are likely to be predicted as positive. This often helps recall, but can create more false positives.
- **Higher threshold:** fewer observations are predicted as positive. This often reduces false positives, but can miss more actual positives.
- **Threshold selection:** choose it using validation data and a metric or cost that matches the task. Do not tune the threshold on the final test set.

For example, a screening system may prioritize catching as many positive cases as possible, while a costly manual-review workflow may need to limit false alarms. The appropriate choice must be justified by the application's costs and requirements.

## Key takeaways

- Probability estimation and class prediction are two separate steps.
- The threshold converts probabilities into class labels.
- At threshold 0.5, the decision boundary is where the linear score z = 0.
- For any threshold t, the boundary score is ln(t / (1 − t)).
- A threshold changes class labels, not the model's learned coefficients or probability estimates.
- Thresholds should be selected on validation data with the task's error costs in mind.

**Next:** Lesson 6 — Log Loss (the Logistic Regression cost function).


---

# Lesson 6: Log Loss / Binary Cross-Entropy

## 1. Why does Logistic Regression need a loss function?

Logistic Regression estimates a probability, but we also need a way to measure how good that prediction is compared with the true label. A **loss function** assigns a numerical penalty to a prediction.

For binary classification:
- y is the true label and is either 0 or 1.
- p is the model's predicted probability that the label is 1.
- The loss should be small when the model assigns high probability to the true class.
- The loss should be large when the model assigns low probability to the true class, especially when it is confidently wrong.

Log Loss is also called **Binary Cross-Entropy (BCE)** for the binary classification setting.

## 2. The formula

For one labelled example, the binary log loss is:


```text
L(y, p) = −[y × ln(p) + (1 − y) × ln(1 − p)]
```


Here, ln is the natural logarithm.

This single formula handles both possible labels.

### Case A: the true label is y = 1

Substitute y = 1:


```text
L(1, p) = −ln(p)
```


Only the predicted probability of class 1 matters. If p is close to 1, the loss is small. If p is close to 0, the loss becomes large.

### Case B: the true label is y = 0

Substitute y = 0:


```text
L(0, p) = −ln(1 − p)
```


Now the model is rewarded for assigning a high probability to class 0, which is 1 − p.

## 3. Numerical examples

### True label is 1

If y = 1 and p = 0.9:


```text
L = −ln(0.9) ≈ 0.1054
```


This is a low loss because the model assigned 90% probability to the correct class.

If y = 1 and p = 0.1:


```text
L = −ln(0.1) ≈ 2.3026
```


This is much larger because the model assigned only 10% probability to the correct class.

### True label is 0

If y = 0 and p = 0.2, the probability assigned to the true class is 1 − p = 0.8:


```text
L = − ln(1 − 0.2) = −ln(0.8) ≈ 0.2231
```


The prediction is reasonably good, so the loss is relatively small.

## 4. Average loss across a dataset

For n examples, calculate the loss for each example and take the mean:


```text
J = −(1/n) Σ(i = 1…n) [yᵢ × ln(pᵢ) + (1 − yᵢ) × ln(1 − pᵢ)]
```


Where:
- n is the number of training examples.
- yᵢ is the true label for example i.
- pᵢ is the predicted probability of class 1 for example i.
- J is the average Log Loss over the dataset.

Training aims to find model parameters that minimize this objective, usually with an optimization algorithm. The precise training setup can include regularization, which we will study later.

## 5. Calculate average Log Loss by hand

Suppose the true labels and predicted probabilities are:

| Example | True label y | Predicted probability p | Loss |
|---:|---:|---:|---:|
| 1 | 1 | 0.9 | −ln(0.9) ≈ 0.1054 |
| 2 | 0 | 0.2 | −ln(0.8) ≈ 0.2231 |
| 3 | 1 | 0.7 | −ln(0.7) ≈ 0.3567 |

The average loss is:


```text
J = (0.1054 + 0.2231 + 0.3567) / 3 ≈ 0.2284
```


This number measures the average probability penalty for these predictions. **Lower Log Loss is better when evaluating comparable predictions on the same target data.** Always compare models on the same evaluation examples and avoid using the test set to tune the model.

## 6. Why does Log Loss penalize confident mistakes?

The logarithm explains the behaviour:

- Correct and confident prediction: the probability assigned to the true class is near 1, so its negative log is near 0.
- Uncertain prediction: the true class receives a middling probability, so the loss is positive.
- Confidently wrong prediction: the true class receives a probability near 0, so the negative log becomes very large.

For example, with true label y = 1, predicting p = 0.01 gives:


```text
L = −ln(0.01) ≈ 4.6052
```


That is much worse than predicting p = 0.9, which gives a loss of only about 0.1054.

Log Loss is a **proper scoring rule**: in expectation, it rewards reporting honest probabilities when the evaluated distribution matches the one being predicted. It does not mean every model's probabilities are automatically calibrated.

## 7. Connection to likelihood

For one binary observation, the Bernoulli probability of observing label y when the model predicts probability p is:


```text
P(y | p) = p^y × (1 − p)^(1 − y)
```


Taking the natural logarithm gives:


```text
ln P(y | p) = y × ln(p) + (1 − y) × ln(1 − p)
```


Negating that log probability produces the binary log loss:


```text
L(y, p) = −ln P(y | p)
```


Across independent labelled examples, minimizing the sum of negative log probabilities is equivalent to maximizing the likelihood of the observed labels. This is the bridge to **Maximum Likelihood Estimation (MLE)**, which is the next major theory lesson.

## 8. Python implementation from scratch

This code computes each example's loss and the average. Probabilities are clipped to avoid evaluating the logarithm at exactly 0 due to numerical rounding or invalid extreme predictions.

```python
import math


def binary_log_loss(y_true, probability, epsilon=1e-15):
    if y_true not in (0, 1):
        raise ValueError("y_true must be 0 or 1.")

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1.")

    # Avoid log(0) in numerical calculations.
    p = min(max(probability, epsilon), 1 - epsilon)

    return -(y_true * math.log(p)
             + (1 - y_true) * math.log(1 - p))


y_true = [1, 0, 1]
predicted_probabilities = [0.9, 0.2, 0.7]

losses = [
    binary_log_loss(y, p)
    for y, p in zip(y_true, predicted_probabilities)
]

average_loss = sum(losses) / len(losses)

for index, loss in enumerate(losses, start=1):
    print(f"Example {index}: loss = {loss:.4f}")

print(f"Average Log Loss = {average_loss:.4f}")
```

Expected output (rounded):

```text
Example 1: loss = 0.1054
Example 2: loss = 0.2231
Example 3: loss = 0.3567
Average Log Loss = 0.2284
```

The clipping is for numerical safety. It should not be used to hide invalid model outputs or data problems.

## 9. Why not simply use Mean Squared Error?

Mean Squared Error can be calculated for probability predictions, but binary Log Loss is the standard objective for Logistic Regression because it follows directly from the Bernoulli likelihood.

Log Loss:
- Is aligned with maximum-likelihood estimation for binary labels.
- Penalizes assigning very low probability to the actual class.
- Uses the full probability prediction rather than only the final thresholded class.
- Can distinguish models that make the same class predictions but assign different probabilities.

Accuracy alone cannot tell these probability-quality differences apart. In practice, use several relevant evaluation metrics rather than relying on a single number.

## Key takeaways

- Binary Log Loss measures the penalty for predicted probabilities relative to true binary labels.
- For y = 1, loss is − ln(p).
- For y = 0, loss is − ln(1 − p).
- Dataset Log Loss is the mean of the individual losses.
- Confidently wrong predictions receive a large penalty.
- Minimizing Log Loss is equivalent to maximizing the likelihood of the observed labels in the unregularized binary model.
- Numerical implementations need to handle probabilities at the extremes safely.

**Next:** Lesson 7 — Maximum Likelihood Estimation (MLE) and why minimizing Log Loss learns Logistic Regression parameters.

---

---

# Lesson 7: Maximum Likelihood Estimation (MLE)

## 1. Why do we need MLE?

In previous lessons, Logistic Regression learned to turn a linear score into a probability:

z = b₀ + b₁x₁ + b₂x₂ + … + bₙxₙ

p = 1 / (1 + e^(−z))

But this raises a key question: **which values of the coefficients b₀, b₁, …, bₙ should the model choose?**

Maximum Likelihood Estimation (MLE) is a principle for choosing parameter values that make the observed training labels most likely under the model.

For Logistic Regression, the model estimates a probability pᵢ for each training example. MLE selects coefficients that maximize the probability of observing the labels we actually have.

## 2. Bernoulli probability for one binary label

For a binary target y, the label must be either 0 or 1. If the model predicts probability p for class 1, then:

- Probability of observing y = 1 is p.
- Probability of observing y = 0 is 1 − p.

Both cases can be represented with one expression:

P(y | p) = pʸ × (1 − p)^(1 − y)

Check the two cases:

- If y = 1: P(y | p) = p¹ × (1 − p)⁰ = p.
- If y = 0: P(y | p) = p⁰ × (1 − p)¹ = 1 − p.

This is the Bernoulli probability model. In Logistic Regression, p is not a fixed constant; it comes from the model's input features and coefficients.

## 3. Likelihood across a dataset

Suppose the dataset has n labelled observations. For each example i, the model predicts probability pᵢ and observes label yᵢ.

Assuming the observations are independent given the model, the likelihood is the product of their individual probabilities:

L(b) = product for i = 1 to n of [pᵢ^yᵢ × (1 − pᵢ)^(1 − yᵢ)]

Here, b represents the collection of model coefficients. Each pᵢ depends on those coefficients:

pᵢ = 1 / (1 + e^(−zᵢ))

zᵢ = b₀ + b₁xᵢ₁ + b₂xᵢ₂ + … + bₙxᵢₙ

**Important distinction:** Likelihood is a function of the model parameters for the observed data. It is not the probability that the parameters themselves are true.

MLE chooses the coefficient values that maximize L(b):

b_MLE = argmax over b of L(b)

The likelihood is a product of values between 0 and 1. As the dataset grows, multiplying many probabilities can produce extremely small numbers, which can cause numerical underflow.

## 4. Why use log-likelihood?

The natural logarithm is strictly increasing. Therefore, maximizing the likelihood and maximizing its natural logarithm select the same parameter values.

Define the log-likelihood:

ℓ(b) = ln(L(b))

Using the rule ln(a × c) = ln(a) + ln(c), the product becomes a sum:

ℓ(b) = Σ(i = 1…n) [yᵢ × ln(pᵢ) + (1 − yᵢ) × ln(1 − pᵢ)]

This is easier to calculate and work with mathematically than a product of many small probabilities.

MLE therefore chooses:

b_MLE = argmax over b of ℓ(b)

## 5. Deriving Log Loss from MLE

Optimization algorithms commonly minimize an objective, so we take the negative log-likelihood:

NLL(b) = −ℓ(b)

NLL(b) = −Σ(i = 1…n) [yᵢ × ln(pᵢ) + (1 − yᵢ) × ln(1 − pᵢ)]

If we divide by the number of examples n, we get the average binary Log Loss:

J(b) = −(1/n) Σ(i = 1…n) [yᵢ × ln(pᵢ) + (1 − yᵢ) × ln(1 − pᵢ)]

This is the Binary Cross-Entropy formula from Lesson 6.

Therefore, for standard unregularized binary Logistic Regression:

**Maximizing log-likelihood is equivalent to minimizing the sum of Log Loss; minimizing average Log Loss gives the same optimum because it only scales the objective by the positive constant 1/n.**

If regularization is added, the objective includes an additional penalty term. We will cover that later.

## 6. Numerical example: compare two probability models

Suppose the true labels are:

y = [1, 0, 1]

Consider two models that produce different probabilities for class 1.

**Model A**

p = [0.8, 0.3, 0.6]

The probabilities assigned to the observed labels are:

- Example 1: y = 1, so probability of its true label = 0.8.
- Example 2: y = 0, so probability of its true label = 1 − 0.3 = 0.7.
- Example 3: y = 1, so probability of its true label = 0.6.

Likelihood:

L_A = 0.8 × 0.7 × 0.6 = 0.336

Log-likelihood:

ℓ_A = ln(0.336) ≈ −1.0906

Average Log Loss:

J_A = −ℓ_A / 3 ≈ 0.3635

**Model B**

p = [0.9, 0.1, 0.8]

The probabilities assigned to the observed labels are 0.9, 0.9, and 0.8.

Likelihood:

L_B = 0.9 × 0.9 × 0.8 = 0.648

Log-likelihood:

ℓ_B = ln(0.648) ≈ −0.4339

Average Log Loss:

J_B = −ℓ_B / 3 ≈ 0.1446

Model B has a higher likelihood and a lower average Log Loss on these examples, so it fits these observed labels better under this criterion. This small illustration alone does not prove that Model B generalizes better to unseen data.

## 7. Python practice: calculate likelihood and log-likelihood

This code compares the two sets of probabilities above. It **does not train** Logistic Regression or learn coefficients; it only calculates the likelihood-based quantities for given predictions.

```python
import math


def evaluate_likelihood(y_true, probabilities):
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Inputs must have equal, non-zero lengths.")

    log_likelihood = 0.0

    for y, p in zip(y_true, probabilities):
        if y not in (0, 1):
            raise ValueError("Every true label must be 0 or 1.")

        if not 0 < p < 1:
            raise ValueError("Probabilities must be strictly between 0 and 1.")

        # Log probability of the observed label for this example.
        log_likelihood += y * math.log(p) + (1 - y) * math.log(1 - p)

    # The sum of log probabilities equals log of the likelihood.
    likelihood = math.exp(log_likelihood)
    average_log_loss = -log_likelihood / len(y_true)

    return likelihood, log_likelihood, average_log_loss


y_true = [1, 0, 1]
model_a = [0.8, 0.3, 0.6]
model_b = [0.9, 0.1, 0.8]

for name, probabilities in [("Model A", model_a), ("Model B", model_b)]:
    likelihood, log_likelihood, avg_loss = evaluate_likelihood(
        y_true, probabilities
    )

    print(name)
    print(f"  Likelihood:       {likelihood:.4f}")
    print(f"  Log-likelihood:   {log_likelihood:.4f}")
    print(f"  Average Log Loss: {avg_loss:.4f}")
```

Expected output (rounded):

```text
Model A
  Likelihood:       0.3360
  Log-likelihood:   -1.0906
  Average Log Loss: 0.3635
Model B
  Likelihood:       0.6480
  Log-likelihood:   -0.4339
  Average Log Loss: 0.1446
```

For larger datasets, calculate log-likelihood directly rather than multiplying probabilities first. The direct product can underflow to zero even when the log-likelihood calculation remains numerically manageable. The code forms a product only at the end for this small teaching example.

## 8. How does MLE learn the actual coefficients?

In a fitted Logistic Regression model, each probability is determined by the coefficients:

pᵢ = 1 / (1 + e^(−(b₀ + b₁xᵢ₁ + … + bₙxᵢₙ)))

MLE changes those coefficients to improve the overall log-likelihood of the observed training labels. For standard Logistic Regression, there is generally no simple closed-form solution for all coefficients, so an iterative numerical optimizer is used (for example, a gradient-based method or a quasi-Newton method).

The next lesson will derive the gradient of the Log Loss objective and show how gradient descent updates model coefficients.

## Key takeaways

- MLE chooses parameters that maximize the likelihood of observed training labels.
- For binary targets, the Bernoulli probability is pʸ × (1 − p)^(1 − y).
- Independent observation probabilities multiply to form the likelihood.
- Log-likelihood converts that product into a sum and is easier to optimize numerically.
- Maximizing log-likelihood is equivalent to minimizing negative log-likelihood.
- Average negative log-likelihood is Binary Cross-Entropy / average Log Loss.
- The code in this lesson evaluates given probability predictions; it does not fit model coefficients yet.

**Next:** Lesson 8 — Gradient Descent and coefficient learning.


---

# Lesson 8: Gradient Descent and Coefficient Learning

## 1. What is Gradient Descent?

MLE tells us what objective to optimize: maximize log-likelihood, or equivalently minimize negative log-likelihood / Log Loss.

Gradient Descent is one iterative optimization algorithm for finding coefficient values that reduce that objective.

The main idea:

1. Start with initial coefficients.
2. Calculate predicted probabilities.
3. Calculate the loss and its gradient.
4. Update the coefficients in the direction that reduces the loss.
5. Repeat until a stopping condition is reached.

For Logistic Regression, the model for observation i is:

zᵢ = b₀ + b₁xᵢ₁ + b₂xᵢ₂ + … + bₘxᵢₘ

pᵢ = 1 / (1 + e^(−zᵢ))

The average Binary Cross-Entropy objective is:

J(b) = −(1/n) Σ(i = 1…n) [yᵢ × ln(pᵢ) + (1 − yᵢ) × ln(1 − pᵢ)]

Gradient Descent tries to minimize J by updating b.

## 2. The coefficient update rule

For a coefficient bⱼ, Gradient Descent uses:

bⱼ(new) = bⱼ(old) − α × (∂J / ∂bⱼ)

Where:

- bⱼ is a model coefficient.
- α (alpha) is the learning rate, which controls the update size.
- ∂J / ∂bⱼ is the gradient: the rate at which the loss changes with that coefficient.

The minus sign matters. We move against the gradient because the gradient points in the direction of the steepest local increase in the objective.

- If the gradient is positive, the update decreases that coefficient.
- If the gradient is negative, the update increases that coefficient.
- If the gradient is near zero, the local first-order change is small.

A learning rate that is too small can make progress slow. A rate that is too large can overshoot, oscillate, or diverge.

## 3. Deriving the Logistic Regression gradient

Start with the loss for one observation:

L = −[y × ln(p) + (1 − y) × ln(1 − p)]

The probability p depends on the linear score z through the sigmoid function:

p = 1 / (1 + e^(−z))

Use the chain rule to find how loss changes with z:

∂L / ∂z = (∂L / ∂p) × (∂p / ∂z)

### Step A: Differentiate the loss with respect to p

∂L / ∂p = −y/p + (1 − y)/(1 − p)

### Step B: Differentiate the sigmoid

The sigmoid derivative is:

∂p / ∂z = p × (1 − p)

### Step C: Apply the chain rule

Multiply the two derivatives:

∂L / ∂z = [−y/p + (1 − y)/(1 − p)] × p × (1 − p)

Simplifying gives a very useful result:

**∂L / ∂z = p − y**

This compact result is the key to the standard Logistic Regression gradient.

### Step D: Differentiate with respect to a coefficient

The linear score is:

zᵢ = b₀ + b₁xᵢ₁ + … + bⱼxᵢⱼ + … + bₘxᵢₘ

The derivative of zᵢ with respect to bⱼ is xᵢⱼ. Applying the chain rule:

∂Lᵢ / ∂bⱼ = (pᵢ − yᵢ) × xᵢⱼ

Average this gradient across all n training examples:

∂J / ∂bⱼ = (1/n) Σ(i = 1…n) [(pᵢ − yᵢ) × xᵢⱼ]

For the intercept, each observation has an implicit feature value of 1, so:

∂J / ∂b₀ = (1/n) Σ(i = 1…n) (pᵢ − yᵢ)

For one input feature x, the formulas become:

∂J / ∂b₀ = mean(p − y)

∂J / ∂b₁ = mean((p − y) × x)

These are the gradients used in the Python implementation below.

## 4. One numerical gradient update by hand

Consider one training observation:

- Input x = 2
- Actual label y = 1
- Initial intercept b₀ = 0
- Initial coefficient b₁ = 0
- Learning rate α = 0.1

### Step 1: Calculate the score and probability

z = b₀ + b₁x = 0 + 0 × 2 = 0

p = sigmoid(0) = 0.5

### Step 2: Calculate the initial loss

Because y = 1:

L = −ln(p) = −ln(0.5) ≈ 0.6931

### Step 3: Calculate the gradients

For one example, the error term p − y is:

p − y = 0.5 − 1 = −0.5

Intercept gradient:

∂L / ∂b₀ = p − y = −0.5

Coefficient gradient:

∂L / ∂b₁ = (p − y) × x = −0.5 × 2 = −1.0

### Step 4: Update the parameters

Update rule: new coefficient = old coefficient − learning rate × gradient.

b₀(new) = 0 − 0.1 × (−0.5) = 0.05

b₁(new) = 0 − 0.1 × (−1.0) = 0.10

The gradients are negative, so both parameters increase.

### Step 5: Check the new prediction

New score:

z(new) = 0.05 + 0.10 × 2 = 0.25

New probability:

p(new) = sigmoid(0.25) ≈ 0.5622

Since the actual label is 1, the new loss is:

L(new) = −ln(0.5622) ≈ 0.5759

The loss decreased from about 0.6931 to 0.5759 after this update. This is one simple update on one example; training on a dataset averages gradients across all examples.

## 5. Batch Gradient Descent

In **batch Gradient Descent**, every update uses the full training dataset.

For all examples at once, the gradient for coefficient bⱼ is:

gradientⱼ = (1/n) Σ(i = 1…n) [(pᵢ − yᵢ) × xᵢⱼ]

This differs from Stochastic Gradient Descent, which updates parameters using one example at a time, and mini-batch Gradient Descent, which uses a subset of examples. We will compare these optimization strategies later.

A convenient matrix form, with a column of ones added for the intercept, is:

gradient = Xᵀ × (p − y) / n

Here X includes the intercept column, p is the vector of predicted probabilities, y is the vector of labels, and Xᵀ is the transpose of X.

## 6. Python implementation from scratch

This example trains a one-feature Logistic Regression model using NumPy and batch Gradient Descent. It does not use scikit-learn.

The feature is standardized first to make optimization more stable. The learned coefficient therefore corresponds to standardized study hours, not the original hour scale.

```python
import numpy as np


def sigmoid(z):
    # Clipping protects exp() from extreme numerical inputs.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y_true, probabilities, epsilon=1e-15):
    p = np.clip(probabilities, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(p)
        + (1 - y_true) * np.log(1 - p)
    )


# Small illustrative binary classification dataset
X_raw = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# Standardize the feature
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_scaled = (X_raw - X_mean) / X_std

# Add a column of ones for the intercept
X = np.c_[np.ones((len(X_scaled), 1)), X_scaled]

# Start with intercept = 0 and coefficient = 0
theta = np.zeros(X.shape[1], dtype=float)

learning_rate = 0.1
epochs = 2000

initial_probabilities = sigmoid(X @ theta)
initial_loss = binary_log_loss(y, initial_probabilities)

for epoch in range(epochs):
    # 1. Forward pass: compute scores and probabilities
    probabilities = sigmoid(X @ theta)

    # 2. Compute the average gradient for all parameters
    gradient = (X.T @ (probabilities - y)) / len(y)

    # 3. Update parameters in the negative-gradient direction
    theta -= learning_rate * gradient

final_probabilities = sigmoid(X @ theta)
final_loss = binary_log_loss(y, final_probabilities)
predicted_classes = (final_probabilities >= 0.5).astype(int)

print("Initial Log Loss:", round(initial_loss, 4))
print("Final Log Loss:", round(final_loss, 4))
print("Learned intercept:", round(theta[0], 4))
print("Coefficient for standardized study hours:", round(theta[1], 4))
print("Predicted probabilities:", np.round(final_probabilities, 3))
print("Predicted classes:", predicted_classes)
```

The initial loss should be about 0.6931, and the final loss should be lower after these updates. Exact final coefficients and probabilities depend on the number of iterations and learning rate.

This small dataset is perfectly separable. In unregularized Logistic Regression, perfectly separable data can cause coefficient magnitudes to keep growing as the loss approaches zero rather than settling at a finite maximum-likelihood estimate. This example demonstrates the optimization mechanism, not a recommended production training setup.

## 7. What to inspect when training does not work

- **Loss is not decreasing:** the learning rate may be too high, the gradient implementation may be wrong, or the data may need preprocessing.
- **Loss barely changes:** the learning rate may be too small, features may have very different scales, or gradients may be small.
- **Probabilities are all near 0.5:** the model may not have learned enough yet, or its features may provide limited signal.
- **Coefficients grow extremely large:** check for perfect or near-perfect separation; regularization can help.
- **Training loss is low but test performance is poor:** check data leakage, overfitting, distribution shift, and the quality of the train/test split.

Gradient Descent is an optimizer, not a guarantee of generalization. Always evaluate the trained model on held-out data.

## Key takeaways

- MLE defines the objective; Gradient Descent is one method to optimize it.
- For binary Logistic Regression, the derivative of one-example Log Loss with respect to the score is p − y.
- The gradient for coefficient bⱼ is the average of (pᵢ − yᵢ) × xᵢⱼ.
- The intercept gradient is the average of pᵢ − yᵢ.
- Parameters update by subtracting the learning rate times the gradient.
- Feature scaling can help Gradient Descent converge more reliably.
- Learning rate, stopping criteria, regularization, and validation all matter in a robust training workflow.

**Next:** Lesson 9 — Training Logistic Regression with scikit-learn and comparing it with the from-scratch implementation.


---

# Lesson 9: scikit-learn Logistic Regression — Training and Evaluation

## Start here: Lesson 9 in simple language

Do not try to memorize the entire lesson at once. Learn the pipeline in this order:

1. **Prepare the data:** `X` contains the information given to the model (features); `y` contains the correct answer (target).
2. **Split the data:** training data is used to learn; test data is kept aside to check performance on unseen examples.
3. **Scale features if needed:** learn the scaling numbers from training data only, then apply the same transformation to the test data.
4. **Create and train the model:** `model.fit(X_train, y_train)` learns the coefficients from the training examples.
5. **Predict:** `model.predict(X_test)` returns class labels; `model.predict_proba(X_test)` returns probabilities for each class.
6. **Evaluate:** compare predictions with the actual test labels using appropriate metrics.

### Step 3: Feature scaling without data leakage

Features can use very different scales. For example, study hours might range from 1 to 8, while attendance might range from 60 to 95. Standardization helps many optimization algorithms work more reliably.

For each feature, StandardScaler uses the training-set mean and standard deviation:

scaled value = (value − training mean) / training standard deviation

After standardization, each training feature has a mean close to 0 and a standard deviation close to 1. This does not mean the feature has become normally distributed.

**Important:** split first, then fit the scaler on training data only. Use the already-fitted scaler to transform test data. Do not call `fit_transform()` on the test set, because that lets test-set statistics influence preprocessing.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 85],
    [6, 90],
    [7, 92],
    [8, 95]
], dtype=float)

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()

# Learn mean/std only from training rows
X_train_scaled = scaler.fit_transform(X_train)

# Reuse those training mean/std values for test rows
X_test_scaled = scaler.transform(X_test)

print("Training shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("Training feature means:", X_train_scaled.mean(axis=0).round(4))
print("Training feature stds:", X_train_scaled.std(axis=0).round(4))
```

Why this matters:
- `fit_transform(X_train)` learns the scaling statistics from training data and transforms it.
- `transform(X_test)` uses those saved statistics without learning anything from test data.
- Scaling can help Logistic Regression's optimization and makes coefficient magnitudes easier to compare across features.
- A `Pipeline(StandardScaler(), LogisticRegression())` automates this correctly when it is fitted on training data.

### Step 4: Create and train the model with a Pipeline

Now we have:
- `X_train`: the input features used for learning.
- `y_train`: the correct labels used for learning.
- `X_test` and `y_test`: held aside for later evaluation.

We will combine scaling and Logistic Regression into one `Pipeline`. This is safer than manually scaling the whole dataset because the Pipeline learns preprocessing only when fitted on the training split.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Two input features: study hours and attendance
X = np.array([
    [1, 60],
    [2, 65],
    [3, 70],
    [4, 75],
    [5, 85],
    [6, 90],
    [7, 92],
    [8, 95]
], dtype=float)

# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Keep the same split as Step 3
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Build a pipeline: first scale, then classify
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
)

# Learn from training data
model.fit(X_train, y_train)

# Inspect the fitted Logistic Regression step
classifier = model.named_steps["logisticregression"]

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Learned intercept:", np.round(classifier.intercept_, 3))
print("Learned coefficients:", np.round(classifier.coef_, 3))
```

Expected output (rounded):

```text
Training rows: 6
Test rows: 2
Learned intercept: [-0.031]
Learned coefficients: [[0.732 0.892]]
```

The exact coefficients depend on the training data, split, preprocessing, and model settings. With the code and split above, these are the expected rounded values.

#### What does each line mean?

- `make_pipeline(...)` connects preprocessing and model training into one object.
- `StandardScaler()` learns the training features' means and standard deviations and standardizes those features.
- `LogisticRegression(...)` creates the classifier; it has not learned coefficients until `fit()` runs.
- `model.fit(X_train, y_train)` fits the scaler on `X_train`, transforms those training features, then learns the Logistic Regression coefficients and intercept from the transformed features and `y_train`.
- `model.named_steps["logisticregression"]` lets us inspect the fitted classifier inside the Pipeline.
- `coef_` contains learned feature coefficients; `intercept_` contains the learned intercept.

#### What did `fit()` learn?

The fitted Logistic Regression model calculates a score from the **standardized** inputs:

z = b₀ + b₁ × standardized study hours + b₂ × standardized attendance

It then turns that score into a probability:

p = 1 / (1 + e^(−z))

During training, the optimizer adjusts the coefficients and intercept to reduce the objective (Log Loss with regularization, under these settings).

The two coefficients above correspond to the standardized features, in this order:
1. Study hours
2. Attendance

They are not coefficients for the original unscaled numbers. A positive coefficient means that increasing that standardized feature increases the modelled log-odds of class 1, holding the other feature fixed. It does not prove causation.

**For this step, focus only on `model.fit(X_train, y_train)`: it is the point where the Pipeline learns from the training examples. We will use `predict()` and `predict_proba()` in the next step.**

### Step 5: `predict()` versus `predict_proba()`

In Step 4, `model.fit(X_train, y_train)` trained the Pipeline. Continue in the same Python file, below the Step 4 code, so that the variables `model`, `X_test`, and `y_test` already exist.

```python
# Predict the final class: 0 or 1
y_pred = model.predict(X_test)

# Predict probabilities for both classes
probabilities = model.predict_proba(X_test)

# Check which column corresponds to each class
classifier = model.named_steps["logisticregression"]
print("Class order:", classifier.classes_)

print("\nActual test labels:", y_test)
print("Predicted labels:", y_pred)
print("\nProbabilities [class 0, class 1]:")
print(np.round(probabilities, 4))

# Column 1 is class 1 (Pass) because classes_ is [0, 1]
p_pass = probabilities[:, 1]
print("\nProbability of Pass:", np.round(p_pass, 4))
```

Expected output (rounded, with the Step 3 split using `random_state=42`):

```text
Class order: [0 1]

Actual test labels: [0 1]
Predicted labels: [0 1]

Probabilities [class 0, class 1]:
[[0.7361 0.2639]
 [0.1481 0.8519]]

Probability of Pass: [0.2639 0.8519]
```

#### What is the difference?

- **`predict(X_test)`** gives a class label for each test row. With the default binary decision rule, the model usually predicts class 1 when its class-1 probability is at least 0.5; otherwise it predicts class 0.
- **`predict_proba(X_test)`** gives the probability for every class. Each row corresponds to one observation, and each column corresponds to a class in the order shown by `classes_`.
- In this toy dataset, `classes_` is `[0, 1]`, so column 0 is Fail and column 1 is Pass.
- For the first test student, the model estimates 26.39% for Pass and 73.61% for Fail, so it predicts class 0.
- For the second test student, it estimates 85.19% for Pass and 14.81% for Fail, so it predicts class 1.

The probabilities are estimates, not guarantees. Also, a predicted class and a predicted probability answer different questions: `predict()` answers “which class?”, while `predict_proba()` answers “what probability did the model assign to each class?”

**Do not evaluate the model yet in this step.** First make sure the difference between labels and probabilities is clear; metrics come later.

### Step 6: Evaluate predictions with accuracy

So far we have two things:
- `y_test`: the correct answers for the held-out test rows.
- `y_pred`: the model's predicted class labels for those same rows.

Accuracy tells us what fraction of the predictions match the correct answers.

Accuracy = number of correct predictions / total number of predictions

For this toy example, both test predictions are correct, so accuracy is 2 / 2 = 1.0, or 100%. However, the test set contains only two rows, so this result is too small to tell us reliably how the model would perform on new students.

```python
from sklearn.metrics import accuracy_score

print("Actual labels:", y_test)
print("Predicted labels:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(f"Accuracy (%): {accuracy * 100:.1f}%")
```

Expected output for the same toy split:

```text
Actual labels: [0 1]
Predicted labels: [0 1]
Accuracy: 1.0
Accuracy (%): 100.0%
```

Accuracy is useful as a first check, but it does not tell us which kinds of mistakes the model makes, and it can be misleading if one class is much more common than the other. We will learn the confusion matrix and other metrics separately rather than adding everything at once.

**Important:** use the test set for evaluation, not for fitting the model or repeatedly tuning it.

### Step 7: Confusion matrix — TP, TN, FP, FN

Accuracy tells us how many predictions were correct overall. The **confusion matrix** shows which predictions were right and which types of mistakes the model made.

For this example, define:
- Class 0 = negative class (Fail)
- Class 1 = positive class (Pass)

A confusion matrix compares the actual labels with the predicted labels. In scikit-learn, when we specify `labels=[0, 1]`, the rows are actual classes and the columns are predicted classes:

```text
                         Predicted
                     0             1
Actual 0           TN              FP
Actual 1           FN              TP
```

The four terms are:

- **TN — True Negative:** actual label is 0, and the model predicts 0.
- **FP — False Positive:** actual label is 0, but the model predicts 1.
- **FN — False Negative:** actual label is 1, but the model predicts 0.
- **TP — True Positive:** actual label is 1, and the model predicts 1.

"True" means the prediction matches the actual class; "False" means it does not. "Positive" refers to class 1, and "negative" refers to class 0.

#### Small numerical example

Use these actual labels and predictions:

```python
y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred_example = [0, 0, 1, 0, 1, 1, 1]
```

Compare each pair:

| Actual | Predicted | Meaning |
|---:|---:|---|
| 0 | 0 | TN |
| 0 | 0 | TN |
| 0 | 1 | FP |
| 1 | 0 | FN |
| 1 | 1 | TP |
| 1 | 1 | TP |
| 1 | 1 | TP |

Now count the outcomes:
- TN = 2
- FP = 1
- FN = 1
- TP = 3

Therefore, the confusion matrix is:

```text
[[2, 1],
 [1, 3]]
```

The top-left value is TN, top-right is FP, bottom-left is FN, and bottom-right is TP. Always check the class order before interpreting the matrix.

#### Python code

```python
from sklearn.metrics import confusion_matrix

# Illustrative predictions chosen to show all four outcomes
y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred_example = [0, 0, 1, 0, 1, 1, 1]

cm = confusion_matrix(
    y_true,
    y_pred_example,
    labels=[0, 1]
)

print("Confusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)
```

Expected output:

```text
Confusion matrix:
[[2 1]
 [1 3]]
True Negatives (TN): 2
False Positives (FP): 1
False Negatives (FN): 1
True Positives (TP): 3
```

This is a separate illustrative example, not the two-row test set from Steps 2–6. Our earlier toy test set had only two observations and both happened to be correct, so its confusion matrix would not demonstrate all four types.

**For this step, learn only how to read the four cells.** We will use these counts to learn additional evaluation metrics in later steps.

### Step 8: Precision

Precision answers this question:

**Out of all examples the model predicted as class 1, how many were actually class 1?**

In our student example:
- Class 0 = Fail
- Class 1 = Pass

Using the confusion matrix from Step 7:

```text
                 Predicted
                 0    1
Actual  0        2    1
        1        1    3
```

We counted:
- TP = 3: the model predicted Pass and the student actually passed.
- FP = 1: the model predicted Pass, but the student actually failed.

The model predicted Pass for 4 students in total: 3 true positives and 1 false positive. Only 3 of those 4 predictions were correct.

Precision = TP / (TP + FP)

Precision = 3 / (3 + 1) = 3 / 4 = 0.75 = 75%

So the model's **precision for class 1 is 75%**. In plain language: among the students the model predicted would Pass, 75% actually passed.

Precision looks specifically at predicted positive cases. A false positive lowers precision because it adds an incorrect positive prediction.

#### Python code

```python
from sklearn.metrics import confusion_matrix, precision_score

y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred_example = [0, 0, 1, 0, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred_example, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Calculate precision manually from the confusion-matrix counts
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0.0

# scikit-learn calculates precision for the positive class (label 1)
precision_sklearn = precision_score(
    y_true,
    y_pred_example,
    pos_label=1,
    zero_division=0
)

print("TP:", tp)
print("FP:", fp)
print("Precision (manual):", precision_manual)
print("Precision (scikit-learn):", precision_sklearn)
print(f"Precision (%): {precision_sklearn * 100:.1f}%")
```

Expected output:

```text
TP: 3
FP: 1
Precision (manual): 0.75
Precision (scikit-learn): 0.75
Precision (%): 75.0%
```

The manual calculation and scikit-learn result should agree. If a model does not predict any positive examples, the denominator TP + FP is zero; in that case precision is mathematically undefined. Here, `zero_division=0` tells scikit-learn to return 0 for that edge case.

**For this step, focus only on Precision = TP / (TP + FP).** We will learn other metrics one at a time in later steps.

### Step 9: Recall

Recall answers this question:

**Out of all examples that actually belong to class 1, how many did the model correctly predict as class 1?**

In our student example:
- Class 0 = Fail
- Class 1 = Pass

Use the confusion matrix from Step 7:

```text
                 Predicted
                 0    1
Actual  0        2    1
        1        1    3
```

We counted:
- **TP = 3:** three students actually passed and the model predicted Pass.
- **FN = 1:** one student actually passed, but the model predicted Fail.

There were four students who actually passed: TP + FN = 3 + 1 = 4. The model correctly identified three of them.

Recall = TP / (TP + FN)

Recall = 3 / (3 + 1) = 3 / 4 = 0.75 = 75%

So the model's **Recall for class 1 is 75%**. In plain language: it correctly identified 75% of all students who actually passed.

A false negative lowers Recall because it is an actual positive case the model missed.

#### Python code

```python
from sklearn.metrics import confusion_matrix, recall_score

y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred_example = [0, 0, 1, 0, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred_example, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Calculate Recall directly from confusion-matrix counts
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# scikit-learn calculates Recall for the positive class (label 1)
recall_sklearn = recall_score(
    y_true,
    y_pred_example,
    pos_label=1,
    zero_division=0
)

print("TP:", tp)
print("FN:", fn)
print("Recall (manual):", recall_manual)
print("Recall (scikit-learn):", recall_sklearn)
print(f"Recall (%): {recall_sklearn * 100:.1f}%")
```

Expected output:

```text
TP: 3
FN: 1
Recall (manual): 0.75
Recall (scikit-learn): 0.75
Recall (%): 75.0%
```

The manual calculation and scikit-learn result should agree. If there are no actual positive examples, TP + FN is zero and Recall is undefined; here, `zero_division=0` tells scikit-learn to return 0 for that edge case.

**For this step, focus only on Recall = TP / (TP + FN).** We will keep learning one evaluation metric at a time.

### Step 10: F1-score

F1-score combines **Precision** and **Recall** into one metric. It is useful when we want a balance between avoiding false positives and finding actual positive cases.

The formula is:

```text
F1-score = 2 × Precision × Recall / (Precision + Recall)
```

Using the counts from our student example:
- **TP = 3:** correctly predicted Pass.
- **FP = 1:** predicted Pass, but the student actually failed.
- **FN = 1:** predicted Fail, but the student actually passed.

We can calculate F1 directly from these counts:

```text
F1-score = 2 × TP / (2 × TP + FP + FN)
         = 2 × 3 / (2 × 3 + 1 + 1)
         = 6 / 8
         = 0.75 = 75%
```

So the F1-score for class 1 is **75%**. A higher F1-score generally indicates a better balance between Precision and Recall. If one of those two metrics is very low, F1-score will also be low. F1-score does not use True Negatives (TN) in its formula.

#### Python code

```python
from sklearn.metrics import confusion_matrix, f1_score

y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred_example = [0, 0, 1, 0, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred_example, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Calculate F1-score directly from the confusion-matrix counts
denominator = 2 * tp + fp + fn
f1_manual = (2 * tp / denominator) if denominator > 0 else 0.0

# scikit-learn calculates F1-score for the positive class (label 1)
f1_sklearn = f1_score(
    y_true,
    y_pred_example,
    pos_label=1,
    zero_division=0
)

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("F1-score (manual):", f1_manual)
print("F1-score (scikit-learn):", f1_sklearn)
print(f"F1-score (%): {f1_sklearn * 100:.1f}%")
```

Expected output:

```text
TP: 3
FP: 1
FN: 1
F1-score (manual): 0.75
F1-score (scikit-learn): 0.75
F1-score (%): 75.0%
```

The manual result and scikit-learn result should agree. If there are no predicted positives and no actual positives, the formula's denominator is zero; `zero_division=0` tells scikit-learn to return 0 for that edge case.

**For this step, focus only on F1-score.** We will continue one evaluation metric at a time.

**Checkpoint:** If TP = 8, FP = 2, and FN = 2, what is the F1-score? Try calculating it before checking the next lesson.

---

### Step 11: Specificity

Specificity answers this question:

**Out of all examples that actually belong to class 0, how many did the model correctly predict as class 0?**

In our student example:
- Class 0 = Fail
- Class 1 = Pass

Use the confusion matrix from Step 7:

```text
                 Predicted
                 0    1
Actual  0        2    1
        1        1    3
```

We counted:
- **TN = 2:** two students actually failed, and the model correctly predicted Fail.
- **FP = 1:** one student actually failed, but the model incorrectly predicted Pass.

There were three students who actually failed: TN + FP = 2 + 1 = 3. The model correctly identified two of them.

Specificity = TN / (TN + FP)

Specificity = 2 / (2 + 1) = 2 / 3 ≈ 0.667 = 66.7%

So the model's **Specificity for class 0 is approximately 66.7%**. A false positive lowers Specificity because it is an actual negative case incorrectly predicted as positive.

#### Python code

```python
from sklearn.metrics import confusion_matrix, recall_score

y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred_example = [0, 0, 1, 0, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred_example, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Calculate Specificity manually
specificity_manual = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Recall for class 0 is equivalent to Specificity for class 1
specificity_sklearn = recall_score(
    y_true,
    y_pred_example,
    pos_label=0,
    zero_division=0
)

print("TN:", tn)
print("FP:", fp)
print("Specificity (manual):", specificity_manual)
print("Specificity (scikit-learn):", specificity_sklearn)
print(f"Specificity (%): {specificity_sklearn * 100:.1f}%")
```

Expected output:

```text
TN: 2
FP: 1
Specificity (manual): 0.6666666666666666
Specificity (scikit-learn): 0.6666666666666666
Specificity (%): 66.7%
```

Specificity is undefined if there are no actual negative examples (TN + FP = 0). Here, `zero_division=0` tells scikit-learn to return 0 for that edge case.

**For this step, focus only on Specificity = TN / (TN + FP).** We will continue one evaluation metric at a time.

**Checkpoint:** If TN = 9 and FP = 3, what is Specificity? Calculate TN / (TN + FP) before continuing.

---

### A tiny example before the real dataset

Imagine this toy dataset:

| Study hours | Pass result |
|---:|---:|
| 1 | 0 |
| 2 | 0 |
| 5 | 1 |
| 6 | 1 |

Here, `X` is the Study hours column and `y` is the Pass result column. The model learns from examples where the answer is already known, then we check whether it predicts unseen examples well.

This toy dataset is only for understanding the workflow. Later code in this lesson uses a larger built-in dataset so evaluation is more meaningful.


## 1. Why use scikit-learn?

In Lesson 8, we implemented Logistic Regression training with NumPy and Gradient Descent. In practice, scikit-learn provides a tested implementation with efficient solvers, regularization, prediction methods, and a consistent API.

In this lesson we will:
- Split data into training and test sets.
- Scale features without data leakage.
- Train Logistic Regression using `fit()`.
- Generate class predictions with `predict()`.
- Generate probabilities with `predict_proba()`.
- Inspect learned coefficients and the intercept.
- Evaluate accuracy, confusion matrix, precision, recall, F1-score, and Log Loss.
- Compare scikit-learn with our from-scratch implementation.

## 2. Install the libraries

Run this in the project's terminal:

```bash
python -m pip install numpy scikit-learn
```

## 3. Dataset and target definition

We will use scikit-learn's built-in Breast Cancer Wisconsin dataset as an educational binary-classification dataset. It has numerical features computed from digitized cell-nucleus images.

For this lesson we explicitly define:
- Class 0 = Benign
- Class 1 = Malignant

The dataset's original target labels use the opposite mapping, so we convert the target using `(data.target == 0).astype(int)`.

This is a learning example, **not a medical diagnostic tool**.

## 4. Train-test split and data leakage

We need an evaluation set that is not used to fit the model. We split the data before training:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data

# Explicitly make Malignant the positive class (label 1).
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- `test_size=0.2` reserves 20% of the observations for testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` keeps approximately the same class proportions in the training and test sets.

### What is data leakage?

Data leakage occurs when information that should be unavailable during training influences the learned model or preprocessing.

For example, fitting a scaler on all rows before splitting allows test-set feature statistics to influence preprocessing. Instead, fit preprocessing on training data only, then apply the learned transform to test data.

A scikit-learn `Pipeline` makes this safer: when we call `fit(X_train, y_train)`, the scaler is fitted only on the training split, followed by model training. At prediction time, the test data is transformed using that already-fitted scaler.

## 5. Train the scikit-learn model

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs"
    )
)

model.fit(X_train, y_train)
```

### What does each part do?

- `StandardScaler()` learns the mean and standard deviation of each feature from the training data, then standardizes the feature values.
- `LogisticRegression(...)` creates the classifier.
- `C=1.0` controls inverse regularization strength for this L2-regularized model: smaller C means stronger regularization; larger C means weaker regularization.
- `max_iter=2000` permits enough solver iterations for convergence on this dataset.
- `solver="lbfgs"` selects the optimization algorithm.
- `fit()` learns the parameters from the training data.

Feature scaling is especially useful when columns have very different numerical scales. The fitted scaler must be reused for future or test examples rather than refitted on them.

## 6. predict() vs predict_proba()

```python
# Predict hard class labels (0 or 1)
y_pred = model.predict(X_test)

# Predict probability for each class
probabilities = model.predict_proba(X_test)

# Pipeline's final estimator has classes_ = [0, 1].
# Column 1 is the probability of Malignant (our positive class).
p_malignant = probabilities[:, 1]

print("First 10 class predictions:", y_pred[:10])
print("First 10 malignant probabilities:", p_malignant[:10])
```

- `predict()` returns a class label, using the estimator's decision rule.
- `predict_proba()` returns one probability per class. The column order corresponds to `model.classes_` on the final estimator.
- Here, column 1 means Malignant because we mapped Malignant to target label 1.

The hard label is not the same as the probability. A default threshold of 0.5 is commonly used for binary prediction, but threshold selection can be changed for application-specific costs (Lesson 5).

## 7. Inspect the learned coefficients

```python
import numpy as np

classifier = model.named_steps["logisticregression"]
coefficients = classifier.coef_[0]
intercept = classifier.intercept_[0]

print("Classes:", classifier.classes_)
print("Intercept:", round(intercept, 4))
print("First five coefficients:")

for feature_name, coefficient in zip(data.feature_names[:5], coefficients[:5]):
    print(f"{feature_name}: {coefficient:.4f}")

# Odds ratio for a one-standard-deviation increase in each feature
odds_ratios = np.exp(coefficients)
print("First five odds ratios:", np.round(odds_ratios[:5], 3))
```

For each observation, the model's linear score is:

z = b₀ + b₁x₁ + b₂x₂ + … + bₘxₘ

p = 1 / (1 + e^(−z))

Because the inputs were standardized, each coefficient represents the change in log-odds associated with a one-standard-deviation increase in that feature, holding the other included features fixed.

- Positive coefficient: increases the modelled log-odds of Malignant as that feature increases.
- Negative coefficient: decreases the modelled log-odds, holding other features fixed.
- `exp(coefficient)` is the associated odds ratio for a one-standard-deviation increase.

These are conditional model associations, not proof of causality. Correlated features can make individual coefficients unstable or difficult to interpret.

## 8. Evaluate the model

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Log Loss:", round(log_loss(y_test, p_malignant), 4))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("Precision (Malignant):", round(precision_score(y_test, y_pred), 4))
print("Recall (Malignant):", round(recall_score(y_test, y_pred), 4))
print("F1-score (Malignant):", round(f1_score(y_test, y_pred), 4))

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign (0)", "Malignant (1)"],
    digits=4
))
```

Metric definitions:
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN).
- **Precision** = TP / (TP + FP): among predicted malignant cases, the fraction that are actually malignant.
- **Recall** = TP / (TP + FN): among actual malignant cases, the fraction detected.
- **F1-score** = 2 × Precision × Recall / (Precision + Recall).
- **Log Loss** measures the quality of predicted probabilities; lower is better when comparing predictions on the same evaluation labels.

For this label mapping, the confusion matrix rows are true labels and columns are predicted labels, with order [Benign, Malignant]:

[[TN, FP], [FN, TP]]

Accuracy alone is insufficient for some tasks. In particular, missing a positive case and producing a false alarm can have very different costs. Use metrics that reflect the task and select thresholds using validation data, not the final test set.

## 9. Compare with our from-scratch implementation

To make a meaningful comparison, use the same train/test split, the same training-only scaling rule, the same 0.5 classification threshold, and approximately the same objective.

The scikit-learn model above uses L2 regularization by default. To approximate its objective with average Binary Cross-Entropy in a simple from-scratch implementation, we add this penalty to the average loss:

J_regularized = average Log Loss + (1 / (2 × C × n)) × sum of squared non-intercept coefficients

The corresponding gradient adds:

gradient for each non-intercept coefficient += coefficient / (C × n)

The intercept is not penalized in the implementation below. Solver details and implementation conventions can vary, so tiny numerical differences are expected.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y_true, probabilities, epsilon=1e-15):
    p = np.clip(probabilities, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(p)
        + (1 - y_true) * np.log(1 - p)
    )


# Load the data and explicitly make Malignant the positive class.
data = load_breast_cancer()
X = data.data
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- A. scikit-learn model -----
C = 1.0

sk_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
)
sk_model.fit(X_train, y_train)

sk_prob = sk_model.predict_proba(X_test)[:, 1]
sk_pred = sk_model.predict(X_test)

# ----- B. from-scratch batch Gradient Descent -----
# Fit the scaler on training data only; transform test data with it.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add a column of ones for the intercept.
X_train_aug = np.c_[np.ones(len(X_train_scaled)), X_train_scaled]
X_test_aug = np.c_[np.ones(len(X_test_scaled)), X_test_scaled]

theta = np.zeros(X_train_aug.shape[1])
learning_rate = 0.1
epochs = 30000
n = len(y_train)

# Approximate matching regularization for average loss.
l2_strength = 1.0 / (C * n)

initial_train_loss = binary_log_loss(
    y_train, sigmoid(X_train_aug @ theta)
)

for epoch in range(epochs):
    p = sigmoid(X_train_aug @ theta)

    # Gradient of average Log Loss
    gradient = (X_train_aug.T @ (p - y_train)) / n

    # Add L2 gradient to coefficients, but not to the intercept.
    gradient[1:] += l2_strength * theta[1:]

    theta -= learning_rate * gradient

scratch_prob = sigmoid(X_test_aug @ theta)
scratch_pred = (scratch_prob >= 0.5).astype(int)

print("Initial scratch train Log Loss:", round(initial_train_loss, 4))
print("\n--- scikit-learn ---")
print("Accuracy:", round(accuracy_score(y_test, sk_pred), 4))
print("Log Loss:", round(log_loss(y_test, sk_prob), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, sk_pred, labels=[0, 1]))

print("\n--- From scratch ---")
print("Accuracy:", round(accuracy_score(y_test, scratch_pred), 4))
print("Log Loss:", round(binary_log_loss(y_test, scratch_prob), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, scratch_pred, labels=[0, 1]))

print("\nMaximum absolute coefficient difference:")
print(round(np.max(np.abs(
    sk_model.named_steps["logisticregression"].coef_[0] - theta[1:]
)), 4))
```

Expected results (small variations can occur across scikit-learn versions and solver implementations):

| Metric | scikit-learn | From scratch |
|---|---:|---:|
| Accuracy | 0.9649 | 0.9649 |
| Log Loss | 0.0773 | 0.0772 |
| Confusion matrix | [[71, 1], [3, 39]] | [[71, 1], [3, 39]] |

In this split, both models make the same class predictions. The tiny difference in Log Loss is expected because one uses scikit-learn's numerical solver and the other uses a fixed number of Gradient Descent updates.

Do not expect every dataset or run to produce identical coefficients or metrics. The comparison depends on preprocessing, regularization, the optimization method, the stopping criterion, and the data split.

## 10. Common mistakes to avoid

- **Data leakage:** never fit a scaler on the entire dataset before splitting. Use a Pipeline or fit preprocessing on training data only.
- **Wrong probability column:** confirm class order with `classes_` before choosing a `predict_proba()` column.
- **Comparing accuracy alone:** inspect Log Loss and class-specific precision/recall where relevant.
- **Ignoring convergence:** check warnings and increase `max_iter` or revisit scaling if the solver has not converged.
- **Reading coefficient signs as causality:** coefficients show modelled conditional associations, not causal effects.
- **Tuning on test data:** use validation data for hyperparameters or threshold selection; keep the test set for final evaluation.

## Key takeaways

- `fit()` learns coefficients from the training data.
- `predict()` returns class labels, while `predict_proba()` returns class probabilities.
- A Pipeline helps prevent preprocessing leakage by fitting transforms only on training data.
- Standardization can improve optimization and makes coefficient scales easier to compare.
- Coefficients in this lesson correspond to standardized features.
- Accuracy, confusion matrix, precision, recall, F1-score, and Log Loss provide complementary information.
- A from-scratch Gradient Descent implementation helps explain the mechanics; scikit-learn is more convenient and robust for normal workflows.

**Next:** Lesson 10 — Classification evaluation metrics in depth, including confusion matrix, precision, recall, F1, specificity, ROC curves, and ROC-AUC.

## Progress tracker

- [x] Lesson 1 — Classification fundamentals
- [x] Lesson 2 — Linear vs Logistic Regression
- [x] Lesson 3 — Sigmoid function
- [x] Lesson 4 — Probability, odds, and log-odds
- [x] Lesson 5 — Decision boundary and classification thresholds
- [x] Lesson 6 — Log Loss / cost function
- [x] Lesson 7 — Maximum Likelihood Estimation
- [x] Lesson 8 — Gradient Descent and coefficient learning
- [x] Lesson 9 — scikit-learn implementation
- [ ] Lesson 10 — Evaluation metrics
- [ ] Lesson 11 — Multiclass Logistic Regression
- [ ] Lesson 12 — L1/L2 regularization
- [ ] Lesson 13 — Imbalance, threshold tuning, calibration, and practical considerations
