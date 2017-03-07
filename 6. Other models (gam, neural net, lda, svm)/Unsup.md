# Other Supervised Learning Methods




# Objective

This tutorial helps you to review various supervised learning techniques, introduce GAM, Neural Networks models, etc., and prepare you to finish Case Study 1.

# Credit Score Data
## Load Data



We remove X9 and id from the data since we will not be using them for prediction.


Now split the data 90/10 as training/testing datasets:


The training dataset has 61 variables, 4500 obs. 

You are already familiar with the credit scoring set. Let's define a cost function for benchmarking testing set performance. Note this is slightly different from the one we used for searching for optimal cut-off probability in logistic regression. Here the 2nd argument is the predict class instead of the predict probability (since many methods are not based on predict probability).




# Generalized Linear Models (Logistic Regression)

Let's build a logistic regression model based on all X variables. Note id is excluded from the model.



You can view the result of the estimation:


The usual stepwise variable selection still works for logistic regression. **caution: this will take a very long time**.



Or you can try model selection with BIC:



Are there better ways of doing variable selection for genearlized linear models? Yes! (And you should probably know about it.) Check the optional lab notes on _Lasso variable selection_ and Section 3.4 of the textbook "Elements of Statistical Learning".

If you want a sneak peek on how to use Lasso for this dataset here it is:




```
## 61 x 1 sparse Matrix of class "dgCMatrix"
##                       1
## (Intercept) -2.52287744
## X2           .         
## X3           .         
## X4           .         
## X5           .         
## X6           .         
## X7           .         
## X8          -0.27239379
## X10_2        .         
## X11_2       -0.36149160
## X12_2        .         
## X13_2        .         
## X14_2        .         
## X15_2        .         
## X15_3        .         
## X15_4        .         
## X15_5        .         
## X15_6        .         
## X16_2        .         
## X16_3        .         
## X16_4        .         
## X16_5        .         
## X16_6        .         
## X17_2        .         
## X17_3        .         
## X17_4        .         
## X17_5        .         
## X17_6       -0.01852074
## X18_2        .         
## X18_3        .         
## X18_4        .         
## X18_5        .         
## X18_6        .         
## X18_7        .         
## X19_2        .         
## X19_3        .         
## X19_4        .         
## X19_5        .         
## X19_6        .         
## X19_7        .         
## X19_8        .         
## X19_9        .         
## X19_10       .         
## X20_2        .         
## X20_3        .         
## X20_4        .         
## X21_2        .         
## X21_3        .         
## X22_2        .         
## X22_3        .         
## X22_4        .         
## X22_5        .         
## X22_6        .         
## X22_7        .         
## X22_8        .         
## X22_9        0.08940061
## X22_10       .         
## X22_11       .         
## X23_2        .         
## X23_3        .         
## X24_2        .
```
The _s_ parameter determines how many variables are included and you can use cross-validation to choose it.

## Prediction and Cross Validation Using Logistic Regression

## Performance on testing set
To do out-of-sample prediction you need to add the testing set as a second argument after the glm object. Remember to add type = "response", otherwise you will get the log odds and not the probability.


```
##         Predicted
## Observed   0   1
##        0 430  29
##        1  31  10
```

```
## [1] 0.12
```

```
## [1] 0.678
```

## ROC Curve
To get the ROC curve you need to install the verification library.

To plot the ROC curve, the first argument of roc.plot is the vector with actual values "A binary observation (coded {0, 1 } )". The second argument is the vector with predicted probability. 



To get the area under the ROC curve:
![](Unsup_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

```
##      Model      Area      p.value binorm.area
## 1 Model  1 0.7134279 2.939155e-06          NA
```


<a href="#top">Back to top</a>

# Generalized Additive Models (GAM)
There are two common implementations of GAMs in R.  The older version (originally made for S-PLUS) is available as the 'gam' package by Hastie and Tibshirani.  The newer version that we will use below is the 'mgcv' package from Simon Wood.  The basic modeling procedure for both packages is similar (the function is gam for both; be wary of having both libraries loaded at the same time), but the behind-the-scenes computational approaches differ, as do the arguments for optimization and the model output.  Expect the results to be slightly different when used with the same model structure on the same dataset.


```
## Loading required package: nlme
```

```
## This is mgcv 1.8-16. For overview type 'help("mgcv-package")'.
```

```
## 
## Family: binomial 
## Link function: logit 
## 
## Formula:
## Y ~ s(X2) + s(X3) + s(X4) + s(X5) + X6 + X7 + X8 + X10_2 + X11_2 + 
##     X12_2 + X13_2 + X14_2 + X15_2 + X15_3 + X15_4 + X15_5 + X15_6 + 
##     X16_2 + X16_3 + X16_4 + X16_5 + X16_6 + X17_2 + X17_3 + X17_4 + 
##     X17_5 + X17_6 + X18_2 + X18_3 + X18_4 + X18_5 + X18_6 + X18_7 + 
##     X19_2 + X19_3 + X19_4 + X19_5 + X19_6 + X19_7 + X19_8 + X19_9 + 
##     X19_10 + X20_2 + X20_3 + X20_4 + X21_2 + X21_3 + X22_2 + 
##     X22_3 + X22_4 + X22_5 + X22_6 + X22_7 + X22_8 + X22_9 + X22_10 + 
##     X22_11 + X23_2 + X23_3 + X24_2
## 
## Parametric coefficients:
##               Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -3.372e+00  7.285e-01  -4.628 3.68e-06 ***
## X6           1.794e-01  1.143e-01   1.569 0.116569    
## X7          -1.491e-01  2.038e-01  -0.731 0.464615    
## X8          -2.234e+00  3.355e-01  -6.659 2.76e-11 ***
## X10_2       -2.299e-01  1.737e-01  -1.323 0.185705    
## X11_2       -8.795e-01  1.554e-01  -5.660 1.51e-08 ***
## X12_2       -4.743e-01  1.977e-01  -2.399 0.016445 *  
## X13_2        2.952e-01  1.629e-01   1.812 0.069991 .  
## X14_2       -4.191e-01  2.753e-01  -1.523 0.127857    
## X15_2        5.005e-01  2.454e-01   2.040 0.041364 *  
## X15_3        2.997e-01  3.019e-01   0.993 0.320901    
## X15_4        8.970e-01  3.202e-01   2.802 0.005083 ** 
## X15_5        5.682e-01  4.159e-01   1.366 0.171805    
## X15_6        9.988e-01  2.786e-01   3.585 0.000337 ***
## X16_2        4.110e-01  2.888e-01   1.423 0.154714    
## X16_3       -1.891e-01  2.879e-01  -0.657 0.511272    
## X16_4        2.064e-01  3.473e-01   0.594 0.552365    
## X16_5       -2.778e-01  2.896e-01  -0.959 0.337407    
## X16_6        1.423e-02  2.993e-01   0.048 0.962076    
## X17_2       -2.920e-01  2.726e-01  -1.071 0.284009    
## X17_3       -1.058e+00  3.105e-01  -3.408 0.000655 ***
## X17_4       -1.825e-01  2.702e-01  -0.676 0.499255    
## X17_5        6.633e-01  4.568e-01   1.452 0.146481    
## X17_6       -1.027e+00  1.765e-01  -5.816 6.03e-09 ***
## X18_2        2.839e-01  3.373e-01   0.842 0.399964    
## X18_3        4.518e-01  2.786e-01   1.622 0.104836    
## X18_4        9.232e-01  2.657e-01   3.474 0.000512 ***
## X18_5        7.121e-01  2.438e-01   2.921 0.003487 ** 
## X18_6        6.439e-01  2.936e-01   2.193 0.028304 *  
## X18_7        7.266e-01  3.178e-01   2.287 0.022217 *  
## X19_2        4.173e-01  3.738e-01   1.116 0.264357    
## X19_3        6.218e-01  3.154e-01   1.972 0.048641 *  
## X19_4        2.216e-01  5.220e-01   0.424 0.671227    
## X19_5        3.507e-01  4.359e-01   0.804 0.421136    
## X19_6        2.868e-01  4.698e-01   0.610 0.541650    
## X19_7        8.277e-01  4.284e-01   1.932 0.053339 .  
## X19_8       -3.488e+01  4.305e+06   0.000 0.999994    
## X19_9        1.021e+00  5.539e-01   1.844 0.065157 .  
## X19_10       4.445e-01  3.514e-01   1.265 0.205948    
## X20_2       -1.236e-01  3.719e-01  -0.332 0.739549    
## X20_3       -8.404e-02  2.753e-01  -0.305 0.760145    
## X20_4        1.678e-01  1.834e-01   0.915 0.360196    
## X21_2        5.581e-01  4.577e-01   1.219 0.222744    
## X21_3        6.003e-01  2.266e-01   2.649 0.008075 ** 
## X22_2       -3.619e-01  4.075e-01  -0.888 0.374492    
## X22_3        9.980e-02  3.522e-01   0.283 0.776902    
## X22_4       -2.112e-01  4.703e-01  -0.449 0.653325    
## X22_5        4.803e-02  4.251e-01   0.113 0.910050    
## X22_6        1.122e-01  5.917e-01   0.190 0.849605    
## X22_7        2.276e-01  3.730e-01   0.610 0.541714    
## X22_8       -6.771e-02  3.721e-01  -0.182 0.855611    
## X22_9        6.354e-01  3.244e-01   1.959 0.050145 .  
## X22_10      -1.273e+00  1.080e+00  -1.179 0.238449    
## X22_11       3.917e-01  3.493e-01   1.121 0.262119    
## X23_2        6.326e-02  2.100e-01   0.301 0.763260    
## X23_3       -7.920e-02  2.261e-01  -0.350 0.726133    
## X24_2        4.887e-01  3.174e-01   1.540 0.123620    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##         edf Ref.df Chi.sq  p-value    
## s(X2) 1.001  1.002  1.962    0.162    
## s(X3) 1.000  1.000 25.706 3.98e-07 ***
## s(X4) 1.685  2.104  1.116    0.559    
## s(X5) 3.168  3.943  6.590    0.172    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.119   Deviance explained = 19.6%
## UBRE = -0.61772  Scale est. = 1         n = 4500
```

![](Unsup_files/figure-html/unnamed-chunk-16-1.png)<!-- -->![](Unsup_files/figure-html/unnamed-chunk-16-2.png)<!-- -->![](Unsup_files/figure-html/unnamed-chunk-16-3.png)<!-- -->![](Unsup_files/figure-html/unnamed-chunk-16-4.png)<!-- -->

Model AIC/BIC and mean residual deviance

```
## [1] 1720.28
```

```
## [1] 2129.702
```

```
## [1] 1592.572
```

## In-sample fit performance
In order to see the in-sample fit performance, you may look into the confusion matrix by using commands as following. 


```
##         Predicted
## Observed    0    1
##        0 3424  817
##        1   97  162
```

Likewise, misclassification rate is another thing you can check:


```
## [1] 0.2031111
```

Training model AIC and BIC:

```
## [1] 1720.28
```

```
## [1] 2129.702
```

## Search for optimal cut-off probability

The following code does a grid search from pcut = 0.01 to pcut = 0.99 with the objective of minimizing overall cost in the training set. I am using an asymmetric cost function by assuming that giving out a bad loan cost 10 time as much as rejecting application from someone who can pay.

![](Unsup_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

```
##       
## 0.384
```

```
## searchgrid 
##       0.09
```

## Out-of-sample fit performance

```
##         Predicted
## Observed   0   1
##        0 365  94
##        1  22  19
```
mis-classifciation rate is

```
## [1] 0.232
```
Cost associated with misclassification is

```
## [1] 0.628
```

<a href="#top">Back to top</a>


# Discriminant Analysis
Linear Discriminant Analysis (LDA) (in-sample and out-of-sample performance measure) is illustrated here. The following illustrate the usage of an arbitrary cut off probability.

## In-sample

```
##    Pred
## Obs    0    1
##   0 3925  316
##   1  149  110
```

```
## [1] 0.1033333
```

## Out-of-sample

```
##    Pred
## Obs   0   1
##   0 396  63
##   1  27  14
```

```
## [1] 0.18
```

```
## [1] 0.666
```
<a href="#top">Back to top</a>


# Neural Networks Models
Neural Networks method (in-sample and out-of-sample performance measure) is illustrated here. The package [**nnet**](http://cran.r-project.org/web/packages/nnet/nnet.pdf) is used for this purpose.

__Note__: 

- For classification problems with nnet you need to code the response to _factor_ first. In addition you want to add type = "class" for _predict()_  function. 

- For regression problems add lineout = TRUE when training model. In addition, the response needs to be standardized to $[0, 1]$ interval.



## Training



```
## # weights:  63
## initial  value 2014.389358 
## iter  10 value 867.533490
## iter  20 value 831.985293
## iter  30 value 817.400749
## iter  40 value 807.250357
## iter  50 value 789.512160
## iter  60 value 779.540214
## final  value 778.838662 
## converged
```

## Out-of-sample Testing

```
##         Predicted
## Observed   0   1
##        0 409  50
##        1  30  11
```

```
## [1] 0.16
```

```
## [1] 0.7
```


<a href="#top">Back to top</a>

# Support Vector Machine (SVM)

SVM is probably one of the best off-the-shelf classifiers for many of problems. It handles nonlinearity, is well regularized (avoids overfitting), have few parameters, and fast for large number of observations. It can be adapted to handle regression problems as well. You can read more about SVM in Chapter 12 of the textbook. 

The R package e1071 offers an interface to the most popular svm implementation libsvm. You should read more about the usage of the package in this short tutorial (http://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf).





```
##    Pred
## Obs   0   1
##   0 377  82
##   1  24  17
```

```
## [1] 0.212
```

```
## [1] 0.644
```

credit.svm = svm(Y ~ ., data = credit.train, cost = 1, gamma = 1/length(credit.train), probability= TRUE)
prob.svm = predict(credit.svm, credit.test)

<a href="#top">Back to top</a>

# Performance Comparisons
At last, after fitting several models, you may want to compare their in-sample and out-of-sample performances. The performance measures are illustrated in previous sections. In your report, you may want to put them in some table format. Note that not all measures are applicable. For example, I didn't find AIC/BIC or deviance for LDA models and Neural Network models. For tree models, *tree* package can give you mean residual deviance but not with *rpart* package. If you find either one of them, I would be interested to know.

## In-sample
You may compare the following
- AIC or BIC
- Mean Residual Deviance (for binary response) or Mean Square Error (for continuous response)
- Cost (asymmetric or symmetric)
- Misclassification Rate
- ROC curve or Area Under the Curve (AUC)

## Out-of-sample
- Cost
- Misclassification Rate
- ROC curve or Area Under the Curve (AUC)


## Symmetric Cost and Multiclass Problems
For classification tasks with symmetric costs many of functions can be simplified. You do not have to worry about the cut-off probability and can focus on the tuning parameters in each model (e.g. cost and gamma in SVM).

Different classifiers deal with multiclass classification differently. Logistic regression can be extended to multinomial logistic regression (using _multinom_ function). Many other binary classifiers can use an either "one-vs-all"(train N binary classifiers to distinguish each class from the rest) or "one-vs-one"(train C(N,2) binary classifiers for each possible pair of classes) approach to deal with multiple classes. 


```
##             Predicted
## Observed     setosa versicolor virginica
##   setosa          7          0         0
##   versicolor      0          9         0
##   virginica       0          2        12
```


<a href="#top">Back to top</a>

# Starter code for German credit scoring
Refer to http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) for variable description. Notice that "It is worse to class a customer as good when they are bad (weight = 5), than it is to class a customer as bad when they are good (weight = 1)." Define your cost function accordingly!





<a href="#top">Back to top</a>
