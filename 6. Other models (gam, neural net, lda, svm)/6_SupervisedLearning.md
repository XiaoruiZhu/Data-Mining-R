# Other Supervised Learning Methods




# Objective

This tutorial helps you to review various supervised learning techniques, introduce GAM, Neural Networks models, etc., and prepare you to finish Case Study 1.

# Credit Score Data
## Load Data


```r
credit.data <- read.csv("http://homepages.uc.edu/~maifg/7040/credit0.csv", header=T)
```

We remove X9 and id from the data since we will not be using them for prediction.

```r
credit.data$X9 = NULL
credit.data$id = NULL
credit.data$Y = as.factor(credit.data$Y)
```

Now split the data 90/10 as training/testing datasets:

```r
id_train <- sample(nrow(credit.data),nrow(credit.data)*0.90)
credit.train = credit.data[id_train,]
credit.test = credit.data[-id_train,]
```

The training dataset has 61 variables, 4500 obs. 

You are already familiar with the credit scoring set. Let's define a cost function for benchmarking testing set performance. Note this is slightly different from the one we used for searching for optimal cut-off probability in logistic regression. Here the 2nd argument is the predict class instead of the predict probability (since many methods are not based on predict probability).


```r
creditcost <- function(observed, predicted){
  weight1 = 10
  weight0 = 1
  c1 = (observed==1)&(predicted == 0) #logical vector - true if actual 1 but predict 0
  c0 = (observed==0)&(predicted == 1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}
```

[go to top](#header)

# Generalized Linear Models (Logistic Regression)

Let's build a logistic regression model based on all X variables. Note id is excluded from the model.


```r
credit.glm0<-glm(Y~., family=binomial,credit.train)
```

You can view the result of the estimation:

```r
summary(credit.glm0)
```

The usual stepwise variable selection still works for logistic regression. **caution: this will take a very long time**.


```r
credit.glm.step <- step(credit.glm0,direction=c("both")) 
```

Or you can try model selection with BIC:


```r
credit.glm.step <- step(credit.glm0, k=log(nrow(credit.train)),direction=c("both")) 
```

Are there better ways of doing variable selection for genearlized linear models? Yes! (And you should probably know about it.) Check the optional lab notes on _Lasso variable selection_ and Section 3.4 of the textbook "Elements of Statistical Learning".

If you want a sneak peek on how to use Lasso for this dataset here it is:


```r
install.packages('glmnet')
```


```r
library(glmnet)
lasso_fit = glmnet(x = as.matrix(credit.train[, 2:61]), y = credit.train[,1], family= "binomial", alpha = 1)
coef(lasso_fit, s = 0.02)
```

```
## 61 x 1 sparse Matrix of class "dgCMatrix"
##                       1
## (Intercept) -2.48924115
## X2           .         
## X3           .         
## X4           .         
## X5           .         
## X6           .         
## X7           .         
## X8          -0.29215255
## X10_2        .         
## X11_2       -0.32316248
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
## X17_6       -0.04919884
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
## X22_9        0.03440006
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


```r
prob.glm0.outsample <- predict(credit.glm0,credit.test,type="response")
predicted.glm0.outsample <-  prob.glm0.outsample> 0.2
predicted.glm0.outsample <- as.numeric(predicted.glm0.outsample)
table(credit.test$Y, predicted.glm0.outsample, dnn=c("Observed","Predicted"))
```

```
##         Predicted
## Observed   0   1
##        0 460   9
##        1  24   7
```

```r
mean(ifelse(credit.test$Y != predicted.glm0.outsample, 1, 0))
```

```
## [1] 0.066
```

```r
creditcost(credit.test$Y, predicted.glm0.outsample)
```

```
## [1] 0.498
```

## ROC Curve
To get the ROC curve you need to install the verification library.

```r
install.packages('verification')
```
To plot the ROC curve, the first argument of roc.plot is the vector with actual values "A binary observation (coded {0, 1 } )". The second argument is the vector with predicted probability. 

```r
library('verification')
```


```r
roc.plot(credit.test$Y == '1', prob.glm0.outsample)
```
To get the area under the ROC curve:

```r
roc.plot(credit.test$Y == '1', prob.glm0.outsample)$roc.vol
```

![](6_SupervisedLearning_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

```
##      Model     Area      p.value binorm.area
## 1 Model  1 0.807277 4.918713e-09          NA
```


[go to top](#header)

# Generalized Additive Models (GAM)
There are two common implementations of GAMs in R.  The older version (originally made for S-PLUS) is available as the 'gam' package by Hastie and Tibshirani.  The newer version that we will use below is the 'mgcv' package from Simon Wood.  The basic modeling procedure for both packages is similar (the function is gam for both; be wary of having both libraries loaded at the same time), but the behind-the-scenes computational approaches differ, as do the arguments for optimization and the model output.  Expect the results to be slightly different when used with the same model structure on the same dataset.


```r
library(mgcv)
```

```
## Loading required package: nlme
```

```
## This is mgcv 1.8-16. For overview type 'help("mgcv-package")'.
```

```r
## Create a formula for a model with a large number of variables:
gam_formula <- as.formula(paste("Y~s(X2)+s(X3)+s(X4)+s(X5)+", paste(colnames(credit.train)[6:61], collapse= "+")))

credit.gam <- gam(formula = gam_formula, family=binomial,data=credit.train);
summary(credit.gam)
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
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -3.343874   0.713091  -4.689 2.74e-06 ***
## X6           0.076871   0.114013   0.674 0.500166    
## X7          -0.242101   0.211441  -1.145 0.252208    
## X8          -2.126245   0.321766  -6.608 3.89e-11 ***
## X10_2       -0.178846   0.168370  -1.062 0.288134    
## X11_2       -0.836837   0.152441  -5.490 4.03e-08 ***
## X12_2       -0.420805   0.195687  -2.150 0.031524 *  
## X13_2        0.287694   0.156913   1.833 0.066733 .  
## X14_2       -0.292813   0.274239  -1.068 0.285643    
## X15_2        0.434498   0.246019   1.766 0.077377 .  
## X15_3        0.220761   0.294629   0.749 0.453686    
## X15_4        0.787571   0.315312   2.498 0.012498 *  
## X15_5        0.353352   0.408339   0.865 0.386853    
## X15_6        0.827227   0.272043   3.041 0.002360 ** 
## X16_2        0.343290   0.277101   1.239 0.215397    
## X16_3       -0.156113   0.276094  -0.565 0.571778    
## X16_4        0.159002   0.341082   0.466 0.641094    
## X16_5       -0.270920   0.277066  -0.978 0.328164    
## X16_6       -0.007888   0.289587  -0.027 0.978270    
## X17_2        0.005289   0.251315   0.021 0.983209    
## X17_3       -0.988311   0.310909  -3.179 0.001479 ** 
## X17_4       -0.364686   0.272868  -1.336 0.181389    
## X17_5        0.614748   0.439246   1.400 0.161647    
## X17_6       -1.031449   0.172867  -5.967 2.42e-09 ***
## X18_2        0.203435   0.332016   0.613 0.540057    
## X18_3        0.334384   0.275401   1.214 0.224681    
## X18_4        0.944636   0.257753   3.665 0.000247 ***
## X18_5        0.797661   0.236985   3.366 0.000763 ***
## X18_6        0.507104   0.292497   1.734 0.082971 .  
## X18_7        0.728602   0.303520   2.401 0.016372 *  
## X19_2        0.253785   0.350937   0.723 0.469579    
## X19_3        0.393225   0.293481   1.340 0.180289    
## X19_4       -0.188251   0.534277  -0.352 0.724579    
## X19_5        0.175952   0.420815   0.418 0.675857    
## X19_6        0.051708   0.456797   0.113 0.909874    
## X19_7        0.622502   0.417080   1.493 0.135562    
## X19_8       -1.020201   0.669141  -1.525 0.127348    
## X19_9        0.508186   0.551683   0.921 0.356969    
## X19_10       0.388175   0.327846   1.184 0.236407    
## X20_2        0.104861   0.360396   0.291 0.771081    
## X20_3        0.139868   0.268125   0.522 0.601911    
## X20_4        0.358828   0.185676   1.933 0.053291 .  
## X21_2        0.373506   0.426202   0.876 0.380835    
## X21_3        0.481471   0.226078   2.130 0.033199 *  
## X22_2       -0.333671   0.398214  -0.838 0.402077    
## X22_3        0.026092   0.353773   0.074 0.941206    
## X22_4        0.010214   0.445937   0.023 0.981727    
## X22_5        0.090832   0.416140   0.218 0.827217    
## X22_6        0.226167   0.562448   0.402 0.687601    
## X22_7        0.326190   0.363580   0.897 0.369634    
## X22_8        0.071986   0.362803   0.198 0.842721    
## X22_9        0.606091   0.319234   1.899 0.057619 .  
## X22_10      -1.333312   1.080467  -1.234 0.217198    
## X22_11       0.401400   0.344980   1.164 0.244609    
## X23_2        0.186789   0.206171   0.906 0.364941    
## X23_3       -0.126852   0.224060  -0.566 0.571291    
## X24_2        0.303748   0.311316   0.976 0.329219    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##         edf Ref.df Chi.sq  p-value    
## s(X2) 1.002  1.003  2.560    0.110    
## s(X3) 1.002  1.003 19.565 9.83e-06 ***
## s(X4) 1.115  1.221  0.141    0.725    
## s(X5) 2.734  3.417  3.371    0.419    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.103   Deviance explained = 17.7%
## UBRE = -0.59966  Scale est. = 1         n = 4500
```

```r
plot(credit.gam, shade=TRUE,,seWithMean=TRUE,scale=0)
```

![](6_SupervisedLearning_files/figure-html/unnamed-chunk-16-1.png)<!-- -->![](6_SupervisedLearning_files/figure-html/unnamed-chunk-16-2.png)<!-- -->![](6_SupervisedLearning_files/figure-html/unnamed-chunk-16-3.png)<!-- -->![](6_SupervisedLearning_files/figure-html/unnamed-chunk-16-4.png)<!-- -->

Model AIC/BIC and mean residual deviance

```r
AIC(credit.gam)
```

```
## [1] 1801.531
```

```r
BIC(credit.gam)
```

```
## [1] 2204.529
```

```r
credit.gam$deviance
```

```
## [1] 1675.826
```

## In-sample fit performance
In order to see the in-sample fit performance, you may look into the confusion matrix by using commands as following. 


```r
pcut.gam <- .08
prob.gam.in<-predict(credit.gam,credit.train,type="response")
pred.gam.in<-(prob.gam.in>=pcut.gam)*1
table(credit.train$Y,pred.gam.in,dnn=c("Observed","Predicted"))
```

```
##         Predicted
## Observed    0    1
##        0 3393  838
##        1  102  167
```

Likewise, misclassification rate is another thing you can check:


```r
mean(ifelse(credit.train$Y != pred.gam.in, 1, 0))
```

```
## [1] 0.2088889
```

Training model AIC and BIC:

```r
AIC(credit.gam)
```

```
## [1] 1801.531
```

```r
BIC(credit.gam)
```

```
## [1] 2204.529
```

## Search for optimal cut-off probability

The following code does a grid search from pcut = 0.01 to pcut = 0.99 with the objective of minimizing overall cost in the training set. I am using an asymmetric cost function by assuming that giving out a bad loan cost 10 time as much as rejecting application from someone who can pay.


```r
#define the searc grid from 0.01 to 0.20
searchgrid = seq(0.01, 0.20, 0.01)
#result.gam is a 99x2 matrix, the 1st col stores the cut-off p, the 2nd column stores the cost
result.gam = cbind(searchgrid, NA)
#in the cost function, both r and pi are vectors, r=Observed, pi=predicted probability
cost1 <- function(r, pi){
  weight1 = 10
  weight0 = 1
  c1 = (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

for(i in 1:length(searchgrid))
{
  pcut <- result.gam[i,1]
  #assign the cost to the 2nd col
  result.gam[i,2] <- cost1(credit.train$Y, predict(credit.gam,type="response"))
}
plot(result.gam, ylab="Cost in Training Set")
```

![](6_SupervisedLearning_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

```r
index.min<-which.min(result.gam[,2])#find the index of minimum value
result.gam[index.min,2] #min cost
```

```
##           
## 0.4088889
```

```r
result.gam[index.min,1] #optimal cutoff probability
```

```
## searchgrid 
##       0.11
```

## Out-of-sample fit performance

```r
pcut <-  result.gam[index.min,1] 
prob.gam.out<-predict(credit.gam,credit.test,type="response")
pred.gam.out<-(prob.gam.out>=pcut)*1
table(credit.test$Y,pred.gam.out,dnn=c("Observed","Predicted"))
```

```
##         Predicted
## Observed   0   1
##        0 421  48
##        1  13  18
```
mis-classifciation rate is

```r
mean(ifelse(credit.test$Y != pred.gam.out, 1, 0))
```

```
## [1] 0.122
```
Cost associated with misclassification is

```r
creditcost(credit.test$Y, pred.gam.out)
```

```
## [1] 0.356
```

[go to top](#header)


# Discriminant Analysis
Linear Discriminant Analysis (LDA) (in-sample and out-of-sample performance measure) is illustrated here. The following illustrate the usage of an arbitrary cut off probability.

## In-sample

```r
credit.train$Y = as.factor(credit.train$Y)
credit.lda <- lda(Y~.,data=credit.train)
prob.lda.in <- predict(credit.lda,data=credit.train)
pcut.lda <- .15
pred.lda.in <- (prob.lda.in$posterior[,2]>=pcut.lda)*1
table(credit.train$Y,pred.lda.in,dnn=c("Obs","Pred"))
```

```
##    Pred
## Obs    0    1
##   0 3892  339
##   1  162  107
```

```r
mean(ifelse(credit.train$Y != pred.lda.in, 1, 0))
```

```
## [1] 0.1113333
```

## Out-of-sample

```r
lda.out <- predict(credit.lda,newdata=credit.test)
cut.lda <- .12
pred.lda.out <- as.numeric((lda.out$posterior[,2]>=cut.lda))
table(credit.test$Y,pred.lda.out,dnn=c("Obs","Pred"))
```

```
##    Pred
## Obs   0   1
##   0 430  39
##   1  14  17
```

```r
mean(ifelse(credit.test$Y != pred.lda.out, 1, 0))
```

```
## [1] 0.106
```

```r
creditcost(credit.test$Y, pred.lda.out)
```

```
## [1] 0.358
```
[go to top](#header)


# Neural Networks Models
Neural Networks method (in-sample and out-of-sample performance measure) is illustrated here. The package [**nnet**](http://cran.r-project.org/web/packages/nnet/nnet.pdf) is used for this purpose.

__Note__: 

- For classification problems with nnet you need to code the response to _factor_ first. In addition you want to add type = "class" for _predict()_  function. 

- For regression problems add lineout = TRUE when training model. In addition, the response needs to be standardized to $[0, 1]$ interval.


```r
Boston.nnet<-nnet(medv~.,size=4,data=Boston,linout=TRUE)
```

## Training

```r
library(nnet)
```


```r
credit.nnet <- nnet(Y~., data=credit.train, size=1, maxit=500)
```

```
## # weights:  63
## initial  value 3055.669366 
## iter  10 value 1018.581755
## iter  20 value 940.135600
## iter  30 value 883.508672
## iter  40 value 869.912515
## iter  50 value 860.581290
## iter  60 value 856.603470
## iter  70 value 855.625803
## iter  80 value 855.567878
## iter  90 value 855.438997
## iter 100 value 855.435679
## final  value 855.435213 
## converged
```

## Out-of-sample Testing

```r
prob.nnet= predict(credit.nnet,credit.test)
pred.nnet = as.numeric(prob.nnet > 0.08)
table(credit.test$Y,pred.nnet, dnn=c("Observed","Predicted"))
```

```
##         Predicted
## Observed   0   1
##        0 260 209
##        1   6  25
```

```r
mean(ifelse(credit.test$Y != pred.nnet, 1, 0))
```

```
## [1] 0.43
```

```r
creditcost(credit.test$Y, pred.nnet)
```

```
## [1] 0.538
```


[go to top](#header)

# Support Vector Machine (SVM)

SVM is probably one of the best off-the-shelf classifiers for many of problems. It handles nonlinearity, is well regularized (avoids overfitting), have few parameters, and fast for large number of observations. It can be adapted to handle regression problems as well. You can read more about SVM in Chapter 12 of the textbook. 

The R package e1071 offers an interface to the most popular svm implementation libsvm. You should read more about the usage of the package in this short tutorial (http://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf).



```r
install.packages('e1071')
```


```r
library(e1071)
credit.svm = svm(Y ~ ., data = credit.train, cost = 1, gamma = 1/length(credit.train), probability= TRUE)
prob.svm = predict(credit.svm, credit.test, probability = TRUE)
prob.svm = attr(prob.svm, 'probabilities')[,2] #This is needed because prob.svm gives a 
pred.svm = as.numeric((prob.svm >= 0.08))
table(credit.test$Y,pred.svm,dnn=c("Obs","Pred"))
```

```
##    Pred
## Obs   0   1
##   0 403  66
##   1  14  17
```

```r
mean(ifelse(credit.test$Y != pred.svm, 1, 0))
```

```
## [1] 0.16
```

```r
creditcost(credit.test$Y, pred.svm)
```

```
## [1] 0.412
```

credit.svm = svm(Y ~ ., data = credit.train, cost = 1, gamma = 1/length(credit.train), probability= TRUE)
prob.svm = predict(credit.svm, credit.test)

[go to top](#header)

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


```r
data(iris)
id_train <- sample(nrow(iris),nrow(iris)*0.80)
iris.train = iris[id_train,]
iris.test = iris[-id_train,]
iris.svm = svm(Species ~ ., data = iris.train)
table(iris.test$Species, predict(iris.svm, iris.test), dnn=c("Observed","Predicted"))
```

```
##             Predicted
## Observed     setosa versicolor virginica
##   setosa         14          0         0
##   versicolor      0          8         0
##   virginica       0          0         8
```


[go to top](#header)

# Starter code for German credit scoring
Refer to http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) for variable description. Notice that "It is worse to class a customer as good when they are bad (weight = 5), than it is to class a customer as bad when they are good (weight = 1)." Define your cost function accordingly!


```r
install.packages('caret')
```


```r
library(caret) #this package contains the german data with its numeric format
data(GermanCredit)
```

[go to top](#header)
