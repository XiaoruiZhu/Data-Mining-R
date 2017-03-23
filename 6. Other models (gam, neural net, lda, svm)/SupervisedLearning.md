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
##                        1
## (Intercept) -2.519725115
## X2           .          
## X3           .          
## X4           .          
## X5           .          
## X6           .          
## X7           .          
## X8          -0.291668579
## X10_2        .          
## X11_2       -0.295353821
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
## X17_6       -0.054737399
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
## X22_9        0.001031891
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
##        0 444  21
##        1  25  10
```

```r
mean(ifelse(credit.test$Y != predicted.glm0.outsample, 1, 0))
```

```
## [1] 0.092
```

```r
creditcost(credit.test$Y, predicted.glm0.outsample)
```

```
## [1] 0.542
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

![](SupervisedLearning_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

```
##      Model      Area      p.value binorm.area
## 1 Model  1 0.7644854 8.880759e-08          NA
```


[go to top](#content)

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
## (Intercept) -3.620559   0.727643  -4.976 6.50e-07 ***
## X6           0.099787   0.114913   0.868 0.385191    
## X7          -0.314288   0.212384  -1.480 0.138924    
## X8          -2.165934   0.326057  -6.643 3.08e-11 ***
## X10_2       -0.365003   0.168617  -2.165 0.030412 *  
## X11_2       -0.799347   0.153966  -5.192 2.08e-07 ***
## X12_2       -0.252378   0.204399  -1.235 0.216930    
## X13_2        0.303940   0.159576   1.905 0.056823 .  
## X14_2       -0.153277   0.285590  -0.537 0.591471    
## X15_2        0.697299   0.250624   2.782 0.005398 ** 
## X15_3        0.318499   0.306263   1.040 0.298361    
## X15_4        1.216188   0.317383   3.832 0.000127 ***
## X15_5        0.486987   0.424796   1.146 0.251629    
## X15_6        1.019752   0.281346   3.625 0.000289 ***
## X16_2        0.303330   0.292503   1.037 0.299729    
## X16_3       -0.029221   0.286545  -0.102 0.918775    
## X16_4        0.319268   0.347530   0.919 0.358264    
## X16_5       -0.157458   0.287128  -0.548 0.583423    
## X16_6        0.096622   0.299366   0.323 0.746881    
## X17_2       -0.059447   0.255601  -0.233 0.816090    
## X17_3       -1.143808   0.323525  -3.535 0.000407 ***
## X17_4       -0.302818   0.270341  -1.120 0.262657    
## X17_5        0.767129   0.418046   1.835 0.066501 .  
## X17_6       -1.066863   0.174984  -6.097 1.08e-09 ***
## X18_2        0.249610   0.332168   0.751 0.452377    
## X18_3        0.361236   0.271288   1.332 0.183005    
## X18_4        1.061571   0.255566   4.154 3.27e-05 ***
## X18_5        0.759140   0.238015   3.189 0.001425 ** 
## X18_6        0.538269   0.289983   1.856 0.063424 .  
## X18_7        0.591878   0.314506   1.882 0.059845 .  
## X19_2        0.276649   0.357750   0.773 0.439343    
## X19_3        0.506153   0.299341   1.691 0.090858 .  
## X19_4        0.028596   0.514520   0.056 0.955679    
## X19_5        0.391434   0.414636   0.944 0.345149    
## X19_6        0.229055   0.455316   0.503 0.614916    
## X19_7        0.536185   0.433623   1.237 0.216264    
## X19_8       -1.361832   0.784855  -1.735 0.082717 .  
## X19_9        0.783425   0.549127   1.427 0.153674    
## X19_10       0.408916   0.334315   1.223 0.221275    
## X20_2       -0.267786   0.401895  -0.666 0.505214    
## X20_3       -0.102302   0.269481  -0.380 0.704222    
## X20_4        0.140997   0.180645   0.781 0.435085    
## X21_2        0.365726   0.444121   0.823 0.410233    
## X21_3        0.568254   0.221889   2.561 0.010437 *  
## X22_2       -0.487012   0.398020  -1.224 0.221109    
## X22_3        0.035337   0.340575   0.104 0.917363    
## X22_4       -0.195420   0.451276  -0.433 0.664987    
## X22_5       -0.157723   0.425622  -0.371 0.710958    
## X22_6       -0.003037   0.585432  -0.005 0.995861    
## X22_7        0.024599   0.366341   0.067 0.946464    
## X22_8       -0.012638   0.352784  -0.036 0.971422    
## X22_9        0.418360   0.313894   1.333 0.182595    
## X22_10      -1.379366   1.073463  -1.285 0.198803    
## X22_11       0.323225   0.338559   0.955 0.339725    
## X23_2        0.108109   0.207156   0.522 0.601757    
## X23_3       -0.265888   0.235118  -1.131 0.258109    
## X24_2        0.440280   0.306301   1.437 0.150601    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##         edf Ref.df Chi.sq p-value    
## s(X2) 1.020  1.040  0.503   0.505    
## s(X3) 1.000  1.000 18.500 1.7e-05 ***
## s(X4) 1.326  1.583  0.184   0.787    
## s(X5) 2.720  3.398  3.761   0.356    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.111   Deviance explained = 18.4%
## UBRE = -0.60654  Scale est. = 1         n = 4500
```

```r
plot(credit.gam, shade=TRUE,,seWithMean=TRUE,scale=0)
```

![](SupervisedLearning_files/figure-html/unnamed-chunk-16-1.png)<!-- -->![](SupervisedLearning_files/figure-html/unnamed-chunk-16-2.png)<!-- -->![](SupervisedLearning_files/figure-html/unnamed-chunk-16-3.png)<!-- -->![](SupervisedLearning_files/figure-html/unnamed-chunk-16-4.png)<!-- -->

Model AIC/BIC and mean residual deviance

```r
AIC(credit.gam)
```

```
## [1] 1770.551
```

```r
BIC(credit.gam)
```

```
## [1] 2174.919
```

```r
credit.gam$deviance
```

```
## [1] 1644.419
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
##        0 3414  821
##        1   99  166
```

Likewise, misclassification rate is another thing you can check:


```r
mean(ifelse(credit.train$Y != pred.gam.in, 1, 0))
```

```
## [1] 0.2044444
```

Training model AIC and BIC:

```r
AIC(credit.gam)
```

```
## [1] 1770.551
```

```r
BIC(credit.gam)
```

```
## [1] 2174.919
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

![](SupervisedLearning_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

```r
index.min<-which.min(result.gam[,2])#find the index of minimum value
result.gam[index.min,2] #min cost
```

```
##           
## 0.4024444
```

```r
result.gam[index.min,1] #optimal cutoff probability
```

```
## searchgrid 
##       0.08
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
##        0 370  95
##        1  14  21
```
mis-classifciation rate is

```r
mean(ifelse(credit.test$Y != pred.gam.out, 1, 0))
```

```
## [1] 0.218
```
Cost associated with misclassification is

```r
creditcost(credit.test$Y, pred.gam.out)
```

```
## [1] 0.47
```

[go to top](#content)


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
##   0 3904  331
##   1  159  106
```

```r
mean(ifelse(credit.train$Y != pred.lda.in, 1, 0))
```

```
## [1] 0.1088889
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
##   0 407  58
##   1  18  17
```

```r
mean(ifelse(credit.test$Y != pred.lda.out, 1, 0))
```

```
## [1] 0.152
```

```r
creditcost(credit.test$Y, pred.lda.out)
```

```
## [1] 0.476
```
[go to top](#content)


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
## initial  value 2617.329390 
## iter  10 value 893.439967
## iter  20 value 858.150310
## iter  30 value 853.791092
## iter  40 value 851.757520
## iter  50 value 850.457905
## iter  60 value 848.878263
## iter  70 value 842.571808
## iter  80 value 820.838498
## iter  90 value 816.774050
## final  value 816.496673 
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
##        0 301 164
##        1   8  27
```

```r
mean(ifelse(credit.test$Y != pred.nnet, 1, 0))
```

```
## [1] 0.344
```

```r
creditcost(credit.test$Y, pred.nnet)
```

```
## [1] 0.488
```


[go to top](#content)

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
##   0 400  65
##   1  17  18
```

```r
mean(ifelse(credit.test$Y != pred.svm, 1, 0))
```

```
## [1] 0.164
```

```r
creditcost(credit.test$Y, pred.svm)
```

```
## [1] 0.47
```

credit.svm = svm(Y ~ ., data = credit.train, cost = 1, gamma = 1/length(credit.train), probability= TRUE)
prob.svm = predict(credit.svm, credit.test)

[go to top](#content)

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
##   setosa          8          0         0
##   versicolor      0          5         0
##   virginica       0          2        15
```


[go to top](#content)

# Starter code for German credit scoring
Refer to http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) for variable description. Notice that "It is worse to class a customer as good when they are bad (weight = 5), than it is to class a customer as bad when they are good (weight = 1)." Define your cost function accordingly!


```r
install.packages('caret')
```


```r
library(caret) #this package contains the german data with its numeric format
data(GermanCredit)
```

[go to top](#content)
