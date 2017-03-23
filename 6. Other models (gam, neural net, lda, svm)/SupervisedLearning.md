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
##                        1
## (Intercept) -2.512491467
## X2           .          
## X3          -0.001295199
## X4           .          
## X5           .          
## X6           .          
## X7           .          
## X8          -0.303457904
## X10_2        .          
## X11_2       -0.302369562
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
## X17_6       -0.084723561
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
## X22_9        0.071368131
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
##        0 448  16
##        1  25  11
```

```r
mean(ifelse(credit.test$Y != predicted.glm0.outsample, 1, 0))
```

```
## [1] 0.082
```

```r
creditcost(credit.test$Y, predicted.glm0.outsample)
```

```
## [1] 0.532
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
##      Model      Area     p.value binorm.area
## 1 Model  1 0.7716116 2.78187e-08          NA
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
## (Intercept) -3.183557   0.713558  -4.462 8.14e-06 ***
## X6           0.103710   0.116930   0.887 0.375114    
## X7          -0.209606   0.210390  -0.996 0.319117    
## X8          -2.202585   0.333699  -6.601 4.10e-11 ***
## X10_2       -0.307876   0.171197  -1.798 0.072118 .  
## X11_2       -0.858492   0.154535  -5.555 2.77e-08 ***
## X12_2       -0.413956   0.200196  -2.068 0.038663 *  
## X13_2        0.387852   0.160433   2.418 0.015626 *  
## X14_2       -0.303220   0.278516  -1.089 0.276288    
## X15_2        0.488843   0.247266   1.977 0.048043 *  
## X15_3        0.128547   0.308363   0.417 0.676774    
## X15_4        0.916111   0.323064   2.836 0.004573 ** 
## X15_5        0.331345   0.422977   0.783 0.433414    
## X15_6        0.912236   0.275359   3.313 0.000923 ***
## X16_2        0.321787   0.283330   1.136 0.256069    
## X16_3       -0.123986   0.281573  -0.440 0.659696    
## X16_4        0.061037   0.352287   0.173 0.862448    
## X16_5       -0.213205   0.281160  -0.758 0.448269    
## X16_6        0.041987   0.293077   0.143 0.886083    
## X17_2        0.028249   0.250164   0.113 0.910093    
## X17_3       -1.024206   0.318188  -3.219 0.001287 ** 
## X17_4       -0.253920   0.265515  -0.956 0.338906    
## X17_5        0.545569   0.446527   1.222 0.221781    
## X17_6       -1.089472   0.176643  -6.168 6.93e-10 ***
## X18_2        0.065193   0.333031   0.196 0.844802    
## X18_3        0.257164   0.271752   0.946 0.343985    
## X18_4        0.907811   0.252095   3.601 0.000317 ***
## X18_5        0.662137   0.231691   2.858 0.004265 ** 
## X18_6        0.337858   0.296741   1.139 0.254887    
## X18_7        0.729751   0.294591   2.477 0.013243 *  
## X19_2        0.321444   0.358833   0.896 0.370357    
## X19_3        0.569584   0.295136   1.930 0.053619 .  
## X19_4       -0.050133   0.540735  -0.093 0.926132    
## X19_5        0.303066   0.414748   0.731 0.464949    
## X19_6        0.256510   0.459841   0.558 0.576966    
## X19_7        0.677716   0.425012   1.595 0.110806    
## X19_8       -0.878569   0.669711  -1.312 0.189567    
## X19_9        0.722999   0.573512   1.261 0.207434    
## X19_10       0.411087   0.334039   1.231 0.218452    
## X20_2        0.074218   0.352701   0.210 0.833335    
## X20_3       -0.033756   0.271390  -0.124 0.901012    
## X20_4        0.172185   0.185326   0.929 0.352841    
## X21_2        0.076154   0.463064   0.164 0.869372    
## X21_3        0.448564   0.226326   1.982 0.047486 *  
## X22_2       -0.396750   0.400785  -0.990 0.322206    
## X22_3       -0.016324   0.352309  -0.046 0.963044    
## X22_4       -0.050392   0.462748  -0.109 0.913283    
## X22_5        0.138700   0.416319   0.333 0.739016    
## X22_6        0.003221   0.591985   0.005 0.995659    
## X22_7        0.259034   0.367953   0.704 0.481442    
## X22_8        0.132105   0.361873   0.365 0.715068    
## X22_9        0.643945   0.320363   2.010 0.044426 *  
## X22_10      -1.341837   1.080827  -1.241 0.214424    
## X22_11       0.327776   0.347026   0.945 0.344900    
## X23_2        0.018241   0.214709   0.085 0.932296    
## X23_3       -0.139796   0.223964  -0.624 0.532503    
## X24_2        0.347800   0.315535   1.102 0.270350    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##         edf Ref.df Chi.sq  p-value    
## s(X2) 1.001  1.002  0.178   0.6738    
## s(X3) 1.000  1.000 20.897 4.85e-06 ***
## s(X4) 1.579  1.961  0.902   0.6020    
## s(X5) 3.169  3.944  7.732   0.0949 .  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.109   Deviance explained = 18.6%
## UBRE = -0.60823  Scale est. = 1         n = 4500
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
## [1] 1762.965
```

```r
BIC(credit.gam)
```

```
## [1] 2171.719
```

```r
credit.gam$deviance
```

```
## [1] 1635.466
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
##        0 3405  831
##        1  100  164
```

Likewise, misclassification rate is another thing you can check:


```r
mean(ifelse(credit.train$Y != pred.gam.in, 1, 0))
```

```
## [1] 0.2068889
```

Training model AIC and BIC:

```r
AIC(credit.gam)
```

```
## [1] 1762.965
```

```r
BIC(credit.gam)
```

```
## [1] 2171.719
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
## 0.3931111
```

```r
result.gam[index.min,1] #optimal cutoff probability
```

```
## searchgrid 
##       0.14
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
##        0 430  34
##        1  23  13
```
mis-classifciation rate is

```r
mean(ifelse(credit.test$Y != pred.gam.out, 1, 0))
```

```
## [1] 0.114
```
Cost associated with misclassification is

```r
creditcost(credit.test$Y, pred.gam.out)
```

```
## [1] 0.528
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
##   0 3917  319
##   1  153  111
```

```r
mean(ifelse(credit.train$Y != pred.lda.in, 1, 0))
```

```
## [1] 0.1048889
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
##   0 415  49
##   1  21  15
```

```r
mean(ifelse(credit.test$Y != pred.lda.out, 1, 0))
```

```
## [1] 0.14
```

```r
creditcost(credit.test$Y, pred.lda.out)
```

```
## [1] 0.518
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
## initial  value 4671.965577 
## final  value 1004.773218 
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
## Observed   0
##        0 464
##        1  36
```

```r
mean(ifelse(credit.test$Y != pred.nnet, 1, 0))
```

```
## [1] 0.072
```

```r
creditcost(credit.test$Y, pred.nnet)
```

```
## [1] 0.72
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
##   0 400  64
##   1  24  12
```

```r
mean(ifelse(credit.test$Y != pred.svm, 1, 0))
```

```
## [1] 0.176
```

```r
creditcost(credit.test$Y, pred.svm)
```

```
## [1] 0.608
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
##   setosa          7          0         0
##   versicolor      0         12         2
##   virginica       0          2         7
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
