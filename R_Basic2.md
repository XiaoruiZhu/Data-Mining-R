Introduction to R
=================

Outline of this lab
-------------------

1.  Install R and RStudio
2.  Learn basic commands and data types in R
3.  Get started with Rattle

Before You Start: Install R, RStudio
------------------------------------

**R**

Windows: <http://cran.r-project.org/bin/windows/base/>

Mac OS X: <http://cran.case.edu/bin/macosx/R-latest.pkg>

**RStudio**

<http://www.rstudio.com/ide/download/desktop>

**Shiny App**

<https://shiny.rstudio.com/>

**R Markdown**

<http://rmarkdown.rstudio.com/>

The download and installation should be straightforward, in case you encounter problems you can check the following video tutorials.

*Install R:* <http://www.youtube.com/watch?v=SJ9sVyqWJn8&hd=1>

*Install R Studio:* <http://www.youtube.com/watch?v=6aTRbo7kdGk&hd=1>

R Basics
--------

### Assignment

You can assign numbers and lists of numbers (vector) to a variable. Assignment is specified with the "&lt;-" or "=" symbol. There are some subtle differences between them but most of the time they are equivalent. I highly suggest you to use "&lt;-" when you want to do assignment, but use "=" in the argument of function(May explain later).

Here we define two variables *x* = 10 and *y* = 5, then we calculate the result of *x* + *y*.

``` r
x <- 10
y = 5
x+y
```

    ## [1] 15

To assign a list of numbers (vector) to a variable, the numbers within the c command are separated by commas. As an example, we can create a new variable, called "z" which will contain the numbers 3, 5, 7, and 9:

``` r
z = c(3,5,7,9)
```

In RStudio, you can view every variable you defined, along with other objects such as imported datasets in the *Workspace* panel.

### Basic Calculation

You can use R as an over-qualified calculator. Try the following commands. You need to have *x*, *y*, *z* defined first.

``` r
x+y
```

    ## [1] 15

``` r
log(x)
```

    ## [1] 2.302585

``` r
exp(y)
```

    ## [1] 148.4132

``` r
cos(x)
```

    ## [1] -0.8390715

The log, exp, cos operators are *functions* in r. They take inputs (also called *arguments*) in parentheses and give outputs.

Logical operations:

``` r
x == y
```

    ## [1] FALSE

``` r
x > y
```

    ## [1] TRUE

Calculations on list of numbers, recall *z* = \[3, 5, 7, 9\]. Note that you can put a \# in front of a line to write comment in code.

``` r
#Average
mean(z)
```

    ## [1] 6

``` r
#Standard devidation
sd(z)
```

    ## [1] 2.581989

``` r
#Median
median(z)
```

    ## [1] 6

``` r
#Max
max(z)
```

    ## [1] 9

``` r
#Min
min(z)
```

    ## [1] 3

``` r
#Summary Stats
summary(z)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     3.0     4.5     6.0     6.0     7.5     9.0

### Data Types in R: Vectors, matrices, data frames

-   **Vector** is a list of numbers (or strings), *z* is a vector with \[3, 5, 7, 9\]

-   **Matrix** is a table of numbers (or strings). *A* is a matrix with 2 rows and 3 columns.

``` r
z = c(3,5,7,9)

A = matrix(data = c(1,2,3,4,5,6), nrow = 2)
```

*matrix()* is a function that creates a matrix from a given vector. Some of the arguments in a function can be optional. For example you can also add the *ncol* arguments, which is unnecessary in this situation.

``` r
A = matrix(data = c(1,2,3,4,5,6), nrow = 2, ncol = 3)
```

Another way to write the function is to ignore the argument names and just put arguments in the right order, but this may cause confusion for readers.

``` r
A = matrix(c(1,2,3,4,5,6), 2, 3)
```

Elementwise operations for vectors and matrices

``` r
z
```

    ## [1] 3 5 7 9

``` r
z/10
```

    ## [1] 0.3 0.5 0.7 0.9

``` r
A
```

    ##      [,1] [,2] [,3]
    ## [1,]    1    3    5
    ## [2,]    2    4    6

``` r
A+2
```

    ##      [,1] [,2] [,3]
    ## [1,]    3    5    7
    ## [2,]    4    6    8

Subsetting, indexing

``` r
length(z)
```

    ## [1] 4

``` r
z[1:3]
```

    ## [1] 3 5 7

``` r
#All but the first element in z
z[-1]
```

    ## [1] 5 7 9

``` r
A[2,2]
```

    ## [1] 4

``` r
A[1, ]
```

    ## [1] 1 3 5

``` r
A[, 2:3]
```

    ##      [,1] [,2]
    ## [1,]    3    5
    ## [2,]    4    6

``` r
A[, c(1,3)]
```

    ##      [,1] [,2]
    ## [1,]    1    5
    ## [2,]    2    6

Matrix Calculation

``` r
# Dimensions of A
dim(A)
```

    ## [1] 2 3

``` r
# Transpose
t(A)
```

    ##      [,1] [,2]
    ## [1,]    1    2
    ## [2,]    3    4
    ## [3,]    5    6

``` r
# Matrix multiplication
t(A) %*% A
```

    ##      [,1] [,2] [,3]
    ## [1,]    5   11   17
    ## [2,]   11   25   39
    ## [3,]   17   39   61

-   **Data frames** in R are the "datasets", that is tables of data with each row as an observation, and each column representing a variable. Data frames have column names (variable names) and row names. In the example, the function *data.frame(A)* transforms the matrix *A* into a data frame. Most of the time you will import a text file as a data frame or use one of the example datasets that come with R.

You can use *data.frame()* to transform a matrix into a dataframe.

``` r
df <-  data.frame(A) 
```

Use the *read.table* or *read.csv* function to import comma/space/tab delimited text files. You can also use the Import Dataset Wizard in RStudio.

``` r
mydata <-  read.table("c:/mydata.csv", header=TRUE, sep=",")
```

Subsetting elements from data frames is similar to subsetting from matrices. Since data frames have variable names (label for each column), you can also use

-   df$var will select var from df
-   df\[, c('var1','var2')\] will select var1 and var2 from df

In RStudio, hitting tab after `df$` allows you to select/autocomplete variable names in df

``` r
#Load cars dataset that comes with R (50 obs, 2 variables)
data(cars)
#Dimension 
dim(cars)
```

    ## [1] 50  2

``` r
#Preview the first few lines
head(cars)
```

    ##   speed dist
    ## 1     4    2
    ## 2     4   10
    ## 3     7    4
    ## 4     7   22
    ## 5     8   16
    ## 6     9   10

``` r
#Variable names
names(cars)
```

    ## [1] "speed" "dist"

You can combine multiple ways to subset data

``` r
#First 2 obs of the variable dist in cars
cars$dist[1:2]
```

    ## [1]  2 10

In the next lab we will see other ways to select data from data frames.

### Regression Model and the Data Type **List**

``` r
# Load car dataset that comes with R
data(cars)
#fit a simple linear regression between braking distance and speed
lm(dist~speed, data=cars)
```

    ## 
    ## Call:
    ## lm(formula = dist ~ speed, data = cars)
    ## 
    ## Coefficients:
    ## (Intercept)        speed  
    ##     -17.579        3.932

-   **List** is a container. You can put different types of objects into a list. For example, the result returned by the lm() function is a list.

There are three ways to get an element from a list: use *listname\[\[i\]\]* to get the ith element of the list; use *listname\[\["elementname"\]\]*; use *listname$elementname*. Note that you use double square brackets for indexing a list.

``` r
reg = lm(dist~speed, data = cars)
reg[[1]]
reg[["coeffcients"]]
reg$coeffcients
```

If you have done object oriented programming before, the list "reg" is actually an object that belongs to class "lm". The element names such as "coeffcients" are fields of the "lm" class.

### Basic Plotting

``` r
plot(cars)
```

<img src="R_Basic_files/figure-markdown_github/unnamed-chunk-14-1.png" style="display: block; margin: auto;" />

### Probability Distributions

Types of distributions: norm, binom, beta, cauchy, chisq, exp, f, gamma, geom, hyper, lnorm, logis, nbinom, t, unif, weibull, wilcox

Four prefixes:

1.  'd' for density (PDF)

2.  'p' for distribution (CDF)

3.  'q' for quantile (percentiles)

4.  'r' for random generation (simulation)

``` r
dbinom(x=4,size=10,prob=0.5)
```

    ## [1] 0.2050781

``` r
pnorm(1.86)
```

    ## [1] 0.9685572

``` r
qnorm(0.975)
```

    ## [1] 1.959964

``` r
rnorm(10)
```

    ##  [1] -0.75103500 -2.26831120 -0.60166045 -0.06982864 -0.39625069
    ##  [6]  0.63673593  1.06150653  0.94319739  0.38068992  0.02175840

``` r
rnorm(n=10,mean=100,sd=20)
```

    ##  [1] 104.03402 105.21126  93.21940  47.87182 149.70196  82.64874 101.04440
    ##  [8] 104.87032 134.06477  79.11772

### Functions

-   Most of R consists of functions
-   The arguments can either be input in the right order, or using argument names
-   In RStudio, pressing tab after function name gives help about arguments
-   You can define your own functions, in the following example the function *abs\_val* returns the absolute value of a number.

``` r
abs_val = function(x){
  if(x >= 0){
    return(x)
  }
  else{
    return(-x)
  }
}
```

``` r
abs_val(-5)
```

    ## [1] 5

### Organizing your computations

1.  R has a "current directory" (Working Directory). To set Working Directory in RStudio: Session -&gt; Set Working Directory.
2.  Your objects (loaded datasets, variables, functions, etc.) are contained in your "current workspace", which can be saved any time. In Rstudio: Session -&gt; Load Workspace/Save Workspace As.
3.  Keep it tidy! Keep separate projects (code, data files) in separate workspaces/directories.

### Help

1.  Use *help(mean)* or *?mean* to find help documents for the function *mean*.
2.  In Rstudio, pressing tab after a function name gives help about the arguments
3.  A google search will probably answer most of your questions.
4.  Where to ask questions?

-   Stackoverflow (a more friendly place): <http://stackoverflow.com/questions/tagged/r>
-   R Help Mailing List: <http://r.789695.n4.nabble.com/R-help-f789696.html>

Install Rattle
--------------

### Rattle: "A Graphical User Interface for Data Mining using R".

With Rattle you can presents statistical and visual summaries of data, transforms data into forms that can be readily modelled, builds both unsupervised and supervised models from the data, presents the performance of models graphically, and scores new datasets.[1](http://rattle.togaware.com/ "Rattle Official Website")

Basically, *R* is a command line language, *RStudio* offers a pleasant development enviroment for you to write code and manage projects, and *Rattle* is a package in R that allows you to do Data Mining tasks in a graphical interface.

To install Rattle, startup R and then run the following command:

``` r
install.packages("rattle")
```

Alternatively, you can use the *Install Packages* in RStudio.

### Starting Rattle

To start Rattle, first load the Rattle package (after you install it), and use *rattle()* command to launch it.

``` r
library(rattle)
rattle()
```

When using Rattle, it will ask you to install many packages along the way, you will have to keep hitting the Yes button.

Note: Part of this handout is built upon Clarkson University's R Tutorial [2](http://www.cyclismo.org/tutorial/R/) and Dr.Yan Yu's slides.
