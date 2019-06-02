Introduction to R and R Studio
==============================

Powerful Statistical language: R
--------------------------------

Without a doubt, R is nibbling other programming languages that popularly used by data scientists or statisticians. Recently, most of statisticians use R to analyze data, fit models, do research. Moreover, a lot of practitioners also tend to use R due to its free, open source and active community full of great researchers. Researchers and many other soft engineers help R community to be organicaly developed. All those cooperations are helping R to become one of the greatest statistical programming language in the world. R Studio, REvolution, GGplot2, Shiny App etc. are a few great tools in this community which play big role to make this language great.

For people who want to study frontier statistical methodologies, R helps them to be able to focus on the implementation of statistical methods. One can easily implement new estimation method or algorithms for simulation or comparison purpose, meanwhile, one can easily develop R packages for the methods or models they proposed, which can be used by practionor, maybe right after the method is published. This cannot happen before when statistician know few about software developing and engineers also care less about statistical theories. However, by learning a few of simple R package developing knowledge, one can develop statistical package for their new methods so that people in industry can quickly apply research fruits to accelare their business or improve their decision making, just as simple as clicking button in SPSS. On the other side, practitioners can also develop package for their own projects, which makes their intellectual property reproducible for further using (they don't necessarily need to publish their package to the public). Although open source may face some issues like infrequent maintenance or security, it makes the flow of knowledge seamless.

And it is easy to learn as learning how to use a package in R. Here is the tutorial: <https://shiny.rstudio.com/tutorial/>

Extreme Stong IDE: R Studio
---------------------------

R Studio is a very active community to enrich functions of R. There are several great features I want to emphisis and recommend.

1.  R Markdown or R Notebook

Markdown is an elegant syntax for writers, no matter he or she is a novelist or even in social science or nature science fields, which need more sophisticated notations or graphical and numerical results. Its simple syntax allows writers to be able to focus on the essential part of knowledge sharing: the content and idea itself. Although LaTex is a great tool to generate beautiful document, researchers, unlike publisher, doesn't really need to know much about a lot of tedious things like: formating, typesetting, referencing, and perfectly positioning statistical tables or figures. Because smooth writing process leads to fluent communication with readers, the advantage of Markdown language is obvious.

Here is Markdown Basics: <http://rmarkdown.rstudio.com/authoring_basics.html> And many other tutorials are available in the same website.

New version of R studio provides R Notebook feature that is similar to R markdown. Both of them provide an easy environment to test and iterate when writing article with code. You can run programs in the article and display whatever results you want to insert.

1.  Shiny Web App

Shiny Web App provide a very important pipeline for statisticians and data scientists to interact with audience or customers. With this simple and user-friendly tool, those who are playing with data can directly interpret what they've found to those who are interested in. This process was used to be complicated and need the involvement of professionals in software developing or web developing. The Shiny App, however, simplify the process and provide a intuitive way to build interactive application upon statistical results. Therefore, it is a nice tool that is worthy to learn so that you can smoothly tranfer your idea into real product.

<!-- 3. Hypothesis test and p-value -->
<!-- This a simple example of hypothesis test and why Statistically speaking, this is the paradigm of frequentist. There is an example which will make this clear. If some day, one girl saw her boyfriend is dating with her roommate. So she wants to test whether her boyfriend cheat and is going to find evidence to support her guessing, which means she will find data to prove her boyfriedn and roommate have close relationship (reject they don’t). Then she can put H0 as “boyfriend doesn’t cheat” (common sense) and H1 as “boyfriend do cheat” (interests). Now, if this girl tried her best to find all evidences(data), but the evidences don’t imply her boyfriend has cheated, which in statistics, fails to reject H0. Okay, right now, the girl would say: “I cannot accept you don’t cheat yet, I just cannot find evidence to prove you’ve cheated”. LOL. The point is hypotheses test is trying to reject something rather to accept something, which is very intuitive idea. We cannot find evidence to show we are not real doesn’t mean we can accept we are truly real.  -->
Preparations
============

Download and Installation
-------------------------

**R**

-   Download at <https://cran.r-project.org/>
-   Windows user: click base and then Download R 3.4.1 (or later version) for Windows. <http://cran.r-project.org/bin/windows/base/>
-   Mac user: click R-3.4.1.pkg (or later version). <http://cran.case.edu/bin/macosx/R-latest.pkg>
-   Follow the instructions to install R.

**R Studio**

-   Download at <https://www.rstudio.com/products/rstudio/download/>
-   Follow the instructions to install RStudio.

**Shiny App**

<https://shiny.rstudio.com/>

**R Markdown**

<http://rmarkdown.rstudio.com/>

The download and installation should be straightforward, in case you encounter problems you can check the following video tutorials.

*Install R:* <http://www.youtube.com/watch?v=SJ9sVyqWJn8&hd=1>

*Install R Studio:* <http://www.youtube.com/watch?v=6aTRbo7kdGk&hd=1>

[go to top](#header)

R Studio
--------

RStudio is running based on R. It is an IDE (Intergrated Development Environment) with many advanced features. This lab notes is created based R Markdown, a very nice and useful tool from RStudio.

After you open RStudio, it should look like this: ![](Rstudio.png)

There are three panels showing. However, you need the forth one, which is the editor window. Click the green-plus icon on left-top corner, and select R Script. You write all your code in this editor window, and remember to save it!

Other Panels

-   Console: It shows any command you have run and corresponding output.
-   Environment: It shows what you currently have. Data you have loaded, functions that have been defined, and other R objectives.
-   File/Plot...: Files in current working directory, latest plot you have generated...

Packages
--------

R is open source software. That means, everyone can contribute to it by writing R packages and sharing to the community. An package usually consists of several R functions and datasets that are designed for specific tasks. There are over 10,000 packages in CRAN.

You need to download the package first and then load it to working environment before using some particular functions. We will see this later.

You may call yourself software developer if you can write R packages. If you are interested in writing package, here is a good book to read <http://r-pkgs.had.co.nz/>.

Learning Resource
-----------------

-   [Google](https://www.google.com/): simply search “how to … with R”.
-   [Stack Overflow](https://stackoverflow.com/): a searchable Q&A site oriented toward programming issues.
-   [Cross Validated](https://stats.stackexchange.com/): a searchable Q&A site oriented toward statistical analysis.
-   [R-bloggers](https://www.r-bloggers.com/): a central hub of content from over 500 bloggers who provide news and tutorials about R.
-   Use *help()* or Question Mark in R console: This is the most convinient way to learn R functions. More than 80% of the time during my programming was looking at the help document in R.
-   Please try **“?lm”** (type it in your console), or **help(lm)**

Fundamentals
============

Before Coding
-------------

### Set Working Directory

Always set working directory before you start coding. Working directory where you may read external data, write data, and save the code.

-   **Look at current working directory**: type **getwd()** in console.
-   **Set working directory**: use **setwd("the path")**,

Or

Click **Session -&gt; Set Working Directory -&gt; Choose Directory**, then choose the folder to which you wish to save your work. This will be the default Create a "R Script", name your R Script and save it. Then you can start writing code in the editor window.

Your objects (loaded datasets, variables, functions, etc.) are contained in your "current workspace", which can be saved any time. In Rstudio: Session -&gt; Load Workspace/Save Workspace As....

**Remember:** Keep it tidy! Keep separate projects (code, data files) in separate workspaces/directories.

Use R as Calculator
-------------------

You can assign numbers and lists of numbers (vector) to a variable. Assignment is specified with the "&lt;-" or "=" symbol. They assign the RHS value to LHS object. There are some subtle differences between them but most of the time they are equivalent. I highly suggest you to use "&lt;-" when you want to do assignment, but use "=" in the argument of function(May explain later).

Here we define two variables *x* = 10 and *y* = 5, then we calculate the result of *x* + *y*=. Type following code in the editor and run line by line. To run a line of code, you can move cursor to that line, and use **Crtl+Enter** (**Command+Enter** for Mac). If you want to run multiple lines of code, simply highlight those lines and use the same command. (Note that you can put a \# in front of a line to write comment in code.)

``` r
x <- 10
y = 5
x+y
```

    ## [1] 15

> After you run the code, what did you find in the Global Environment (Workspace) window?

In RStudio, you can view every variable you defined in the Global Environment (Workspace) window, along with other objects such as imported datasets in the *Workspace* panel. You can use R as an over-qualified calculator. Try the following commands. You already have *x*, *y* defined. Then you can calculate *l**o**g*(*x*)=

``` r
log(x)
```

    ## [1] 2.302585

*e**x**p*(*y*)=

``` r
exp(y)
```

    ## [1] 148.4132

*c**o**s*(*x*)=

``` r
cos(x)
```

    ## [1] -0.8390715

The log, exp, cos operators are *functions* in r. They take inputs (also called *arguments*) in parentheses and give outputs.

You can also run logical operations, such as *x* = =*y*, *x* &gt; *y*:

``` r
x == y
```

    ## [1] FALSE

``` r
x > y
```

    ## [1] TRUE

> **Exercise: **
>
> Economic Order Quantity Model: $Q= \\sqrt{2DK/h}$
>
> -   D=5000: annual demand quantity
> -   K=$4: fixed cost per order
> -   h=$0.5: holding cost per unit
> -   **Q=?**

Data Structure: Vector, Matrix, Data frame, and List
----------------------------------------------------

There are four types of data structure in R: Vector, Matrix, Data frame, and List

### **Vector**

-   **Vector** is a list of numbers (or strings), such as the *z* above. It is a vector with elements: \[3, 5, 7, 9\]. There are some basic calculations on list of numbers.

To assign a list of numbers (vector) to a variable, the numbers within the c command are separated by commas. As an example, we can create a new variable, called "z" which will contain the numbers 3, 5, 7, and 9.

``` r
# Define numerical vector z
z<- c(3,5,7,9)
# Define character vector zz, where numerical operations cannot be directly applied.
zz<- c("cup", "plate", "pen", "paper")
```

#### Average

``` r
#Average
mean(z)
```

    ## [1] 6

#### Standard devidation

``` r
#Standard devidation
sd(z)
```

    ## [1] 2.581989

#### Median

``` r
#Median
median(z)
```

    ## [1] 6

#### Maximum

``` r
#Max
max(z)
```

    ## [1] 9

#### Minimum

``` r
#Min
min(z)
```

    ## [1] 3

#### Summary statistics

``` r
#Summary Stats
summary(z)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     3.0     4.5     6.0     6.0     7.5     9.0

#### Calculation

Elementwise operations for single vector or vectors:

``` r
z
```

    ## [1] 3 5 7 9

``` r
z+2
```

    ## [1]  5  7  9 11

``` r
z/10
```

    ## [1] 0.3 0.5 0.7 0.9

``` r
# define vector z1
z1 <- c(2,4,6,8)
# Elementwise operations (must be the same length)
z+z1
```

    ## [1]  5  9 13 17

``` r
z*z1
```

    ## [1]  6 20 42 72

Vector of multiple vectors is still a vector. z2

``` r
# define vector z2
z2 <- c(z, z1)
z2
```

    ## [1] 3 5 7 9 2 4 6 8

#### Indexing

How to extract the second entry of vector z2=(3,5,7,9,2,4,6,8)?

``` r
z2[2]
```

    ## [1] 5

How to extract all elements greater than 3 from vector z2?

``` r
z2[z2>3]
```

    ## [1] 5 7 9 4 6 8

How to extract all elements greater than 3 and smaller than 6 from vector z2?

``` r
z2[z2>3 & z2<6]
```

    ## [1] 5 4

How to order the vector z2 from smallest to largest?

``` r
z2[order(z2)]
```

    ## [1] 2 3 4 5 6 7 8 9

> **Exercise:**
>
> -   What is dot product(inner product) of z and z1?
> -   Find the elements of z2 that smaller than 3 or greater than 7.

### **Matrix**

**Matrix** is a table of numbers (or strings). *A* is a matrix with 2 rows and 3 columns.

#### Creating

``` r
z = c(3,5,7,9)

A = matrix(data = c(1,2,3,4,5,6), nrow = 2)
```

*matrix()* is a function that creates a matrix from a given vector. Some of the arguments in a function can be optional. For example you can also add the *ncol* arguments, which is unnecessary in this situation.

``` r
A <- matrix(data = c(1,2,3,4,5,6), nrow = 2, ncol = 3)
```

Another way to write the function is to ignore the argument names and just put arguments in the right order, but this may cause confusion for readers.

``` r
A <- matrix(c(1,2,3,4,5,6), 2, 3)
```

> **Question**: Think about what would it be if specify ncol=2, or ncol=4?

The default order to position the numbers of a vector to matrix is by column(from top to bottom), but you can specify it as by row using an additional argument **byrow=TRUE**.

``` r
A <- matrix(data = z2, nrow = 4, ncol = 2, byrow = TRUE)
```

#### Calculation

Elementwise operations for matrices:

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

Deimension

``` r
# Dimensions of A
dim(A)
```

    ## [1] 2 3

Transpose and Multiplication

``` r
# Transpose
t(A)
```

    ##      [,1] [,2]
    ## [1,]    1    2
    ## [2,]    3    4
    ## [3,]    5    6

``` r
# Matrix multiplication is doable if and only if the number of columns in A1 equals the number of rows in A2
t(A) %*% A
```

    ##      [,1] [,2] [,3]
    ## [1,]    5   11   17
    ## [2,]   11   25   39
    ## [3,]   17   39   61

``` r
# New matrix with dimension 4*2
A2 <- A * 2
# Matrix calculation should satisfy the rules of matrix algebra
A + A2
```

    ##      [,1] [,2] [,3]
    ## [1,]    3    9   15
    ## [2,]    6   12   18

> **Question**: What would happen if run A %\*% A2?

#### Indexing

How to extract the second entry of second row from matrix A?

``` r
A[2,2]
```

    ## [1] 4

How to extract the first row from matrix A?

``` r
A[1, ]
```

    ## [1] 1 3 5

How to extract first two column from matrix A?

``` r
A[,1:2]
```

    ##      [,1] [,2]
    ## [1,]    1    3
    ## [2,]    2    4

> **Exercise:**
>
> -   What are the diagonal elements of t(A) %\*% A?

### **Data frames**

**Data frames** in R are the "datasets", that is tables of data with each row as an observation, and each column representing a variable. Data frames have column names (variable names) and row names.

#### Creating

-   Convert a matrix to data frame

You can use *data.frame()* to transform a matrix into a dataframe. Most of the time you will import a text file as a data frame or use one of the example datasets that come with R.

``` r
mydf <- data.frame(A) 
class(mydf)
```

    ## [1] "data.frame"

-   Read external data file (.txt and .csv files)

Use the *read.table* or *read.csv* function to import comma/space/tab delimited text files. You can also use the Import Dataset Wizard in RStudio. Package “readxl” allows you to read xls/xlsx files. First, download the storks datasets ( [storks.cvs](data/storks.csv) and [storks.txt](data/storks.txt) files) and save them into your Working Directory.

``` r
mydata_csv <- read.csv("storks.csv", header=TRUE)
mydata_txt <- read.table("storks.txt", header=TRUE, sep = "\t")
```

-   Load built-in dataset

``` r
#Load cars dataset that comes with R (50 obs, 2 variables)
data(cars)
```

-   Summary of a dataset

``` r
#Dimension 
dim(cars)
#Preview the first few rows
head(cars)
#Variable names
names(cars)
#Summary
summary(cars)
#Structure
str(cars)
```

#### Manipulating

Subsetting elements from data frames is similar to subsetting from matrices. On the other hand, since data frames have variable names (label for each column), you can also use the following two ways to refer variables of a data frame:

-   df$var will select var from df
-   df\[, c('var1','var2')\] will select var1 and var2 from df

In RStudio, hitting tab after `df$` allows you to select/autocomplete variable names in df

-   Adding and dropping variables

Add new variable to the data

``` r
#First 2 obs of the variable dist in cars
cars$dist[1:2]
```

    ## [1]  2 10

``` r
cars1<- cars
cars1$time<- cars$dist/cars$speed
```

Drop variable *time*

``` r
# since "time" is the third column, we can do
cars2<- cars1[,-3]
# we can also drop "time" by keeping the other two variables
cars3<- cars1[c("speed", "dist")]
```

### **List**

**List** is a container. You can put different types of objects into a list to create your own list of all you have in hand.

``` r
mylist<- list(myvector=z, mymatrix=A, mydata=cars)
```

Most of the output of R function is a list that contains severl objects.

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

There are three ways to get an element from a list:

-   use *listname\[\[i\]\]* to get the ith element of the list;
-   use *listname\[\["elementname"\]\]*;
-   use *listname$elementname*.

Note that you use double square brackets for indexing a list.

``` r
reg <- lm(dist~speed, data = cars)
reg[[1]]
reg[["coeffcients"]]
reg$coeffcients
```

If you have done object oriented programming before, the list "reg" is actually an object that belongs to class "lm". The element names such as "coeffcients" are fields of the "lm" class.

### **Exercise**

> 1.  Define a vector with values (5, 2, 11, 19, 3, -9, 8, 20, 1). Calculate the sum, mean, and standard deviation.
> 2.  Re-order the vector from largest to smallest, and make it a new vector.
> 3.  Convert the vector to a 3\*3 matrix ordered by column. What is the sum of first column? What is the number in column 2 row 3? What is the column sum?
>
> 4.  Download the [CustomerData](data/CustomerData.csv) to your working directory. Load it to R.
>
> -   How many rows and columns are there?
> -   Extract all variable names.
> -   What is the average “Debt to Income Ratio”?
> -   What is the proportion of “Married” customers?

Basic Plotting
--------------

A Simple Scatter Plot

``` r
plot(cars)
```

<img src="1_IntroR_files/figure-markdown_github/unnamed-chunk-34-1.png" style="display: block; margin: auto;" />

Probability Distributions
-------------------------

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

    ##  [1]  2.0187667  0.5251947 -1.4379246  0.2690220 -1.5316621 -0.2366348
    ##  [7]  1.0171028  1.5966312 -0.5837695  1.6401373

``` r
rnorm(n=10,mean=100,sd=20)
```

    ##  [1]  80.02516  95.42259 142.60259  90.98820  52.64100  85.11787  94.80475
    ##  [8]  90.76059 115.50536 110.76228

Functions
---------

-   R programming is essential applying and writing functions. Most of R consists of functions.
-   An R function may require multiple inputs, we call them argument. The arguments can either be input in the right order, or using argument names. In RStudio, pressing tab after function name gives help about arguments
-   Using “?+function name” to learn how to use that funcion.
-   We introduce how to write simple functions here. In the following example the function *abs\_val* returns the absolute value of a number.

``` r
abs_val = function(x){
  if(x >= 0){
    return(x)
  }
  else{
    return(-x)
  }
}
abs_val(-5)
```

    ## [1] 5

A function for vector truncation

``` r
mytruncation<- function(v, lower, upper){
  v[which(v<lower)]<- lower
  v[which(v>upper)]<- upper
  return(v)
}
```

You just defined a global function for truncation. Now let’s apply it to vector z2, where we truncate at lower=3 upper=7.

``` r
mytruncation(v = z2, lower = 3, upper = 7)
```

    ## [1] 3 5 7 7 3 4 6 7

Loop
----

There are two ways to write a loop: while and for loop. Loop is very useful to do iterative and duplicated computing.

For example: calculate 1 + 1/2 + 1/3 + ... + 1/100.

### Using **while** loop

``` r
i<- 1
x<- 1
while(i<100){
  i<- i+1
  x<- x+1/i
}
x
```

    ## [1] 5.187378

### Using **for** loop

``` r
x<- 1
for(i in 2:100){
  x<- x+1/i
}
x
```

    ## [1] 5.187378

> **Exercise:**
>
> 1.  Do you think 1 + 1/2<sup>2</sup> + 1/3<sup>2</sup> + ... + 1/*n*<sup>2</sup> converges or diverges as *n* → ∞? Use R to verify your answer.
> 2.  Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13,… What is the next number? What is the 50th number? Creat a vector of first 30 Fibonacci numbers.
> 3.  Write a function that can either calculate the summation of the serie in Question 1 or generate and print Fibonacci sequence in Question 2.

[go to top](#header)

Summary
=======

Do you like R?
--------------

### YES!

That is great! Go R!

### NO!

You may have trouble in the rest of the semester..., so please try to get used to it!

What you need to remember
-------------------------

-   Set working directory.
-   How to creat a vector?
-   How to load an external data file?
-   How to subset/index a vector, a matrix, and a data frame?
-   Basic calculations and syntax.

[go to top](#header)

Get started with Rattle
=======================

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

Starting Rattle
---------------

To start Rattle, first load the Rattle package (after you install it), and use *rattle()* command to launch it.

``` r
library(rattle)
rattle()
```

When using Rattle, it will ask you to install many packages along the way, you will have to keep hitting the Yes button.

Note: Part of this handout is built upon Clarkson University's R Tutorial [2](http://www.cyclismo.org/tutorial/R/) and Dr.Yan Yu's slides.

[go to top](#header)
