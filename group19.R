## Stats 506, Fall 2018, Group Project.
## Author: Mimi Tran.
## Date: November 25, 2018.
## Data: Insurance

# Library----------------------------------------------------------------------
library(readr)
library(dplyr)
library(faraway)
library(MASS)
# Load data--------------------------------------------------------------------
ds0 <- read_csv("insurance.csv")
names(ds0)

# Check if is there any null---------------------------------------------------
is.na(ds0)
# No missing value, we're good.

# Recode: sex:femal = 1, male = 0, smoke: yes=1, no=0
ds0$sex[ds0$sex == "male"]="0"
ds0$sex[ds0$sex == "female"]="1"
ds0$smoker[ds0$smoker == "no"]="0"
ds0$smoker[ds0$smoker == "yes"]="1"

# Examining the distribution of each variables---------------------------------
hist(ds0$age,xlab="Age", main="Distribution of Age")
hist(ds0$bmi,xlab="BMI", main="Distribution of BMI")
hist(ds0$children,xlab="Children", main="Distribution of Children")
hist(ds0$charges,xlab="Charges", main="Distribution of Charges")

# Take log for charge since its heavy tail-------------------------------------
ds0$logcharges <- log(ds0$charges+1)

## Model diagnostic:-----------------------------------------------------------

# VIF of each column-----------------------------------------------------------
# Load library(faraway) to get funtion vif, and we see vif of each column is ok
# since they're all smaller than 5, even smaller than 2.
fit0 <- lm(logcharges ~age+sex+bmi+children+smoker+as.factor(region), data=ds0)
X <- model.matrix(fit0)[, -1]
round(vif(X),2)


# Residual distribution--------------------------------------------------------
hist(fit0$residuals, xlab="Residuals")
plot(fit0$res, xlab="Residuals")
abline(h=0) # acting like noraml so it's good.

# Partial residual plots-------------------------------------------------------
#Which attempts to show how covariate is related to dependent variable
# if we control for the effects of all other covariates
# partial residual plots look acceptable.
fit <- lm(logcharges~ bmi, data=ds0)
plot(fit)

## Model Selection-------------------------------------------------------------
# Original model
fit0 <- lm(logcharges ~age+sex+bmi+children+smoker+as.factor(region), data=ds0)
summary(fit0)
AIC(fit0)
BIC(fit0)

# Try No.1: add interactive covariate smoker*bmi
fit1 <-lm(logcharges ~age+sex+bmi+children+smoker+as.factor(region)+smoker*bmi, data=ds0)
summary(fit1)
BIC(fit1)
# which certainly improve the performance of the model

# Try No.2: add an interactive covariate smoker*age
fit2 <-lm(logcharges ~age+sex+bmi+children+smoker+as.factor(region)+smoker*bmi+as.numeric(smoker)*age, data=ds0)
summary(fit2)
BIC(fit2)
# which increase the performance of the model significantly

# Try No.3: since we only have two continuous covariates, we can try to give them an extra power.
# add bmi^1.5
fit3 <-lm(logcharges ~age+sex+bmi+children+smoker+as.factor(region)+
            smoker*bmi,as.numeric(smoker)*age+bmi^1.5, data=ds0)
summary(fit3)
BIC(fit3)

# Try No.4: add age^1.5
fit4 <-lm(logcharges ~age+sex+bmi+children+smoker+as.factor(region)+
            smoker*bmi,as.numeric(smoker)*age+bmi^1.5+age^1.5, data=ds0)
summary(fit4)
BIC(fit4)

# Try No.5: use charges instead of logcharges for the response.
fit5 <-lm(charges ~age+sex+bmi+children+smoker+as.factor(region)+
            smoker*bmi,as.numeric(smoker)*age+bmi^1.5+age^1.5, data=ds0)
summary(fit5)
BIC(fit5)

#Now we have added all the possible covariates we can, we can consider drop some of them.
#As we can see from the summary, it seems that the region parameters is not that important.
# so we only keep northeast

#Recode------------------------------------------------------------------------
new_ds0= ds0
new_ds0$region[new_ds0$region=="northeast"]="1"
new_ds0$region[new_ds0$region !="1"]="0"
names(new_ds0)

fit6 <- lm(charges ~age+sex+bmi+children+smoker+smoker*bmi+
             region+as.numeric(smoker)*age+I(bmi^1.5)+I(age^1.5), data=new_ds0)
summary(fit6)
AIC(fit6)
BIC(fit6)

# The AIC and BIC actually get smaller.

# Similarly smoker*age is not that important either.
fit7 <- lm(charges ~age+sex+bmi+children+smoker+smoker*bmi+region+I(bmi^1.5)+I(age^1.5), data=new_ds0)
summary(fit7)
AIC(fit7)
BIC(fit7)

# After dropping smoker*age, AIC and BIC values stay the same; however, the goodness
# of fit increase.

# For now, the best model would be
# charges = constant + age + sex + bmi + children + smoker + region + bmi*smoker + bmi^1.5 + age^1.5
# caution: as you may see, the coefficient of smoker is negative.
# Please don't take the result as smoking is good for your health. Since smoke is also used in smoke*bmi, bmi is large.
# So the bad influence of smoking has been transferred to the smoke*bmi term.


