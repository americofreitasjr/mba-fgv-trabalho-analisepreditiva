library(ggplot2)
#install.packages("dplyr")
library(dplyr)
#install.packages("glmnet")
library(glmnet)
#install.packages("ISLR")
library(ISLR)
install.packages("caTools")
library(caTools)

#### carrego base ### 
setwd("E:/MBA FGV/Análise Preditiva/Trabalho")
df = read.csv("data_tratada.csv")

df <- df[,-1] #Tirar a coluna X - primeira coluna, que é só um índice

#### Análise Descritiva da variável resposta JobSatisfaction
hist(df$JobSatisfaction)
boxplot(df$JobSatisfaction)

#### Random Forest Classification
install.packages("randomForest")
library(randomForest)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')

set.seed(123)
split = sample.split(df$JobSatisfaction, SplitRatio = 0.75)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

ind_col <- c(2,50,67,68,70,71,73)
df_clean <- df[,-ind_col]

set.seed(123)
classifier = randomForest(x = training_set[,-ind_col],
                          y = training_set$JobSatisfaction,
                          ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set)

# Making the Confusion Matrix
cm = table(test_set[,14], y_pred)
cm_dat = data.frame(cm)

# Visualising the Training set results
set = training_set
#pdf("../../figuras/rf_treino.pdf",width = 7, height = 7)
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
#dev.off()

# Visualising the Test set results
set = test_set
#pdf("../../figuras/rf_teste.pdf",width = 7, height = 7)
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
#dev.off()

# Choosing the number of trees
plot(classifier)

####
#### obtenho a variável dependente (y) e independentes (x)####
y <- df$JobSatisfaction
x <- model.matrix(JobSatisfaction ~ .,df)[,-14]
colnames(df)
colnames(x)
#Split Treino e Teste
sample_size <- floor(0.75 * nrow(df))
ind <- sample(1:nrow(df), size = sample_size, replace = F)
train <- df[ind,]
test <- df[-ind,]

y.treino <- y[ind]
y.val <- y[-ind]
x.treino <- x[ind,]
x.val <- x[-ind,]
#forma alternativa de fazer o split Treino e Test

#split = sample.split(df$Balance, SplitRatio = 0.75)
# train = subset(dataset, split == TRUE)
# test = subset(dataset, split == FALSE)
# x.treino <- model.matrix(Salary~., data=train)
# y.treino <- train$Salary
# 
# x.val <- model.matrix(Salary~., data=test)
# y.val <- test$Salary

x.treino
dim(x.treino)


#######LINEAR MODEL #####
mod = lm(Balance ~ Limit + Student, data=train)
print(mod)
summary(mod)
yhat <- predict(mod, test)

mean((yhat-y.val)^2)
dim(x.treino)
dim(y.treino)

hist(y.treino)
hist(y.val)
hist(yhat)

# Model performance metrics:

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

R2 <- function (x, y) cor(x, y) ^ 2

data.frame(
  RMSE = RMSE(yhat, y.val),
  Rsquare = R2(yhat, y.val)
)

#####LASSO##### (glmnet com alpha =1)

cv.out <- cv.glmnet(x.treino, y.treino, alpha=1)
plot(cv.out)

lasso.mod=glmnet(x=x.treino, y=y.treino, alpha=1, lambda=cv.out$lambda.min)


# vamos olhar os coeficientes deste modelo e erro no conjunto de validação
predict(lasso.mod, s =cv.out$lambda.min , type="coefficients")

yhat <- predict(lasso.mod, s = cv.out$lambda.min, type="response", newx=x.val)
mean((yhat-y.val)^2)

hist(y.treino)
hist(y.val)
hist(yhat)
# Model performance metrics:

data.frame(
  RMSE = RMSE(yhat, y.val),
  Rsquare = R2(yhat, y.val)
)

#### RIDGE ###### (mesmo que o lasso mas com alplha = 0)

cv.out <- cv.glmnet(x.treino, y.treino, alpha=0)
plot(cv.out)

lasso.mod=glmnet(x=x.treino, y=y.treino, alpha=0, lambda=cv.out$lambda.min)


# vamos olhar os coeficientes deste modelo e erro no conjunto de validação
predict(lasso.mod, s =cv.out$lambda.min , type="coefficients")

yhat <- predict(lasso.mod, s = cv.out$lambda.min, type="response", newx=x.val)
mean((yhat-y.val)^2)

hist(y.treino)
hist(y.val)
hist(yhat)

# Model performance metrics:

data.frame(
  RMSE = RMSE(yhat, y.val),
  Rsquare = R2(yhat, y.val)
)