library(ggplot2)
#install.packages("dplyr")
library(dplyr)
#install.packages("glmnet")
library(glmnet)
#install.packages("ISLR")
library(ISLR)
#install.packages("caTools")
library(caTools)
#install.packages("randomForest")
library(randomForest)
library(car)
#install.packages("outliers")
library(outliers)

#### carrego base ### 
setwd("D:/MBA Big Data/7. Análise Preditiva/Trabalho")
df = read.csv("data_tratada.csv")

df <- df[,-1] #Tirar a coluna X - primeira coluna, que é só um índice

#### Análise Descritiva da variável resposta JobSatisfaction
hist(df$JobSatisfaction)
# Nota-se que a variável Job Satisfaction possui assimetria à direita, apresnetando maior frequência
# de satisfação de grau 6 ou mais.
boxplot(df$JobSatisfaction)
# Além da assimetria já evidenciada pelo histograma, há presença de outliers na categoria JobSatisfaction = 0.
# Logo, faremos um teste para evidenciar se há presença de dados discrepantes.

# Testando a presença de outliers
outlier(df$JobSatisfaction)

# O teste apontou que a categoria zero de JobSatisfaction é outlier e, sendo assim a retiraremos da base de dados
# a ser modelada.

df<-df[ df$JobSatisfaction != "0", , drop=FALSE]

# novo boxplot e histograma sem outliers em JobSatisf
boxplot(df$JobSatisfaction)
hist(df$JobSatisfaction)

# Como a base possui muitas variáveis, utilizaremos o Modelo de Classificação Random Forest para nos ajudar
# na escolha das melhores variáveis preditoras de JobSatisfacion

#### Random Forest Classification
# Splitting the dataset into the Training set and Test set
set.seed(123)
ind <- sample(1:nrow(df), size = .75*nrow(df), replace = F)
#split = sample.split(df$JobSatisfaction, SplitRatio = 0.75)
training_set = df[ind,]
test_set = df[-ind,]

# Identificando as variáveis com mais de 53 níveis (que a função randomForest não aceita) 
ind_col <- NULL
for (i in (1:ncol(training_set))) {
  ifelse(length(levels(training_set[,i]))<54,ind_col[i]<-TRUE,ind_col[i]<-FALSE)   
}

# base de treino só com variáveis preditoras
training_set_pred <- training_set[,-14]

# Fazendo a floresta aleatória só com as variáveis preditoras que tenham menos de 53 níveis
set.seed(123)
classifier = randomForest(x = training_set_pred[,which(ind_col[-14]==TRUE)],
                          y = training_set$JobSatisfaction,
                          ntree = 100)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set)

# Making the Confusion Matrix
cm = table(test_set[,14], y_pred)
cm_dat = data.frame(cm)
plot(test_set[,14], y_pred)

# Choosing the number of trees
plot(classifier)

# Escolhendo as variáveis mais significativas, tendo como critério diminuição média na impureza dos nós (estatística de Gini)
#install.packages("caret")
library(caret)
varImpPlot(classifier)
varImp
# Tabela com o valor da estatística de Gini em cada variável
importance_dat = data.frame(importance(classifier))
View(importance_dat)

# A partir das variáveis preditoras pré-selecionadas pelo Modelo de Classificação Random Forest, faremos uma
# regressão linear:

#######LINEAR MODEL #####
#Selecionadas (a princípio) as variáveis com índice de Gini > 60
mod = lm(JobSatisfaction ~ CareerSatisfaction + JobSeekingStatus + YearsProgram
         + YearsCodedJob + HoursPerWeek + MajorUndergrad
         #+ Currency 
         + CompanySize
         #+ WorkStart
         + ResumePrompted + CompanyType + HighestEducationParents
         + InfluenceWorkstation + Overpaid + HomeRemote,
         data=training_set)
# O modelo linear apresentou R2 ajustado = 0.4229. O p valor associado indica que podemos rejeitar a hipótese
# nula. Logo, aceitamos de que as variáveis preditoras escolhidas e a Job satisfaction são relacionadas.
print(mod)
summary(mod)

# deu um erro no predict abaixo pois tirei jobsatisf = zero. Alguém tem uma solução ?
yhat <- predict(mod, test_set)
mean((yhat-test_set$JobSatisfaction)^2)

par(mfrow=c(2,2))
hist(training_set$JobSatisfaction, main= "Job Satisfaction - Base de treino", col="lightblue")
hist(test_set$JobSatisfaction, main= "Job Satisfaction - Base de teste", col="lightblue")
hist(yhat, main= "Job Satisfaction - Previsão na base de teste", col="lightblue")
plot(test_set$JobSatisfaction, yhat, main= "Job Satisfaction - Base de teste vs Previsão na base de teste")

# Model performance metrics
#Erro médio residual 
RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

#R2
R2 <- function (x, y) cor(x, y) ^ 2

data.frame(
  RMSE = RMSE(yhat, test_set$JobSatisfaction),
  Rsquare = R2(yhat, test_set$JobSatisfaction)
)
# Aplicamos o modelo linear à base de teste.
# O erro RMSE associado ao modelo linear foi de 1.5167 e o R2 de 0.3693

#####LASSO##### (glmnet com alpha =1)

# Definindo base de teste e treino no formato pedido pelo glmnet
x <- model.matrix(JobSatisfaction ~ .,df)[,-1]
x.treino <- x[ind,]
y.treino <- df$JobSatisfaction[ind]
x.teste <- x[-ind,]
y.teste <- df$JobSatisfaction[-ind]

# Escolha do grau de regularização (lambda)
cv.out <- cv.glmnet(x.treino, y.treino, alpha=1)
plot(cv.out)

# Modelo
lasso.mod=glmnet(x=x.treino, y=y.treino, alpha=1, lambda=cv.out$lambda.min)

# vamos olhar os coeficientes deste modelo e erro no conjunto de validação
predict(lasso.mod, s= cv.out$lambda.min , type="coefficients")

yhat <- predict(lasso.mod, s = cv.out$lambda.min, type="response", newx=x.teste)
mean((yhat-y.teste)^2)

par(mfrow=c(2,2))
hist(y.treino, main= "Job Satisfaction - Base de treino", col="lightblue")
hist(y.teste, main= "Job Satisfaction - Base de teste", col="lightblue")
hist(yhat, main= "Job Satisfaction - Previsão na base de teste", col="lightblue")
plot(y.teste, yhat, main= "Job Satisfaction - Base de teste vs Previsão na base de teste")

# Model performance metrics
data.frame(
  RMSE = RMSE(yhat, y.teste),
  Rsquare = R2(yhat, y.teste)
)

# Utilizando agora o modelo lasso
# O modelo lasso indica um R2 menor do que o do modelo linear e com erro RMSE maior do que o linear.
# Pela análise do coeficiente de determinação e do erro associado, opto pelo modelo linear até aqui.

#### RIDGE ###### (mesmo que o lasso mas com alpha = 0)
# Escolha do grau de regularização (lambda)
cv.out <- cv.glmnet(x.treino, y.treino, alpha=0)
plot(cv.out)

# Modelo
lasso.mod=glmnet(x=x.treino, y=y.treino, alpha=0, lambda=cv.out$lambda.min)

# vamos olhar os coeficientes deste modelo e erro no conjunto de validação
predict(lasso.mod, s= cv.out$lambda.min , type="coefficients")

yhat <- predict(lasso.mod, s = cv.out$lambda.min, type="response", newx=x.teste)
mean((yhat-y.teste)^2)

par(mfrow=c(2,2))
hist(y.treino, main= "Job Satisfaction - Base de treino", col="lightblue")
hist(y.teste, main= "Job Satisfaction - Base de teste", col="lightblue")
hist(yhat, main= "Job Satisfaction - Previsão na base de teste", col="lightblue")
plot(y.teste, yhat, main= "Job Satisfaction - Base de teste vs Previsão na base de teste")

# Model performance metrics
data.frame(
  RMSE = RMSE(yhat, y.teste),
  Rsquare = R2(yhat, y.teste)
)

# O modelo Ridge também indica um R2 menor do que o do modelo linear e com erro RMSE maior do que o linear.
# Pela análise do coeficiente de determinação e do erro associado, optamos pelo modelo linear para prever a variável
# JobSatisfaction.
