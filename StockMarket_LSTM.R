require(keras)
require(caret)
require(kableExtra)

library(keras)
library(caret)
library(kableExtra)

dataset <- read.csv2('TSLA.csv', header = T, sep = ',')

dataset$Open <- as.numeric(dataset$Open)
dataset$High <- as.numeric(dataset$High)
dataset$Low <- as.numeric(dataset$Low)
dataset$Close <- as.numeric(dataset$Close)
dataset$Adj.Close <- as.numeric(dataset$Adj.Close)
dataset$Volume <- as.numeric(dataset$Volume)

#MODELLING

days_to_forecast = 60
n = days_to_forecast + 1
X_train = as.matrix(dataset[1:(nrow(dataset) - (n - 1)), -c(1, 5, 6, 7)])
y_train = as.matrix(dataset[n:nrow(dataset), 6])
X_test = as.matrix(dataset[((nrow(dataset) - (n - 2)):nrow(dataset)), -c(1, 5, 6, 7)])
y_test = as.matrix(dataset[((nrow(dataset) - (n - 2)):nrow(dataset)), 6])

ker = ncol(X_train)

keras_model <- keras_model_sequential() 

keras_model %>% 
  layer_dense(units = 60, activation = 'relu', input_shape = ker) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'linear')

summary(keras_model)

keras_model %>% compile(optimizer = "rmsprop", loss = "mse", metrics = "mse")


keras_history <- keras_model %>% fit(X_train, y_train, epochs = 500, batch_size = 32, 
                                     validation_split = 0.1, callbacks = callback_tensorboard("logs/run_a"))

#Prediction

keras_pred <- keras_model %>% predict(X_test, batch_size = 28)

real_VS_pred <- data.frame(keras_pred, y_test)
real_VS_pred$Error <- real_VS_pred$keras_pred/real_VS_pred$y_test
colnames(real_VS_pred) <- c("Predict", "Real", "Ratio - accuracy (0.8< x < 1)")

kable(real_VS_pred) %>% kable_styling(bootstrap_options = "bordered", full_width = F, 
                                      position = "center") %>% column_spec(1, bold = T, color = "red")

par(mfrow=c(1,2))
plot(y_test, type = "l", col= "green")
plot(keras_pred, type = "l", col="red")
