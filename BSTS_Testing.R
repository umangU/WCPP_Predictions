library(bsts)
library(xts)
library(tsbox)
setwd("C:/Documents/Codes/R")

wcpp <- read.csv('WCPP_Input_4D.csv', header = TRUE)
sample <- wcpp[,c(1,2)]

subset_data_xts <- xts(sample[,-1], as.Date(sample$Date, format = "%d/%m/%y"))
subset_data_ts <- ts_ts(subset_data_xts)

plot(subset_data_ts, xlab = 'Seasonal Dates', ylab = 'World Citations Per Paper')

Y <- window(subset_data_ts, start=c(2020, 1), end=c(2020, 24))
y <- log10(Y)

# Testing on Local Level Model
ll_ss <- list()
ll_ss <- AddLocalLevel(state.specification = ll_ss, y = y)
ll_fit <- bsts(y, state.specification = ll_ss, niter = 2000)
ll_pred <- predict(ll_fit, horizon = 6)
plot(ll_pred, plot.original = 24, main = "Original along with Predictions", xlab = 'Seasonal Dates', ylab = 'World Citations Per Paper')

# Testing on Local Linear Trend Model
llt_ss <- list()
llt_ss <- AddLocalLinearTrend(state.specification = llt_ss, y = y)
llt_fit <- bsts(y, state.specification = llt_ss, niter = 2000)
llt_pred <- predict(llt_fit, horizon = 6)
plot(llt_pred, plot.original = 24, main = "Original along with Predictions", xlab = 'Seasonal Dates', ylab = 'World Citations Per Paper')

# Testing on Local Linear Trend and Seasonal Model
lts_ss <- list()
lts_ss <- AddLocalLinearTrend(state.specification = lts_ss, y = y)
lts_ss <- AddSeasonal(lts_ss, y, nseasons = 6)
lts_fit <- bsts(y, state.specification = lts_ss, niter = 2000)
lts_pred <- predict(lts_fit, horizon = 6)
burn <- SuggestBurn(0.1, lts_fit)
plot(lts_pred, plot.original = 24, main = "Original along with Predictions", xlab = 'Seasonal Dates', ylab = 'World Citations Per Paper')

d2 <- data.frame(
        c(10^as.numeric(-colMeans(lts_fit$one.step.prediction.errors[-(1:burn),])+y),  
        10^as.numeric(lts_pred$mean)))

Actual <- data.frame(sample[,2])
Predicted <- data.frame(d2[1:24,])
MAPE <- mean(as.numeric(unlist(abs(Actual-Predicted)/Actual)))*100

# Mean Absolute Percentage Error
print(paste0("The Mean Absolute Percentage Error is: ",round(MAPE, digits=2)))