library(bsts)
library(xts)
library(tsbox)
setwd("C:/Documents/Codes/R")

wcpp <- read.csv('WCPP_Subset_4D.csv', header = TRUE)
wcpp_list <- as.list(wcpp)

predictions <- data.frame(matrix(c("2016","2017","2018","2019","2020","2021"), nrow = 6, ncol = 1))
colnames(predictions) <- "Year Range"

for (i in 2:ncol(wcpp))
{
  subset_data <- data.frame(wcpp_list[c(1,i)])
  subset_data_xts <- xts(subset_data[,-1], as.Date(subset_data$Date, format = "%d/%m/%y"))
  subset_data_ts <- ts_ts(subset_data_xts)

  Y <- window(subset_data_ts, start=c(2020, 1), end=c(2020, 24))
  y <- log10(Y)

  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 6)

  bsts.model <- bsts(y, state.specification = ss, niter = 2000, ping=0)
  burn <- SuggestBurn(0.1, bsts.model)

  p <- predict.bsts(bsts.model, horizon = 6, burn = burn, quantiles = c(.025, .975))

  predictions <- cbind(predictions,data.frame(10^p$mean))
  
}

predictions
write.csv(predictions, "C:/Documents/Codes/R/WCPP_ERA2023_Predictions.csv")