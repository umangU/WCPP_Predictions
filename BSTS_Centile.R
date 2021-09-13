library(bsts)
library(xts)
library(tsbox)

wcpp <- read.csv('C:/Documents/Codes/R/World Centile/World_Centile_All.csv', header = TRUE)
wcpp_list <- as.list(wcpp)

predictions <- data.frame(matrix(c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16",
                                   "17","18","19","20","21","22","23","24","25","26","27","28","29","30"), 
                                 nrow = 30, ncol = 1))
colnames(predictions) <- "Centiles"

for (i in 2:ncol(wcpp))
{
  subset_data <- data.frame(wcpp_list[c(1,i)])
  subset_data_xts <- xts(subset_data[,-1], as.Date(subset_data$Date, format = "%d/%m/%Y"))
  subset_data_ts <- ts_ts(subset_data_xts)
  
  Y <- window(subset_data_ts, start=c(2004, 1), end=c(2005, 30))
  y <- log10(Y)
  
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 1)
  
  bsts.model <- bsts(y, state.specification = ss, niter = 2000, ping=0)
  burn <- SuggestBurn(0.1, bsts.model)
  
  p <- predict.bsts(bsts.model, horizon = 30, burn = burn, quantiles = c(.025, .975))
  
  predictions <- cbind(predictions,data.frame(10^p$mean))
  
}

predictions
write.csv(predictions, "C:/Documents/Codes/R/World Centile/World_ERA2023_Centile_Predictions.csv")