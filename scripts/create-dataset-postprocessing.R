library(reticulate)
np <- import("numpy")

x_train <- np$load("/home/key/code/dlscience-code/data/x_train_pre.npy")
x_val <- np$load("/home/key/code/dlscience-code/data/x_val_pre.npy")
x_test <- np$load("/home/key/code/dlscience-code/data/x_test_pre.npy")
y_train <- np$load("/home/key/code/dlscience-code/data/y_train_pre.npy")
y_val <- np$load("/home/key/code/dlscience-code/data/y_val_pre.npy")
y_test <- np$load("/home/key/code/dlscience-code/data/y_test_pre.npy")

##############################   Original input  ##############################
# forecast reference time (unix timestamp)   --- purely informational
# station id
# air temperature ensemble average           --- also a target variable
# dew-point temperature ensemble average     --- also a target variable
# dew-point depression ensemble average
# surface air pressure ensemble average      --- also a target variable
# relative humidity ensemble average         --- also a target variable
# water vapor mixing ratio ensemble average  --- also a target variable
# forecast lead time                         --- to be moved to pos. 2 in dataset
# cosine component of hour of the day
# sine component of hour of the day
# cosine component of day of the year
# sine component of the day of the year

##############################   Original target ##############################
# forecast reference time (unix timestamp)   --- purely informational
# station id                                 --- not to be included in final target
# forecast lead time                         --- not to be included in final target
# 2m air_temperature
# 2m dew_point_temperature
# 2m surface_air_pressure
# 2m relative_humidity
# 2m water_vapor_mixing_ratio



##############################   Further processing  ##############################
# (1) in input set, move lead time to position 3
# (2) input: standardize all but forecast reference time, station id, forecast lead time
# (3) target: nothing

to_scale <- cbind(x_train[ , 3:8], x_train[ , 10:13])
scaled <- scale(to_scale)
means <- attr(scaled, "scaled:center")
# [1]  7.119372e+00  2.034859e+00  5.084513e+00  8.900014e+02  7.241576e+01  5.550713e+00 -9.869878e-05
# [8] -1.278408e-04  5.835503e-03  8.589854e-03
sds <- attr(scaled, "scaled:scale")
# [1]  9.1460209  7.5133547  3.6748409 75.6375738 15.5615503  2.6142745  0.7071194  0.7070942
# [9]  0.7001365  0.7139335
x_train_proc <- cbind(x_train[ , 1:2], x_train[ , 9], scaled)

x_val_proc <- cbind(x_val[ , 1:2],
                    x_val[ , 9],
                    scale(x_val[ , 3:8], center = means[1:6], scale = sds[1:6]),
                    scale(x_val[ , 10:13], center = means[7:10], scale = sds[7:10])
                    )
x_test_proc <- cbind(x_test[ , 1:2],
               x_test[ , 9],
               scale(x_test[ , 3:8], center = means[1:6], scale = sds[1:6]),
               scale(x_test[ , 10:13], center = means[7:10], scale = sds[7:10])
)

saveRDS(x_train_proc, "/home/key/code/dlscience-code/data/x_train.rds")
saveRDS(x_val_proc, "/home/key/code/dlscience-code/data/x_val.rds")
saveRDS(x_test_proc, "/home/key/code/dlscience-code/data/x_test.rds")
saveRDS(y_train, "/home/key/code/dlscience-code/data/y_train.rds")
saveRDS(y_val, "/home/key/code/dlscience-code/data/y_val.rds")
saveRDS(y_test, "/home/key/code/dlscience-code/data/y_test.rds")
