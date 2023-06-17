library(torch)

### Input variables (x) ###
# station id
# forecast lead time
# air temperature ensemble average (standardized; also a target variable)
# dew-point temperature ensemble average (standardized; also a target variable)
# dew-point depression ensemble average (standardized)
# surface air pressure ensemble average (standardized; also a target variable)
# relative humidity ensemble average (standardized; also a target variable)
# water vapor mixing ratio ensemble average (standardized; also a target variable)
# cosine component of hour of the day (standardized)
# sine component of hour of the day (standardized)
# cosine component of day of the year (standardized)
# sine component of the day of the year (standardized)

### Target variables (y) ###
# 2m air_temperature
# 2m dew_point_temperature
# 2m surface_air_pressure
# 2m relative_humidity
# 2m water_vapor_mixing_ratio

### Forecast reference time (forecast_time) (supplementary information) ###

physical_postprocessing_dataset <- torch::dataset(
  "physical_postprocessing_dataset",
  initialize = function(root, split = "train", download = FALSE, ...,
                        transform = NULL, target_transform = NULL) {
    data_path <- "~/code/dlscience-code/data/"
    self$split <- split
    self$xpath <- fs::path(data_path, paste0("x_", self$split, ".rds"))
    self$x <- readRDS(self$xpath)
    self$ypath <- fs::path(data_path, paste0("y_", self$split, ".rds"))
    self$y <- readRDS(self$ypath)

    self$transform <- if (is.null(transform)) identity else transform
    self$target_transform <- if (is.null(target_transform)) identity else target_transform
  },
  .getitem = function(i) {
    # all intended predictors, including station id and lead time (excluded: forecast reference time)
    x <- self$x[i, 2:13]
    # excluded: forecast reference time, station id, lead time
    y <- self$y[i, 4:8]
    forecast_time <- self$x[i, 1]
    list(x = torch::torch_tensor(x), y = torch::torch_tensor(y), forecast_time = forecast_time)
  },
  .length = function() {
    dim(self$x)[1]
  }
)

train_ds <- tbd_dataset()
train_ds[1000]

train_dl <- dataloader(train_ds, batch_size = 1024, shuffle = TRUE)
train_dl


### Physical constraints ###
#
# https://arxiv.org/pdf/2212.04487.pdf
#
# Constraints are formulated on relative humidity and mixing ratio, but really involve
# 2 variables not present in the dataset:
#   water vapor pressure (e), and
#   saturation water vapor pressure (e_s).
#
# The relevant equations are:
#   (5) e = c exp((a * T_d)(b + T_d)) and e_s = c exp((a * T)(b + T)), where T_d is the
#       dew point temperature, and a,b,c and constants (defined as is practice at Meteo Swiss).
#       These formulae directly yield constraint (1), formulated as (6)/(4a, resp.):
#       RH = e/e_s * 100 = exp((a * T_d)(b + T_d)/(a * T)(b + T)) where RH is relative humidity.
#   (7) r = 1000 * (0.622 * e)/(p - e) where r is water vapor mixing ratio. This yields (4b).
#

physics_layer <- nn_module(
  forward = function(x) {
    t <- x[ , 1]
    t_def <- x[ , 2]
    p <- x[ , 3]
    # make sure deficit temperature is non-negative
    t_d <- t - torch_relu(t_def)
    e_s <- torch_where(
      t >= 0,
      6.107 * torch_exp((17.368 * t) / (t + 238.83)),
      6.108 * torch_exp((17.856 * t) / (t + 245.52)),
    )
    e <- torch_where(
      t >= 0.0,
      6.107 * torch_exp((17.368 * t_d) / (t_d + 238.83)),
      6.108 * torch.exp((17.856 * t_d) / (t_d + 245.52)),
    )
    rh <- e / e_s * 100
    r <- 622.0 * (e / (p - e))
    pred <- torch_stack(list(t, t_d, p, rh, r), dim = 2)
    pred
  }
)

model <- nn_module(
  initialize = function(in_size, n_stations, embedding_size, fc1_size, fc2_size, constrained = FALSE) {
    self$out_bias <-  c(15.0, 10.0, 900.0, 70.0, 5.0)  # t, t_d, p, rh, r
    self$out_bias_constrained <- c(15.0, 5.0, 900)  # t, t_def, p
    self$embedding <- nn_embedding(n_stations, embedding_size)
    self$fc1 <- nn_linear(in_size + embedding_size, fc1_size)
    self$fc2 <- nn_linear(fc1_size, fc2_size)
    if (constrained) {
      self$out <- nn_sequential(nn_linear(fc2_size, 3), physics_layer())
      self$out[1]$bias <- nn_parameter(torch_tensor(self$out_bias_constrained))
    } else {
      self$out = nn_linear(fc2_size, 5)
      self$out$bias = nn_parameter(torch_tensor(self$out_bias))
    }
  },
  forward = function(x, station_id) {
    station_embedding <- self$embedding(station_id)
    out <- x |>
      torch_cat(list(x, station_embedding), dim = -1) |> # tbd check dim!!
      torch_relu(self$fc1) |>
      torch_relu(self$fc2) |>
      self$out
    out
  }
)




