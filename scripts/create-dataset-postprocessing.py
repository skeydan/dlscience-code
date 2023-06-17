# Data from:
# https://github.com/ecmwf-projects/mooc-machine-learning-weather-climate/blob/main/tier_2/physic_informed/Physically_constrained_postprocessing.ipynb

path_data = 'https://unils-my.sharepoint.com/:u:/g/personal/tom_beucler_unil_ch/'

# load training data 
x_path = path_data + 'EdAG3RBBgk5Kmvo54RPgT2kBp-NJqqGF6Il-gTmh9DbdeA?download=1'
y_path = path_data + 'EdVQCVKqnb9Bh495opeuRCEBBZFPDdG0g3xSpIFgNGJeJA?download=1'
x_open = pooch.retrieve(x_path,known_hash='c6acaf62051b81dfd3d5a4aa516d545615fd2597c8c38f4db4e571a621201878')
y_open = pooch.retrieve(y_path,known_hash='6265a5f0272e5427c823b95725b8aabbc48a9a97d7554fd5732e6c4b480f3ab3')

# load saved model's weights and biases to optionally accelerate
# the notebook's completion
u_path = path_data + 'ERb1IhuBFfZAjgcAF8N6pikBWet4WBZtheh9zvOyH7QNUg?download=1'
u_open = pooch.retrieve(u_path, known_hash='9dfb659526fa686062057de77bcb2d14ca46fc11212e799e9a5ec7175679b756')
a_path = path_data + 'EQuCPW4q2ilFpFQSZz_UDXYBjR2ZRRcJCDFW-EbI_xWKGg?download=1'
a_open = pooch.retrieve(a_path, known_hash='c6930d908c21785217b20895e19dbf6ddf6cd312b95b331ba590e2267d03e6f4')
l_path = path_data + 'EVb6Gw0ri6JJsazJ6mxUMXUBm4ndfmMjoj0M9gdIPDkd3A?download=1'
l_open = pooch.retrieve(l_path, known_hash = 'ab7e9013631d29806a8f73fbf7275cb37ca3561cb0d0c532107ea00ae496b187')

# build pointers to the data in the notebook using Xarray
x = (
    xr.open_dataset(x_open)
    .set_coords(["forecast_reference_time","t", "station_id"])
    .set_xindex("forecast_reference_time")
    .to_array("var")
    .transpose("s","var")
)
y = (
    xr.open_dataset(y_open)
    .set_coords(["forecast_reference_time","t"])
    .set_xindex("forecast_reference_time")
    .to_array("var")
    .transpose("s","var")
)

# split
# Training: Jan 1, 2017 to Dec 25, 2019
# Validation: Jan 1, 2020 to Dec 25, 2020
# Test: Jan 1, 2021 to Jan 1, 2022
train_sel = dict(forecast_reference_time = slice("2017-01-01", "2019-12-25"))
val_sel = dict(forecast_reference_time = slice("2020-01-01", "2020-12-25"))
test_sel = dict(forecast_reference_time = slice("2021-01-01", "2022-01-01"))
train_x, train_y = x.sel(train_sel), y.sel(train_sel)
val_x, val_y = x.sel(val_sel), y.sel(val_sel)
test_x, test_y = x.sel(test_sel), y.sel(test_sel)
train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape
#((11142117, 11),
# (11142117, 5),
# (3668358, 11),
# (3668358, 5),
# (3802386, 11),
# (3802386, 5))

# create x_train array with station ids and forecast reference times
x_train_np = train_x.to_numpy()
# column 7 contains forecast lead times
# this is the same information as comes in separate array t, so we discard that one
# unique values range between 3 and 120, the time interval being 3
x_train_sid_np = train_x.station_id.to_numpy()
# station ids range between 0 and 130
x_train_ref_np = train_x.forecast_reference_time.to_numpy().astype('datetime64[s]').astype('int')
x_train = np.hstack((x_train_ref_np.reshape((11142117, 1)), x_train_sid_np.reshape(((11142117, 1))), x_train_np))
x_train.shape
# (11142117, 13)

# analogously, for x_val and x_test
x_val_np = val_x.to_numpy()
x_val_sid_np = val_x.station_id.to_numpy()
x_val_ref_np = val_x.forecast_reference_time.to_numpy().astype('datetime64[s]').astype('int')
x_val = np.hstack((x_val_ref_np.reshape((3668358, 1)), x_val_sid_np.reshape(((3668358, 1))), x_val_np))
x_val.shape
# (3668358, 13)

x_test_np = test_x.to_numpy()
x_test_sid_np = test_x.station_id.to_numpy()
x_test_ref_np = test_x.forecast_reference_time.to_numpy().astype('datetime64[s]').astype('int')
x_test = np.hstack((x_test_ref_np.reshape((3802386, 1)), x_test_sid_np.reshape(((3802386, 1))), x_test_np))
x_test.shape
#(3802386, 13)

# create y_train array with forecast reference times, lead times and station ids
y_train_np = train_y.to_numpy()
# station ids are taken from the corresponding input data set (x_train) because they aren't available separately
y_train_sid_np = train_x.station_id.to_numpy()
# for lead times, now need to make use of t array because they are not given in the main target set
y_train_t_np = train_y.t.to_numpy()
y_train_ref_np = train_y.forecast_reference_time.to_numpy().astype('datetime64[s]').astype('int')
y_train = np.hstack((
    y_train_ref_np.reshape((11142117, 1)),y_train_sid_np.reshape((11142117, 1)), y_train_t_np.reshape((11142117, 1)),y_train_np
))
y_train.shape
#(11142117, 8)

# analogously, for y_val and _test
y_val_np = val_y.to_numpy()
y_val_sid_np = val_x.station_id.to_numpy()
y_val_t_np = val_y.t.to_numpy()
y_val_ref_np = val_y.forecast_reference_time.to_numpy().astype('datetime64[s]').astype('int')
y_val = np.hstack((
    y_val_ref_np.reshape((3668358, 1)),y_val_sid_np.reshape((3668358, 1)), y_val_t_np.reshape((3668358, 1)),y_val_np
))
y_val.shape
#(3668358, 8)

y_test_np = test_y.to_numpy()
y_test_sid_np = test_x.station_id.to_numpy()
y_test_t_np = test_y.t.to_numpy()
y_test_ref_np = test_y.forecast_reference_time.to_numpy().astype('datetime64[s]').astype('int')
y_test = np.hstack((
    y_test_ref_np.reshape((3802386, 1)),y_test_sid_np.reshape((3802386, 1)), y_test_t_np.reshape((3802386, 1)),y_test_np
))
y_test.shape
# 3802386, 8)

# save
np.save("/home/key/code/dlscience-code/data/x_train_pre.npy", x_train)
np.save("/home/key/code/dlscience-code/data/x_val_pre.npy", x_val)
np.save("/home/key/code/dlscience-code/data/x_test_pre.npy", x_test)
np.save("/home/key/code/dlscience-code/data/y_train_pre.npy", y_train)
np.save("/home/key/code/dlscience-code/data/y_val_pre_pre.npy", y_val)
np.save("/home/key/code/dlscience-code/data/y_test_pre_pre.npy", y_test)




