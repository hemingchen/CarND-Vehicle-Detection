import os

###############################################################################
# Training sets
###############################################################################
train_img_root = "test_images"

veh_train_img_dir_name = "vehicles"
veh_train_img_set_gti_far = "GTI_Far"
veh_train_img_set_gti_left = "GTI_Left"
veh_train_img_set_gti_middle_close = "GTI_MiddleClose"
veh_train_img_set_gti_right = "GTI_Right"
veh_train_img_set_kitti_extracted = "KITTI_extracted"
selected_veh_train_img_sets = [
    veh_train_img_set_gti_far,
    veh_train_img_set_gti_left,
    veh_train_img_set_gti_middle_close,
    veh_train_img_set_gti_right,
    veh_train_img_set_kitti_extracted]
selected_veh_train_img_set_paths = map(
    lambda x: os.path.join(train_img_root, veh_train_img_dir_name, x),
    selected_veh_train_img_sets)

non_veh_train_img_dir_name = "non-vehicles"
non_veh_train_img_set_extras = "Extras"
non_veh_train_img_set_gti = "GTI"
selected_non_veh_train_img_sets = [
    non_veh_train_img_set_extras,
    non_veh_train_img_set_gti]
selected_non_veh_train_img_set_paths = map(
    lambda x: os.path.join(train_img_root, non_veh_train_img_dir_name, x),
    selected_non_veh_train_img_sets)

sub_training_set_size = None  # None if use entire training data

###############################################################################
# Training parameters
###############################################################################
train_img_size = (64, 64)
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
use_spatial_features = True  # Spatial features on or off
use_hist_features = True  # Histogram features on or off
use_hog_features = True  # HOG features on or off

###############################################################################
# Sliding window parameters
###############################################################################
x_start_stop = [None, None]  # Min and max in x to search in sliding window
y_start_stop = [None, None]  # Min and max in y to search in sliding window
sliding_window_size = (96, 96)
sliding_window_overlap = (0.5, 0.5)

###############################################################################
# Image test parameters
###############################################################################
y_crop_start = 400
y_crop_stop = 656
scale = 1.5
# Instead of overlap, define # of cells to slide when extracting hog features for the patch while hog features of all
#  cells are computed in advance
cells_per_step = 2
# Bounding box color and thickness
bbox_color = (0, 0, 255)
bbox_thickness = 6
