from helper_funcs import *
from param_defs import *

raw_veh_img = mpimg.imread("test_images/vehicles/GTI_Left/image0009.png")
raw_non_veh_img = mpimg.imread("test_images/non-vehicles/GTI/image6.png")

color_converted_veh_img = cv2.cvtColor(raw_veh_img, cv2.COLOR_RGB2YCrCb)
color_converted_non_veh_img = cv2.cvtColor(raw_non_veh_img, cv2.COLOR_RGB2YCrCb)

###############################################################################
# 0) Visualize vehicle and non-vehicle data
###############################################################################
visualize_image_transformation(
    before_img=raw_veh_img, before_img_title='vehicle image',
    after_img=raw_non_veh_img, after_img_title='non vehicle image')

###############################################################################
# 1) Test HOG feature extraction on vehicle images
###############################################################################
channel = 0
features_0, hog_image_0 = get_hog_features(
    color_converted_veh_img[:, :, channel],
    orient, pix_per_cell, cell_per_block,
    vis=True, feature_vec=True)

visualize_image_transformation(
    before_img=color_converted_veh_img, before_img_title='vehicle image',
    after_img=hog_image_0, after_img_title='hog feature channel {}'.format(channel))

channel = 1
features_1, hog_image_1 = get_hog_features(
    color_converted_veh_img[:, :, channel],
    orient, pix_per_cell, cell_per_block,
    vis=True, feature_vec=True)

visualize_image_transformation(
    before_img=color_converted_veh_img, before_img_title='vehicle image',
    after_img=hog_image_1, after_img_title='hog feature channel {}'.format(channel))

channel = 2
features_2, hog_image_2 = get_hog_features(
    color_converted_veh_img[:, :, channel],
    orient, pix_per_cell, cell_per_block,
    vis=True, feature_vec=True)

visualize_image_transformation(
    before_img=color_converted_veh_img, before_img_title='vehicle image',
    after_img=hog_image_2, after_img_title='hog feature channel {}'.format(channel))

###############################################################################
# 2) Test HOG feature extraction on non-vehicle images
###############################################################################
channel = 0
features_0, hog_image_0 = get_hog_features(
    color_converted_non_veh_img[:, :, channel],
    orient, pix_per_cell, cell_per_block,
    vis=True, feature_vec=True)

visualize_image_transformation(
    before_img=color_converted_non_veh_img, before_img_title='non vehicle image',
    after_img=hog_image_0, after_img_title='hog feature channel {}'.format(channel))

channel = 1
features_1, hog_image_1 = get_hog_features(
    color_converted_non_veh_img[:, :, channel],
    orient, pix_per_cell, cell_per_block,
    vis=True, feature_vec=True)

visualize_image_transformation(
    before_img=color_converted_non_veh_img, before_img_title='non vehicle image',
    after_img=hog_image_1, after_img_title='hog feature channel {}'.format(channel))

channel = 2
features_2, hog_image_2 = get_hog_features(
    color_converted_non_veh_img[:, :, channel],
    orient, pix_per_cell, cell_per_block,
    vis=True, feature_vec=True)

visualize_image_transformation(
    before_img=color_converted_non_veh_img, before_img_title='non vehicle image',
    after_img=hog_image_2, after_img_title='hog feature channel {}'.format(channel))