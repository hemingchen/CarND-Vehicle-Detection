import glob
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from helper_funcs import *
from param_defs import *


def print_selected_training_sets():
    print("selected vehicle data training sets:")
    for p in selected_veh_train_img_set_paths:
        print(p)
    print()
    print("selected non-vehicle data training sets:")
    for p in selected_non_veh_train_img_set_paths:
        print(p)


if __name__ == "__main__":
    ###############################################################################
    # Get training data and divide up into cars and notcars
    ###############################################################################
    car_img_paths = []
    for car_p in selected_veh_train_img_set_paths:
        car_img_paths.extend(glob.glob(car_p + "/" + "*.png"))
    print("{} car images in training data".format(len(car_img_paths)))

    non_car_img_paths = []
    for non_car_p in selected_non_veh_train_img_set_paths:
        non_car_img_paths.extend(glob.glob(non_car_p + "/" + "*.png"))
    print("{} non-car images in training data".format(len(non_car_img_paths)))

    ###############################################################################
    # Use a subset of training data for preliminary tests
    ###############################################################################
    if sub_training_set_size is not None:
        car_img_paths = car_img_paths[0:sub_training_set_size]
        non_car_img_paths = non_car_img_paths[0:sub_training_set_size]

    ###############################################################################
    # Feature extraction
    ###############################################################################
    t1 = time.time()
    car_features = extract_features_from_multiple_imgs(
        img_paths=car_img_paths,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        use_spatial_features=use_spatial_features,
        use_hist_features=use_hist_features,
        use_hog_features=use_hog_features)
    non_car_features = extract_features_from_multiple_imgs(
        img_paths=non_car_img_paths,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        use_spatial_features=use_spatial_features,
        use_hist_features=use_hist_features,
        use_hog_features=use_hog_features)

    t2 = time.time()
    print(round(t2 - t1, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    ###############################################################################
    # Training
    ###############################################################################
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t1, 2), 'Seconds to train SVC...')

    ###############################################################################
    # Evaluation
    ###############################################################################
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t1 = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t1, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    ###############################################################################
    # Save trained model
    ###############################################################################
    dist_pickle = {}
    dist_pickle["trained_model"] = svc
    dist_pickle["X_scaler"] = X_scaler
    pickle.dump(dist_pickle, open("training_data.p", "wb"))
    print("training data saved")
