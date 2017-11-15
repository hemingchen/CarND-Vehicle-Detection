import functools
import glob
import pickle

import matplotlib
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from helper_funcs import *
from param_defs import *


# Define a single function that can extract features using hog sub-sampling and make predictions
def vehicle_detection_pipeline(
        img,
        train_img_size,
        color_space,
        y_crop_start,
        y_crop_stop,
        scales,
        svc,
        X_scaler,
        orient,
        pix_per_cell,
        cell_per_block,
        cells_per_step,
        spatial_size,
        hist_bins,
        hog_channel,
        use_spatial_features,
        use_hist_features,
        use_hog_features,
        bbox_color,
        bbox_thickness,
        heatmap_history,
        min_heat,
        video_mode):
    # Image with individual block prediction result boxes
    img_with_raw_boxes = np.copy(img)

    # Image with final prediction result boxes, based on heat map
    img_with_heat_boxes = np.copy(img)

    # 1) Scale jpg images from 0~255 range to 0~1 range, since the training data was png format in 0~1 range
    img = img.astype(np.float32) / 255

    # 2) Crop out the lower part of the image since the upper part is not useful
    cropped_img = img[y_crop_start:y_crop_stop, :, :]

    # 3) Apply color conversion if other than 'RGB'
    color_converted_img = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            color_converted_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            color_converted_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            color_converted_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            color_converted_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            color_converted_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2YCrCb)
    else:
        color_converted_img = np.copy(cropped_img)

    # 4) Scaling/resize of the image - use multiple scales to better detect vehicles far/close to own vehicle
    boxes_with_vehicles = []
    for scale in scales:
        if not video_mode:
            print("detecting vehicle with scale {}".format(scale))
        if scale != 1:
            imshape = color_converted_img.shape
            rescaled_img = cv2.resize(color_converted_img, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        else:
            rescaled_img = np.copy(color_converted_img)

        # 5) Define blocks along entire image to be used for hog feature extraction
        nxblocks = (rescaled_img.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (rescaled_img.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = pix_per_cell ** 2
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        # print("nblocks_per_window: {}".format(nblocks_per_window))
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # 6) Compute individual channel HOG features for the entire image (after cropping, scaling, etc.)
        entire_img_hog_features = []
        for channel in range(rescaled_img.shape[2]):
            entire_img_hog_features.append(
                get_hog_features(rescaled_img[:, :, channel], orient, pix_per_cell, cell_per_block, feature_vec=False))

        # 7) Extract patches from image and test if vehicle exists in any patch/block
        for xb in range(nxsteps):
            for yb in range(nysteps):
                # 8) Extract one image patch and test if vehicle exists in such patch/block
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell
                # Resize subimg to training img size before starting vehicle detection
                subimg = cv2.resize(rescaled_img[ytop:ytop + window, xleft:xleft + window], train_img_size)
                img_block_features = []
                # print("subimg size:{}".format(subimg.shape))

                # 9) Compute spatial features if flag is set
                if use_spatial_features:
                    spatial_features = get_bin_spatial_features(subimg, size=spatial_size)
                    # 4) Append features to list
                    img_block_features.append(spatial_features)
                # print("spatial_features size:{}".format(len(spatial_features)))

                # 10) Compute histogram features if flag is set
                if use_hist_features:
                    hist_features = get_color_hist_features(subimg, nbins=hist_bins)
                    # 6) Append features to list
                    img_block_features.append(hist_features)
                # print("hist_features size:{}".format(len(hist_features)))

                # 11) Compute HOG features if flag is set
                if use_hog_features:
                    if hog_channel == 'ALL':
                        # Extract HOG for this patch
                        hog_features = []
                        for channel in range(subimg.shape[2]):
                            channel_hog_features = \
                                entire_img_hog_features[channel][ypos:ypos + nblocks_per_window,
                                xpos:xpos + nblocks_per_window].ravel()
                            hog_features.extend(channel_hog_features)
                    else:
                        channel_hog_features = \
                            entire_img_hog_features[hog_channel][ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                        hog_features = channel_hog_features
                    img_block_features.append(hog_features)
                # print("hog_features size:{}".format(len(hog_features)))

                # 12) Reshape feature vector
                img_block_features = np.concatenate(img_block_features).reshape(1, -1)
                # print(img_block_features.shape)

                # 13) Normalize feature vector
                final_img_block_features = X_scaler.transform(img_block_features)

                # 14) Predict if vehicle exists in this block/patch
                img_block_pred_result = svc.predict(final_img_block_features)

                # 15) Draw raw box if vehicle exists
                if img_block_pred_result == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    box = (
                        (xbox_left, ytop_draw + y_crop_start),
                        (xbox_left + win_draw, ytop_draw + win_draw + y_crop_start))
                    boxes_with_vehicles.append(box)
                    cv2.rectangle(img_with_raw_boxes, box[0], box[1], color=bbox_color, thickness=bbox_thickness)

    # Generate and draw heat map
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, boxes_with_vehicles)

    # In video mode, store history of heatmap to help filter out noises
    if video_mode:
        heatmap_history.append(heat)
        hist_heat_avrg = sum(heatmap_history) / heatmap_history_len
        heat = hist_heat_avrg

        # Start filtering when the history stack is full, otherwise heat is averaged out in the first few frames
        if len(heatmap_history) == heatmap_history_len:
            heat[heat < min_heat] = 0

    # Remove abnormal values
    img_heatmap = np.clip(heat, 0, 255)

    # Draw heat map boxes
    labels = label(img_heatmap)
    img_with_heat_boxes = draw_labeled_bboxes(img_with_heat_boxes, labels, color=bbox_color, thickness=bbox_thickness)

    if video_mode:
        # Debug img 1: raw boxes
        debug_img_1 = img_with_raw_boxes

        # Debug img 2: heatmap, convert 1 channel heatmap to RGB color map with matplotlib
        hot_cmap = matplotlib.cm.get_cmap('hot')
        debug_img_2 = (img_heatmap - np.min(img_heatmap)) / (np.max(img_heatmap) - np.min(img_heatmap))
        debug_img_2 = hot_cmap(debug_img_2).astype(np.float) * 255  # change scale
        debug_img_2 = debug_img_2[:, :, 0:3]  # drop the 'a' channel in 'rgba' format

        final_output_img = add_debug_imgs(img_with_heat_boxes, debug_img_1, debug_img_2)

        return final_output_img
    else:
        return img_with_raw_boxes, img_heatmap, img_with_heat_boxes


def test_pipeline_on_multiple_images(input_path, trained_model, X_scaler):
    raw_images = []
    for img_f in glob.glob(input_path + "/*.jpg"):
        ###############################################################################
        # Search for vehicles in all sliding windows
        ###############################################################################
        print("processing {}...".format(os.path.basename(img_f)))
        img_with_raw_boxes, img_heatmap, img_with_heat_boxes = vehicle_detection_pipeline(
            img=mpimg.imread(img_f),
            train_img_size=train_img_size,
            color_space=color_space,
            y_crop_start=y_crop_start,
            y_crop_stop=y_crop_stop,
            scales=scales,
            svc=trained_model,
            X_scaler=X_scaler,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            cells_per_step=cells_per_step,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            hog_channel=hog_channel,
            use_spatial_features=use_spatial_features,
            use_hist_features=use_hist_features,
            use_hog_features=use_hog_features,
            bbox_color=bbox_color,
            bbox_thickness=bbox_thickness,
            heatmap_history=heatmap_history,
            min_heat=min_heat,
            video_mode=False)
        print("")
        ###############################################################################
        # Draw boxes on detected vehicles
        ###############################################################################
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(img_with_raw_boxes)
        plt.title('Raw Prediction Results')

        plt.subplot(132)
        plt.imshow(img_heatmap, cmap='hot')
        plt.title('Heat Map')

        plt.subplot(133)
        plt.imshow(img_with_heat_boxes)
        plt.title('Predicted Vehicle Positions')

        fig.tight_layout()

        plt.savefig("examples/vehicle_detection_test_result_{}".format(os.path.basename(img_f)))

    plt.show()


def test_pipeline_on_video(input_path, trained_model, X_scaler, output_path, subclip_range=None):
    partial_vehicle_detection_pipeline = functools.partial(
        vehicle_detection_pipeline,
        train_img_size=train_img_size,
        color_space=color_space,
        y_crop_start=y_crop_start,
        y_crop_stop=y_crop_stop,
        scales=scales,
        svc=trained_model,
        X_scaler=X_scaler,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        cells_per_step=cells_per_step,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        hog_channel=hog_channel,
        use_spatial_features=use_spatial_features,
        use_hist_features=use_hist_features,
        use_hog_features=use_hog_features,
        bbox_color=bbox_color,
        bbox_thickness=bbox_thickness,
        heatmap_history=heatmap_history,
        min_heat=min_heat,
        video_mode=True)

    input_clip = VideoFileClip(input_path)
    if subclip_range is not None:
        input_clip = input_clip.subclip(subclip_range[0], subclip_range[1])

    output_clip = input_clip.fl_image(partial_vehicle_detection_pipeline)
    output_clip.write_videofile(output_path, audio=False)


if __name__ == "__main__":
    ###############################################################################
    # Load trained model and X_scaler
    ###############################################################################
    with open("training_data.p", mode='rb') as f:
        training_data = pickle.load(f)
    trained_model = training_data["trained_model"]
    X_scaler = training_data["X_scaler"]

    ###############################################################################
    # 1) Test trained model on multiple images
    ###############################################################################
    input_path = "test_images"
    test_pipeline_on_multiple_images(
        input_path=input_path,
        trained_model=trained_model,
        X_scaler=X_scaler)

    ###############################################################################
    # 2) Test trained model on video
    ###############################################################################
    input_path = "test_videos/project_video.mp4"
    output_path = "output_videos/project_video_output.mp4"
    subclip_range = None  # (5, 10)  # (35, 35.5)  # (35, 45)  # (0, 5) # (41, 42)
    test_pipeline_on_video(
        input_path=input_path,
        trained_model=trained_model,
        X_scaler=X_scaler,
        output_path=output_path,
        subclip_range=subclip_range)
