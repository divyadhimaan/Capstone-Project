import numpy as np
from os import listdir
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
from sklearn import svm
import imagehash
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import linear_model
import preproc
import features
import pickle

def svm_algo():
    genuine_image_filenames = listdir("data/genuine")
    forged_image_filenames = listdir("data/forged")
    genuine_image_paths = "data/genuine"
    forged_image_paths = "data/forged"
    image_test_paths = "static/LineSweep_Results"
    image_test = listdir(image_test_paths)

    # Check if test images exist. If not, return a default message.
    if len(image_test) == 0:
        print("No test images found in folder:", image_test_paths)
        return "No test images"

    image_features = []
    genuine_image_features = [[] for _ in range(29)]
    forged_image_features = [[] for _ in range(29)]

    countSignaturePerUser = 0
    # Group genuine signatures by user ID
    for name in genuine_image_filenames:
        signature_id = int(name.split('_')[0][-3:])
        genuine_image_features[signature_id - 1].append({"name": name})
    # Count genuine signatures per user (from first group)
    for group in genuine_image_features:
        for _ in group:
            countSignaturePerUser += 1
        break

    for name in forged_image_filenames:
        signature_id = int(name.split('_')[0][-3:])
        forged_image_features[signature_id - 1].append({"name": name})

    for name in image_test:
        signature_id = int(name.split('_')[0][-3:])
        image_features.append({"name": name})

    def preprocess_image(path, display=False):
        return preproc.preproc(path, display=display)

    des_list = []
    def sift(im, path, display=False):
        raw_image = cv2.imread(path)
        try:
            sift_detector = cv2.xfeatures2d.SIFT_create()
        except AttributeError:
            sift_detector = cv2.SIFT_create()
        kp, des = sift_detector.detectAndCompute(im, None)
        if display:
            cv2.drawKeypoints(im, kp, raw_image)
            cv2.imshow('sift_keypoints.jpg', cv2.resize(raw_image, (0, 0), fx=3, fy=3))
            cv2.waitKey()
        return (path, des)

    cor_gen = 0
    wrong_gen = 0
    cor_for = 0
    wrong_for = 0

    im_contour_features = []
    im_contour_features_test = []
    print("processing svm model...")

    for i in range(29):
        des_list = []
        # Process genuine images for group i
        for im in genuine_image_features[i]:
            image_path = genuine_image_paths + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            phash = imagehash.phash(Image.open(image_path))
            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                features.get_contour_features(preprocessed_image.copy(), display=False)
            phash_int = int(str(phash), 16)
            im['hash'] = phash_int
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area
            im['ratio'] = features.Ratio(preprocessed_image.copy())
            im['centroid_0'], im['centroid_1'] = features.Centroid(preprocessed_image.copy())
            im['eccentricity'], im['solidity'] = features.EccentricitySolidity(preprocessed_image.copy())
            (im['skewness_0'], im['skewness_1']), (im['kurtosis_0'], im['kurtosis_1']) = \
                features.SkewKurtosis(preprocessed_image.copy())
            im_contour_features.append(
                [aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area,
                 im['ratio'], im['centroid_0'], im['centroid_1'], im['eccentricity'], im['solidity'],
                 im['skewness_0'], im['skewness_1'], im['kurtosis_0'], im['kurtosis_1']])
            des_list.append(sift(preprocessed_image.copy(), image_path))

        # Process forged images for group i
        for im in forged_image_features[i]:
            image_path = forged_image_paths + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            phash = imagehash.phash(Image.open(image_path))
            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                features.get_contour_features(preprocessed_image.copy(), display=False)
            phash_int = int(str(phash), 16)
            im['hash'] = phash_int
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area
            im['ratio'] = features.Ratio(preprocessed_image.copy())
            im['centroid_0'], im['centroid_1'] = features.Centroid(preprocessed_image.copy())
            im['eccentricity'], im['solidity'] = features.EccentricitySolidity(preprocessed_image.copy())
            (im['skewness_0'], im['skewness_1']), (im['kurtosis_0'], im['kurtosis_1']) = \
                features.SkewKurtosis(preprocessed_image.copy())
            im_contour_features.append(
                [aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area,
                 im['ratio'], im['centroid_0'], im['centroid_1'], im['eccentricity'], im['solidity'],
                 im['skewness_0'], im['skewness_1'], im['kurtosis_0'], im['kurtosis_1']])
            des_list.append(sift(preprocessed_image.copy(), image_path))

        # Process test images (common for all groups)
        for im in image_features:
            image_path = image_test_paths + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            phash = imagehash.phash(Image.open(image_path))
            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                features.get_contour_features(preprocessed_image.copy(), display=False)
            phash_int = int(str(phash), 16)
            im['hash'] = phash_int
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area
            im['ratio'] = features.Ratio(preprocessed_image.copy())
            im['centroid_0'], im['centroid_1'] = features.Centroid(preprocessed_image.copy())
            im['eccentricity'], im['solidity'] = features.EccentricitySolidity(preprocessed_image.copy())
            (im['skewness_0'], im['skewness_1']), (im['kurtosis_0'], im['kurtosis_1']) = \
                features.SkewKurtosis(preprocessed_image.copy())
            im_contour_features_test.append(
                [aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area,
                 im['ratio'], im['centroid_0'], im['centroid_1'], im['eccentricity'], im['solidity'],
                 im['skewness_0'], im['skewness_1'], im['kurtosis_0'], im['kurtosis_1']])
            des_list.append(sift(preprocessed_image.copy(), image_path))

        # Combine all SIFT descriptors for this group
        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))

        # Use fixed desired_k to ensure constant feature vector length
        desired_k = 500
        effective_k = desired_k
        if descriptors.shape[0] < desired_k:
            effective_k = descriptors.shape[0]
        voc, variance = kmeans(descriptors, effective_k, 1)

        # Build histogram features for training images
        n_train = len(genuine_image_features[i]) + len(forged_image_features[i])
        if n_train < 10:
            print("Skipping group", i, "due to insufficient training samples:", n_train)
            continue
        im_features = np.zeros((n_train, desired_k + 12), "float32")
        for ii in range(n_train):
            words, distance = vq(des_list[ii][1], voc)
            for w in words:
                im_features[ii][w] += 1
            for j in range(12):
                im_features[ii][desired_k + j] = im_contour_features[ii][j]

        # Build histogram features for test images
        n_test = len(image_features)
        if n_test == 0:
            print("No test images found for group", i)
            continue
        im_features_test = np.zeros((n_test, desired_k + 12), "float32")
        # Test descriptors follow training descriptors in des_list
        for ii in range(n_test):
            words, distance = vq(des_list[n_train + ii][1], voc)
            for w in words:
                im_features_test[ii][w] += 1
            for j in range(12):
                im_features_test[ii][desired_k + j] = im_contour_features_test[ii][j]

        # Scale the features
        stdSlr = StandardScaler().fit(im_features)
        im_features = stdSlr.transform(im_features)
        stdSlrTest = StandardScaler().fit(im_features_test)
        im_features_test = stdSlrTest.transform(im_features_test)

        # Split training features into genuine and forged sets (expecting at least 10 samples)
        train_genuine_features, test_genuine_features = im_features[0:3], im_features[3:5]
        train_forged_features, test_forged_features = im_features[5:8], im_features[8:10]

        # Train a LinearSVC model (or load a pre-trained one)
        clf = LinearSVC()
        clf.fit(np.concatenate((train_forged_features, train_genuine_features)),
                np.array([1] * len(train_forged_features) + [2] * len(train_genuine_features)))
        # Predict on the test split
        genuine_res = clf.predict(test_genuine_features)
        forged_res = clf.predict(test_forged_features)

        for res in genuine_res:
            if int(res) == 2:
                cor_gen += 1
            else:
                wrong_gen += 1

        for res in forged_res:
            if int(res) == 1:
                cor_for += 1
            else:
                wrong_for += 1

    res_forged = float(cor_for / (cor_for + wrong_for)) if (cor_for + wrong_for) > 0 else 0
    res_genuine = float(cor_gen / (cor_gen + wrong_gen)) if (cor_gen + wrong_gen) > 0 else 0

    print("res_forged: " + str(res_forged))
    print("res_genuine: " + str(res_genuine))

    result = "Genuine" if res_genuine > res_forged else "Forged"

    print("Processing Complete.")
    return result
