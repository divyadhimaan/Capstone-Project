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

# Load filenames from directories and count using len()
genuine_image_filenames = listdir("data/genuine")
print("Total Number of Files in genuine folder: " + str(len(genuine_image_filenames)))
forged_image_filenames = listdir("data/forged")
print("Total Number of Files in forged folder: " + str(len(forged_image_filenames)))

genuine_image_paths = "data/genuine"
forged_image_paths = "data/forged"
image_test_paths = "data/origin"

image_test = listdir("data/origin")
image_features = []

# Initialize lists for 29 user groups
genuine_image_features = [[] for x in range(29)]
forged_image_features = [[] for x in range(29)]

countSignaturePerUser = 0
# Group genuine signatures by user ID (extracted from filename)
for name in genuine_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    genuine_image_features[signature_id - 1].append({"name": name})

# Count the number of genuine signatures per user (from the first group)
for _ in genuine_image_features:
    for __ in _:
        countSignaturePerUser += 1
    break
print("Total Genuine Signatures per User : " + str(countSignaturePerUser))
print("Total Forged Signatures per User : " + str(countSignaturePerUser))

# Group forged signatures by user ID
for name in forged_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    forged_image_features[signature_id - 1].append({"name": name})

for name in image_test:
    signature_id = int(name.split('_')[0][-3:])
    image_features.append({"name": name})

def preprocess_image(path, display=False):
    # Preprocess the image (e.g., converting to binary image)
    return preproc.preproc(path, display=display)

des_list = []

def sift(im, path, display=False):
    raw_image = cv2.imread(path)
    try:
        # Try using the xfeatures2d version of SIFT
        sift_detector = cv2.xfeatures2d.SIFT_create()
    except AttributeError:
        # Fallback to the default SIFT_create (for newer OpenCV versions)
        sift_detector = cv2.SIFT_create()
    kp, des = sift_detector.detectAndCompute(im, None)

    if display:
        cv2.drawKeypoints(im, kp, raw_image)
        cv2.imshow('sift_keypoints.jpg', cv2.resize(raw_image, (0, 0), fx=3, fy=3))
        cv2.waitKey()

    return (path, des)

cor = 0
wrong = 0

im_contour_features = []

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

    # Combine all SIFT descriptors for this group
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))


    desired_k = 500
    effective_k = desired_k
    if descriptors.shape[0] < desired_k:
        effective_k = descriptors.shape[0]
    voc, variance = kmeans(descriptors, effective_k, 1)

    n_images = len(genuine_image_features[i]) + len(forged_image_features[i])
    im_features = np.zeros((n_images, desired_k + 12), "float32")
    for ii in range(n_images):
        words, distance = vq(des_list[ii][1], voc)
        for w in words:
            im_features[ii][w] += 1

        for j in range(12):
            im_features[ii][desired_k + j] = im_contour_features[ii][j]

    # Scale the feature vectors
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Split into training and testing sets
    train_genuine_features, test_genuine_features = im_features[0:3], im_features[3:5]
    train_forged_features, test_forged_features = im_features[5:8], im_features[8:10]

    # Load the pre-trained SVM model
    model = pickle.load(open('model.pkl', 'rb'))
    genuine_res = model.predict(test_genuine_features)
    forged_res = model.predict(test_forged_features)

    for res in genuine_res:
        if int(res) == 2:
            cor += 1
        else:
            wrong += 1

    # FIX: Count correct forged predictions properly
    for res in forged_res:
        if int(res) == 1:
            cor += 1
        else:
            wrong += 1

print("Final Accuracy SVM: " + str(float(cor) / (cor + wrong)))
