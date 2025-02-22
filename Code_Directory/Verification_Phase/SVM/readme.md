# SVM - Support Vector Machine

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes. 

The key concepts of SVM include:
- Hyperplane: A decision boundary that separates different classes in the feature space.
- Support Vectors: Data points that are closest to the hyperplane and influence its position and orientation.
- Margin: The distance between the hyperplane and the nearest support vectors. SVM aims to maximize this margin.

## Dataset

We have obtained two sets of data:
1. Genuine Signatures
2. Forged Signatures

This dataset is used to train our model. 

We have collected data from 29 users. Each user has 5 genuine signatures and 5 forged signatures, resulting in 10 signatures per user.  

In total, we have 290 images, comprising 145 genuine signatures and 145 forged signatures.


## Preprocessing

We have already pre-processed the images in the Signature Detection phase.
But before using them in the **Signature Verification Phase**, we need more processing.

## User-Defined Functions

1. rgbgrey(image)
    To convert rgb image to grayscale
2. greybin(image)
    To convert grayscale image to binary
3. preproc(path_of_image, image, display)
    Main function that first checks for valid image then calls rgbgrey and then greybin function
    Next we will try to crop the image further if possible and return the image.


## Features File

This file is included to find the features of the image (contour features).
Features include Ratio, centroid, Eccentricity, Solidity, SkewKurtosis, 
contour features.

## Tools: sklearn, OpenCV, PIL, imagehash, scipy.cluster

3. **imagehash**: Image hashes tell whether two images look nearly identical. 
   This is different from cryptographic hashing algorithms (like MD5, SHA-1) 
   where tiny changes in the image give completely different hashes. 
   In image fingerprinting, we actually want our similar inputs to have similar 
   output hashes as well.
    
    The image hash algorithms (average, perceptual, difference, wavelet) analyse 
   the image structure on luminance (without color information). The color hash 
   algorithm analyses the color distribution and black & gray fractions 
   (without position information).

## Steps

1. We will load all the images from genuine and forged folder.
2. Now, for all the images in genuine folder and forged folder:
   - For each image we will preprocess it. i.e. we will extract the SIFT features of the image. (refer to Preprocessing section above)
   - Next we will find the `phash` using imagehash module
   - We will now find the features of the image using the features files. These features include:
     - Aspect ratio
     - bounding rectangle area
     - convex hull area
     - contour area
   - All the features obtained are added to the features list of the image alongwith a unique hash. Also these features
   are added to **im_contour_features** list.
3. Combine the features from both genuine and forged images into a single dataset.
4. Split the dataset into training and testing sets.
5. Train the SVM classifier using the training set.
6. Evaluate the classifier using the testing set and calculate the accuracy.


## Glossary
1. `phash`: pHash is a simple algorithm that calculates image hash based on the DCT value of the image.