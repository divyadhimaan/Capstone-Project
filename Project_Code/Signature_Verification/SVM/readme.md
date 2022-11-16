## Dataset

We have obtained two sets of data:
1. Genuine Signatures
2. Forged Signatures

This dataset is obtained to train our model.

We have collected this data for 29 users.
For each user we have 5 pairs of each genuine and forged signatures, which means we have 8 signatures per user, 
5 genuine and 5 forged.

That is we have in total of 29 * (5 + 5) = 290 images.  145 forged signatures and 145 genuine signatures.

## Preprocessing

We have already obtained the resulting images from our Signature detection phase.

Before using them in the **Signature Verification Phase**, we need to process them.

### User-Defined Functions

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

### Tool: sklearn, OpenCV, PIL, imagehash, scipy.cluster

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
2. Now, for all the images in genuine folder
   1. For each image we will preprocess it. (refer to Preprocessing section above)
   2. Next we will find the phash using imagehash module
      pHash is a simple algorithm that calculates image hash based on the DCT value of the image.
   3. We will now find the features of the image using the features files. (Aspect ratio, bounding rectangle area, 
       convex hull area, contour area)
   4. All the features obtained are added to the features list of the image alongwith a unique hash.
   5. Also these features are added to **im_contour_features** list.
   6. 
