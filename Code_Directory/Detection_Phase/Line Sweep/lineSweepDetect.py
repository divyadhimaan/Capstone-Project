from PIL import Image

import os
import numpy as np
import cv2


def main():
    images_dir = "../OCR/OCR_Results"
    # images_dir = "image"
    input_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), images_dir
    )
    total_files = 0
    processed_files = 0

    for filename in os.listdir(input_path):
        fileSize = os.stat(os.path.join(input_path, filename)).st_size
        total_files = total_files + 1
        if fileSize != 0:
            processed_files = processed_files + 1
            print("Processing " + filename)
            img = Image.open(os.path.join(input_path,filename))
            temp = np.array(img)

            grayscale = img.convert("L")
            xtra, thresh = cv2.threshold(
                np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV
            )

            # cv2.imshow('Binary Threshold', thresh)
            #
            # cv2.waitKey();

            # thresh = cv2.medianBlur(thresh, 2)

            rows = thresh.shape[0]
            cols = thresh.shape[1]

            flagx = 0
            indexStartX = 0
            indexEndX = 0

            for i in range(rows):
                line = thresh[i, :]

                if flagx == 0:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexStartX = i
                        flagx = 1
                        # print('start x: ', indexStartX, flagx)

                elif flagx == 1:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexEndX = i
                        # print('end x: ', indexEndX)
                    elif indexStartX + 5 > indexEndX:
                        indexStartX = 0
                        flagx = 0
                        # print('elif x: ', indexStartX, flagx)
                    else:
                        break

            flagy = 0
            indexStartY = 0
            indexEndY = 0

            for j in range(cols):
                line = thresh[indexStartX:indexEndX, j : j + 20]

                if flagy == 0:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexStartY = j
                        flagy = 1
                        # print('start y: ', indexStartY, flagy)

                elif flagy == 1:
                    ele = [255]
                    mask = np.isin(ele, line)

                    if True in mask:
                        indexEndY = j
                        # print('end y: ', indexEndY)
                    elif indexStartY + 20 > indexEndY:
                        indexStartY = 0
                        flagy = 0
                        # print('elif y: ', indexStartY, flagy)
                    else:
                        break

            # print(indexStartX, indexEndX, indexStartY, indexEndY)
            cv2.line(
                thresh,
                (indexStartY, indexStartX),
                (indexEndY, indexStartX),
                (255, 0, 0),
                1,
            )
            cv2.line(
                thresh,
                (indexStartY, indexEndX),
                (indexEndY, indexEndX),
                (255, 0, 0),
                1,
            )

            cv2.line(
                thresh,
                (indexStartY, indexStartX),
                (indexStartY, indexEndX),
                (255, 0, 0),
                1,
            )
            cv2.line(
                thresh,
                (indexEndY, indexStartX),
                (indexEndY, indexEndX),
                (255, 0, 0),
                1,
            )
            temp_np = temp[
                indexStartX : indexEndX + 1, indexStartY : indexEndY + 1
            ]

            # cv2.imshow('New cropped image', temp_np)
            #
            # cv2.waitKey();

            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LineSweep_Results")
            if not os.path.exists(path):
                os.makedirs(path)

            s1 = "LineSweep_Result_" + filename
            cv2.imwrite(os.path.join(path, s1), temp_np)

    print(str(processed_files) + "/" + str(total_files) + " files processed successfully")
    print("Processing Complete.")
    print("You may check the Result folder in the same directory to see the cropped Project_Images.")


if __name__ == "__main__":
    main()
