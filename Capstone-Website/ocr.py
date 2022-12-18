
try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract
import cv2
import os
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Simple image to string
# print(pytesseract.image_to_string(Image.open('Cheque083654.jpg')))

# # French text image to string
# # print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))




# images_dir = "../../../Our_Dataset/Testing"
images_dir = "static/uploads"
# print("Hey")
# images_dir = "../../../Capstone-Website/static/uploads"
# images_dir = "Data"


def ocr_algo():

    print("processing image through OCR algo...")

    # images_dir = "image"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # print(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(
        dir_path, images_dir
    )
    print(input_path)
    # # Get bounding box estimates
    please = 0
    sign = 0
    above = 0
    total_files = 0
    processed_files = 0
    result = ""
    for filename in os.listdir(input_path):

        total_files = total_files + 1
        if ".jpg" in filename:
            print("OCR Processing file -", filename)
            img = cv2.imread(os.path.join(input_path,filename))
        else:
            continue
        h, w, _ = img.shape  # assumes color image

        # Get verbose data including boxes, confidences, line, page numbers and text
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([103, 79, 60])
        upper = np.array([129, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

        mask = 255 - mask
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        data = pytesseract.image_to_data(Image.open(os.path.join(input_path,filename)))

        pleaseCd = [0, 0, 0, 0]
        aboveCd = [0, 0, 0, 0]

        for d in data.splitlines():

            d = d.split("\t")
            # print(d)

            if len(d) == 12:
                # # d[11] => text field of the image
                # # d[6] => left pointer of the image
                # # d[7] => right pointer of the image
                # # d[8] => width of the image
                # # d[9] => height of the image
                flag1 = 0
                # if(len(d[11]) == 12):
                #     print(d[11])

                if(len(d[11]) == 11):
                    s = d[11][:4]
                    temp = d[11]
                    # print(d[11])
                    if(s == "SYNB" or s == "SBIN" or s == "HDFC" or s == "CNRB" or s == "HDFC" or s == "PUNB" or
                            s == "UTIB" or s == "ICIC"):
                        # result.append(d[11])
                        result = d[11]
                        print("IFSC CODE : ", d[11])

                    if(s == "1C1C"):
                        str1 = temp
                        list1 = list(str1)
                        list1[0] = 'I'
                        list1[2] = 'I'
                        str1 = ''.join(list1)
                        print("IFSC CODE : ", str1)
                    # for x in range(5):
                    #     if(d[11][x] >= 'A' and d[11][x] <= 'Z'):
                    #         flag = 1
                    # if (flag == 1):
                    #     print(d[11])
                    #     result.append(str1)
                        result = str1
                print(result)
                if d[11].lower() == "please":
                    pleaseCd[0] = int(d[6])
                    pleaseCd[1] = int(d[7])
                    pleaseCd[2] = int(d[8])
                    pleaseCd[3] = int(d[9])
                    please = please + 1
                if d[11].lower() == "sign":
                    sign = sign + 1
                if d[11].lower() == "above":
                    aboveCd[0] = int(d[6])
                    aboveCd[1] = int(d[7])
                    aboveCd[2] = int(d[8])
                    aboveCd[3] = int(d[9])
                    above = above + 1

        lengthSign = aboveCd[0] + aboveCd[3] - pleaseCd[0]
        scaleY = 2
        scaleXL = 2.5
        scaleXR = 0.5

        lengthSignCd = [0, 0, 0, 0]

        lengthSignCd[0] = int(pleaseCd[0] - lengthSign * 2.5)
        lengthSignCd[1] = int(pleaseCd[1] - lengthSign * 2)

        img = cv2.rectangle(
            img,
            (lengthSignCd[0], lengthSignCd[1]),
            (
                lengthSignCd[0] + int((scaleXL + scaleXR + 1) * lengthSign),
                lengthSignCd[1] + int(scaleY * lengthSign),
            ),
            (255, 255, 255),
            2,
        )
        cropImg = img[
            lengthSignCd[1]: lengthSignCd[1] + int(scaleY * lengthSign),
            lengthSignCd[0]: lengthSignCd[0]
            + int((scaleXL + scaleXR + 1) * lengthSign),
        ]

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static",  "OCR_Results")
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        s1 = "OCR_Result_" + filename
        if cropImg.size != 0:

            processed_files = processed_files + 1
            cv2.imwrite(os.path.join(path, s1), cropImg)

    # print(str(processed_files) + "/" + str(total_files) + " files processed successfully.")
    print("Processing Complete.")
    # print("You may check the Result folder in the same directory to see the cropped Project_Images.")
    # result.append("OCR Algorithm Successfully completed.")
    return str(result)
