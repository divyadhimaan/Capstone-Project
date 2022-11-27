try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Simple image to string
# print(pytesseract.image_to_string(Image.open('Cheque083654.jpg')))

# # French text image to string
# # print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# # Get bounding box estimates
please = 0
sign = 0
above = 0
total_files = 0
processed_files = 0


images_dir = "../../../Our_Dataset/cheque_images"
# images_dir = "image"
input_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), images_dir
)

for filename in os.listdir(input_path):
    total_files = total_files + 1
    if ".jpg" in filename:
        print("OCR Processing file -", filename)
        img = cv2.imread(os.path.join(input_path,filename))
    else:
        continue
    h, w, _ = img.shape  # assumes color image

    # Get verbose data including boxes, confidences, line, page numbers and text
    data = pytesseract.image_to_data(Image.open(os.path.join(input_path,filename)))

    # print (data)
    pleaseCd = [0, 0, 0, 0]
    aboveCd = [0, 0, 0, 0]

    for d in data.splitlines():
        d = d.split("\t")
        if len(d) == 12:
            # # d[11] => text field of the image
            # # d[6] => left pointer of the image
            # # d[7] => right pointer of the image
            # # d[8] => width of the image
            # # d[9] => height of the image

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

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OCR_Results")
    if not os.path.exists(path):
        os.makedirs(path)

    s1 = "OCR_Result_" + filename
    if cropImg.size != 0:

        processed_files = processed_files + 1
        cv2.imwrite(os.path.join(path, s1), cropImg)

print(str(processed_files) + "/" + str(total_files) + " files processed successfully.")
print("Processing Complete.")
print("You may check the Result folder in the same directory to see the cropped Project_Images.")