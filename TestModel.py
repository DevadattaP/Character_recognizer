import cv2
import numpy as np
import operator
import os


class CountourWithData():
    npaCountour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectW = 0
    intRectH = 0
    fitArea = 0.0

    def calcRect(self):
        [self.intRectX, self.intRectY, self.intRectW, self.intRectH] = self.boundingRect

    def contour_validity(self):
        return self.fitArea >= 100

def main():
    allContourWithData = []
    validContourWithData = []
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("ERROR : unable to open classifications.txt")
        os.system("pause")
        return

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt",np.float32)
    except:
        print("ERROR : unable to open flattened_images.txt")
        os.system("pause")
        return

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    imgTest = cv2.imread("ft1.png")

    if imgTest is None:
        print("\nERROR : cannot read image\n\n")
        os.system("pause")
        return

    imgGray = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      225,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)
    imgThreshCopy = imgThresh.copy()
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
    for npaContour in npaContours:
        contourWithData = CountourWithData()
        contourWithData.npaCountour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaCountour)
        contourWithData.calcRect()
        contourWithData.fitArea = cv2.contourArea(contourWithData.npaCountour)
        allContourWithData.append(contourWithData)

    for contourWithData in allContourWithData:
        if contourWithData.contour_validity():
            validContourWithData.append(contourWithData)

    validContourWithData.sort(key=operator.attrgetter("intRectX"))
    strFinalString = ""
    position =[]

    for contourWithData in validContourWithData:
        temp=[]
        cv2.rectangle(imgTest,
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX+contourWithData.intRectW, contourWithData.intRectY+contourWithData.intRectH),
                      (0, 225, 0),
                      2)
        xCenter = (contourWithData.intRectX + contourWithData.intRectX+contourWithData.intRectW)/2
        yCenter = (contourWithData.intRectY + contourWithData.intRectY+contourWithData.intRectH)/2
        temp.append(int(xCenter))
        temp.append(int(yCenter))

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY+contourWithData.intRectH,
                 contourWithData.intRectX: contourWithData.intRectX+contourWithData.intRectW]
        imgROIResized = cv2.resize(imgROI, (40, 50))
        npaROIResized = imgROIResized.reshape((1, 40*50))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        #strFinalString = strFinalString+strCurrentChar

        temp.append(strCurrentChar)
        temp.append(contourWithData.intRectX)
        temp.append(contourWithData.intRectY)
        temp.append(contourWithData.intRectW+contourWithData.intRectX)
        temp.append(contourWithData.intRectH+contourWithData.intRectY)

        position.append(temp)

    newposition = sorted(position, key=lambda k: [k[1], k[0]])
    x_end = 0
    for x in newposition:
        x_start = x[3]
        if abs(x_start-x_end) >20:
            strFinalString = strFinalString+" "
        strFinalString = strFinalString+x[2]
        x_end = x[5]

    print("\n"+strFinalString+"\n")
    cv2.imshow("imgTest", imgTest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
