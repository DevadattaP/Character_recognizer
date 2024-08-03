import sys
import numpy as np
import cv2
import os


def main():
    imgTrainChar = cv2.imread("training_chars.png")

    if imgTrainChar is None:
        print("\nERROR : cannot read image\n\n")
        os.system("pause")
        return

    imgGray = cv2.cvtColor(imgTrainChar, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)
    cv2.imshow("imgThresh",imgThresh)
    imgThreshCopy = imgThresh.copy()
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)
    npaFlattenedImages = np.empty((0, 40*50))
    intClassifications = []
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'),
                     ord('6'), ord('7'), ord('8'), ord('9'), ord('A'), ord('B'),
                     ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'),
                     ord('I'), ord('J'), ord('K'), ord('L'), ord('M'), ord('N'),
                     ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'),
                     ord('g'), ord('h'), ord('i'), ord('j'), ord('k'), ord('l'),
                     ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'),
                     ord('s'), ord('t'), ord('u'), ord('v'), ord('w'), ord('x'),
                     ord('y'), ord('z')]
    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > 100:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
            cv2.rectangle(imgTrainChar,
                          (intX, intY),
                          (intX + intW, intY + intH),
                          (0, 0, 225),
                          2)
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            imgROIResized = cv2.resize(imgROI, (40, 50))
            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("train", imgTrainChar)

            intChar = cv2.waitKey(0)

            if intChar ==27:
                print("Exiting...")
                sys.exit()
            elif intChar in intValidChars:
                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResized.reshape((1, 40*50))
                npaFlattenedImages = np.append(npaFlattenedImages,npaFlattenedImage,0)


    fitClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fitClassifications.reshape((fitClassifications.size, 1))
    print("\n\n training complete !! \n\n")
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
