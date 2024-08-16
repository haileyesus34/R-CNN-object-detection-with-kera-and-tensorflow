from imagesearch import config
from imagesearch import iou
from bs4 import BeautifulSoup 
from imutils import paths
import cv2 
import os 

for dirPath in (config.positive_path, config.negative_path):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

imagePaths = list(paths.list_images(config.orig_images))

totalPositive = 0
totalNegative = 0 

for (i,imagePath) in enumerate(imagePaths):
    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind('.')]
    annots = os.path.sep.join([config.orig_annots, '{}.xml'.format(filename)])


    contents = open(annots).read()
    soup = BeautifulSoup(contents, 'html.parser')
    gtBoxes = []

    w = int(soup.find('width').string)
    h = int(soup.find('height').string)

    for o in soup.find_all('object'):
        label = o.find('name').string 
        xmin = int(o.find('xmin').string)
        ymin  = int(o.find('ymin').string)
        xmax = int(o.find('xmax').string)
        ymax = int(o.find('ymax').string)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        gtBoxes.append((xmin, ymin, xmax, ymax))
    
    image = cv2.imread(imagePath)
    clone = image.copy()
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(clone)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRects = []

    for (x,y,w,h) in rects: 
        proposedRects.append((x,y,x+w,y+h))


    positiveROIs = 0
    negativeROIs = 0 

    for proposedRect in proposedRects[:config.max_proposals]:
        x1, y1, x2, y2 = proposedRect
        

        for gtbox in gtBoxes:
            iou_val = iou.compute_iou(proposedRect, gtbox)
            gx1, gy1, gx2, gy2 = gtbox

            roi = None 
            outputPath = None 

            if iou_val > 0.7 and positiveROIs<= config.max_positve:
                roi = image[y1: y2, x1: x2]
                filename = '{}.png'.format(totalPositive)
                outputPath = os.path.sep.join([config.positive_path, filename])

                positiveROIs +=1
                totalPositive +=1
            
            fullOverlap = x1 >= gx1
            fullOverlap = fullOverlap and y1 >= gy1
            fullOverlap = fullOverlap and x2 <= gx2
            fullOverlap = fullOverlap and y2 <= gy2

            if not fullOverlap and iou_val < 0.05 and negativeROIs <=config.max_negative:
                 roi = image[y1: y2, x1: x2]
                 filename = '{}.png'.format(totalNegative)
                 outputPath = os.path.sep.join([config.negative_path, filename])

                 negativeROIs +=1
                 totalNegative +=1

            if roi is not None and outputPath is not None: 

                roi = cv2.resize(roi, config.input_dims, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)
