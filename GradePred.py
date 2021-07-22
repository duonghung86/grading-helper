import os
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import cv2
import numpy as np
from imutils.contours import sort_contours
import imutils
import time
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory
import tensorflow as tf
import tensorflow_hub as hub



# Convert PIL image to OpenCV image
def pil2open(img):
    img = img.copy().convert('RGB')
    return np.array(img)[:, :, ::-1].copy()


# red filter
def find_red(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    return output_img


# remove row and columns that contain only zeros
def crop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def resize_pad(img):
    # Step 1: Resize to 20x20
    nsize = 20
    old_size = img.shape[:2]  # old_size is in (height, width) format
    ratio = float(nsize) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]

    # new_size should be in (width, height) format
    for i in range(2):  # in case for number 1
        if new_size[i] == 0:
            new_size[i] = 2
    new_size = tuple(new_size)
    im = cv2.resize(img, (new_size[1], new_size[0]))

    # Pad to 28x28
    delta_w = nsize - new_size[1]
    delta_h = nsize - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    new_im = cv2.copyMakeBorder(new_im, 4, 4, 4, 4, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def plot_img(ax, img, gray=False, title=None):
    if gray:
        ax.imshow(img, 'gray')
    else:
        ax.imshow(img)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')


classifier_model = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
model = hub.KerasLayer(classifier_model)


def pred_grade(img, size=0.35, vis=False):
    # crop the top right area
    xsize, ysize = img.size
    img = img.crop((xsize * (1 - size - 0.1), 0, xsize, xsize * size * 0.7))
    # Convert PIL image to OpenCV image
    img = pil2open(img)

    # Red filter
    img = find_red(img)

    # Convert to binary image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray_img = cv2.threshold(gray_img, 70, 255, cv2.THRESH_BINARY)
    # crop
    try:
        crop_img = crop(gray_img)
    except:
        return 'NA'

    # preprocess
    blurred = cv2.GaussianBlur(crop_img, (3, 3), 0)
    edged = cv2.Canny(blurred, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edged, kernel)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = sort_contours(cnts, method="left-to-right")[0]

    heights = [cv2.boundingRect(c)[-1] for c in cnts]
    hmean, hstd = np.mean(heights), np.std(heights)

    rois = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if h < 40: continue
        if (h < hmean - 2 * hstd) or (h > hmean + 2 * hstd): continue
        roi = dilated[y:y + h, x:x + w]
        _, roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.dilate(roi, kernel)
        roi = resize_pad(roi)
        np_roi = np.reshape(roi / 255, (28, 28, 1))
        rois.append(np_roi)
    # Prediction
    if rois != []:
        rois = np.stack(rois)
        predicts = model(rois)
        digits = predicts.numpy().argmax(1)
        score = ''.join([str(elem) for elem in digits])

    # Visualization
    if vis == True:
        plt.imshow(img)
        plt.axis('off')
    elif vis == 'detail':
        fig, ax = plt.subplots(1, 6, figsize=(12, 2), dpi=200)
        plot_img(ax[0], image, title='1st page')
        plot_img(ax[1], gray_img, title='Cropped')
        plot_img(ax[2], dilated, True, title='Dilated')
        # Plot each number
        for j in range(rois.shape[0]):
            roi = rois[j, :, :, 0]
            plot_img(ax[j + 3], roi, gray=True, title=score[j])
        plt.show()
    if int(score) > 100: score = score[1:]
    return score

#path0 = os.path.normpath(folderName)
path0 = '''C:/Users/duong/OneDrive - University Of Houston/Math PhD/2021 Summer/TA-MATH3321/Midterm FR'''
print(path0)
#Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
#folderName = askdirectory()  # show an "Open" dialog box and return the path to the selected file
#print(folderName)

# Get all pdf files
fnames = os.listdir(path0)
pdf_names = [x for x in fnames if '.pdf' in x]
print(len(pdf_names), pdf_names[0])
grade = []
for i in range(len(pdf_names)):
    filename = path0 + '/' + pdf_names[i]
    image = convert_from_path(filename, last_page=1, )[0]
    score = pred_grade(image)
    firstPath = filename.split('_')[:-1]
    firstPath = '_'.join(firstPath)
    newName = firstPath + '_' + score+ '.pdf'
    print(newName)
    os.rename(filename,newName)
    grade.append(score)

import pandas as pd
d = {'oldName':pdf_names,'grade': grade}
df = pd.DataFrame(d)
df.to_csv('scores.csv')
#plt.show()

# n_test = 25 # Select the number of image you want to test
# nrow = n_test//5+1
# fig, ax = plt.subplots(nrow,5,figsize=(10,2*nrow,),dpi = 200)
# count = 1
# for i in range(len(pdf_names)):
#     print(i, end=' -> ')
#     start = time.time()
#     if (i+1)%15 == 0 : print()
#     true_score = pdf_names[i].split('_')[-1].split('.')[0]
#     if true_score.find('9') != -1: continue
#     # import the first page of the chosen pdf as an image
#     filename = os.path.join(path0, pdf_names[i])
#     image = convert_from_path(filename,last_page=1,)[0]
#     plt.subplot(nrow,5,count)
#     score = pred_grade(image,vis=True)
#     print(score)
#     title='#{}: {}({:.2f}s)'.format(i,score,time.time()-start)+score
#     plt.title(title)
#     count+=1
#     if count > n_test: break
# #plt.savefig('hwdr_{}.png'.format(n_test),dpi =300, bbox_inches='tight')
# plt.show()
