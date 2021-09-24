from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QLabel, QGridLayout
import sys
import os
import onnxruntime as rt

from pdf2image import convert_from_path
from PIL.ImageQt import ImageQt

import cv2
from numpy import reshape,argmax, array,where, nonzero
import imutils
from imutils.contours import sort_contours



def get_obj_names(objects):
    ob_dict = {}
    for ob in objects:
        ob_dict[ob.objectName()] = ob
    return ob_dict

# Convert PIL image to OpenCV image
def pil2open(img):
    img = img.copy().convert('RGB')
    return array(img)[:, :, ::-1].copy()


# Red filter
def find_red(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = array([0, 50, 50])
    upper_red = array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = array([170, 50, 50])
    upper_red = array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[where(mask == 0)] = 0

    return output_img


# remove row and columns that contain only zeros
def crop(image):
    y_nonzero, x_nonzero = nonzero(image)
    return image[min(y_nonzero):max(y_nonzero), min(x_nonzero):max(x_nonzero)]


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

def preprocess(img):
    img = pil2open(img)
    # Red filter
    img = find_red(img)

    #Convert to binary image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray_img = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    #crop
    try:
       gray_img = crop(gray_img)
    except:
       pass
    # Preprocess
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edged = cv2.Canny(blurred, 100, 200)
    if img.shape[0]<400:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilated = cv2.dilate(edged, kernel)
    return dilated

def pred_grade(dilated):

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts == []: return 'NA'
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = sort_contours(cnts, method="left-to-right")[0]

    rois = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if h < dilated.shape[0]/3: continue
        roi = dilated[y:y + h, x:x + w]
        _, roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.dilate(roi, kernel)
        roi = resize_pad(roi)
        np_roi = reshape(roi / 255, (1, 1, 28, 28)).astype('float32')
        pred = sess.run(None, {input_name: np_roi})
        pred = argmax(pred)
        # print(digit)
        rois.append(pred)

    score = 'NA'
    if rois != []:
        score = ''.join([str(elem) for elem in rois])
    return score

def rename(nam,s):
    # Display the new name
    strs = nam.split('_')  # A_B_C.pdf - [A,B,C.pdf]
    lastStr = strs[-1].split('.')[0]  # C.pdf - C
    firstPart = '_'.join(strs[:-1])  # [A,B] - A_B
    if (len(lastStr) < 10) and (len(lastStr) > 4):
        # add to the last position
        # A_B + _ + C + _ + score + .pdf - A_B_C_score.pdf
        newName = firstPart + '_' + lastStr + '_' + s + '.pdf'
    else:
        # replace the last postion
        # A_B + _ + score + .pdf - A_B_score.pdf
        newName = firstPart + '_' + s + '.pdf'
    return newName

def update_name():
    for i, name in enumerate(pdf_names):
        score = labels[i][3].text()
        new_name = rename(name, score)
        if new_name != labels[i][4].text():
            labels[i][4].setText(new_name)
    qlabels['label_status'].setText(' ')

def create_grid(nrow):
    global labels,grid,ROWS,COLS,widget
    ROWS = nrow+1
    COLS = 5
    widget = QtWidgets.QWidget()  # Widget that contains the collection of Vertical Box
    grid = QtWidgets.QGridLayout()  # The Vertical Box that contains the Horizontal Boxes of  labels and buttons

    col_names = ['Index', 'Original name', 'Image', 'Prediction', 'New name']
    header_font = QtGui.QFont("Times", 10, weight=QtGui.QFont.Bold)
    for i in range(COLS):
        label = QLabel(col_names[i])
        label.setFont(header_font)
        grid.addWidget(label, 0, i)
    labels = []
    for i in range(1, ROWS):
        rlabels = []
        grid.setRowMinimumHeight(i, 100)
        for j in range(0, COLS):
            if j == 3:  # Prediction columns
                label = QtWidgets.QLineEdit('0')
                label.setMaxLength(3)
                label.setFixedWidth(40)
                label.textChanged.connect(update_name)
            else:
                label = QLabel(str(i))
            grid.addWidget(label, i, j)
            rlabels.append(label)
        labels.append(rlabels)

    widget.setLayout(grid)

    # Scroll Area Properties
    scr_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    scr_area.setWidgetResizable(True)
    scr_area.setWidget(widget)


def select_folder():
    global folder_path,pdf_names
    qlabels['label_status'].setText(' ')
    folder_path = QtWidgets.QFileDialog.getExistingDirectory()
    qlabels['label_path'].setText(folder_path)

    # Get all the pdf file name in the chosen path
    fnames = os.listdir(folder_path)
    pdf_names = [x for x in fnames if '.pdf' in x]
    no_files = len(pdf_names)

    create_grid(no_files)
    pro_bar.setMaximum(no_files)
    # for i in range(no_files, ROWS - 1):
    #     for j in range(COLS):
    #         labels[i][j].hide()
    #         grid.removeWidget(labels[i][j])
    #         labels[i][j].deleteLater()

    for i, name in enumerate(pdf_names):
        labels[i][1].setText(name)
        filename = folder_path + '/' + name
        try:
            image = convert_from_path(filename, last_page=1,
                                      poppler_path=POPPLER_PATH
                                      )[0]
            xsize, ysize = image.size
            image = image.crop((xsize * (1 - CROP_RATIO), 0, xsize, xsize * CROP_RATIO * 0.7))
            # Display the cropped image
            qim = image.resize((70,70))
            qim = ImageQt(qim.copy())
            labels[i][2].setPixmap(QtGui.QPixmap.fromImage(qim).copy())

            image = preprocess(image.copy())
            #print(image)
            score = pred_grade(image)
            #print(score)
            # Display score in the QLineEdit

        except Exception as e:
            print("Error in {}: {}".format(i,e))
            score = 'NA'
        labels[i][3].setText(str(score))
        #labels[i][3].textChanged.connect(update_name)

        new_name = rename(name, score)
        # Construct new name
        labels[i][4].setText(new_name)
        pro_bar.setValue(i+1)

# Rename function
def click_rename():
    global folder_path, pdf_names
    fnames = os.listdir(folder_path)
    pdf_names = [x for x in fnames if '.pdf' in x]
    for i, name in enumerate(pdf_names):
        new_name = labels[i][4].text()
        new_name = folder_path + '/' + new_name
        old_name = folder_path + '/' + name
        if new_name != old_name:
            os.rename(old_name, new_name)
    qlabels['label_status'].setText('Done! Please verify filenames again!')

# CONSTANT ###############

POPPLER_PATH = r'./poppler'
CROP_RATIO = 0.3

# Prediction ################
sess = rt.InferenceSession('mnist-8.onnx')
input_name = sess.get_inputs()[0].name

# MAIN WINDOW ###################
app = QtWidgets.QApplication([])

win = uic.loadUi("gh_gui.ui")  # specify the location of your .ui file
win.setWindowTitle('Grading helper v1')
wgs = win.findChildren(QtWidgets.QWidget)
wgs = get_obj_names(wgs)

qlabels = win.findChildren(QtWidgets.QLabel)
qlabels = get_obj_names(qlabels)


buttons = win.findChildren(QtWidgets.QPushButton)
pbs = get_obj_names(buttons)
pbs['pb_folder'].clicked.connect(select_folder)
pbs['pb_rename'].clicked.connect(click_rename)

pro_bar = win.findChildren(QtWidgets.QProgressBar)[0]

scr_area = win.findChildren(QtWidgets.QScrollArea)[0]


win.show()

sys.exit(app.exec())
