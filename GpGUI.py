# Import packages
from tkinter import Label,Text,Tk
from tkinter import Frame,Button,Canvas, Scrollbar
from tkinter.ttk import Progressbar
from tkinter.filedialog import askdirectory
from os import listdir, rename
from pdf2image import convert_from_path
from PIL import Image, ImageTk
from numpy import stack, reshape,array,where, nonzero
import cv2
import imutils
from imutils.contours import sort_contours
from tensorflow_hub import KerasLayer
# at the top of the file, before other imports



# ALL THE SUB FUNCTIONS
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
        np_roi = reshape(roi / 255, (28, 28, 1))
        rois.append(np_roi)
    score = 'NA'
    if rois != []:
        rois = stack(rois)
        predicts = model(rois)
        digits = predicts.numpy().argmax(1)
        score = ''.join([str(elem) for elem in digits])
    return score

poppler_path = r'./poppler'

def click_folder():
    global folderpath,pdf_names
    # Select the path
    folderpath = askdirectory()
    label2.configure(text = 'Path: ' +folderpath)

    # Get all the pdf file name in the chosen path
    fnames = listdir(folderpath)
    pdf_names = [x for x in fnames if '.pdf' in x]
    no_files = len(pdf_names)
    pbar.configure(maximum=no_files)

    # destroy unused label
    for i in range(no_files+1,ROWS):
        for j in range(COLS):
            labels[i][j].destroy()


    for i,name in enumerate(pdf_names):
        pbar.step()
        frame_main.update()

        i+=1
        labels[i][1].configure(text = name)
        filename = folderpath + '/' + name
        image = convert_from_path(filename, last_page=1,
                                  poppler_path=poppler_path
                                  )[0]
        xsize, ysize = image.size
        image = image.crop((xsize * (1 - CROP_RATIO), 0, xsize, xsize * CROP_RATIO * 0.7))

        image = preprocess(image)
        score = pred_grade(image)
        #print(score)
        labels[i][3].delete('1.0','end')
        labels[i][3].insert('1.0',score)

        # Display the cropped image
        image = Image.fromarray(image)
        image = Image.Image.resize(image, (50, 50))
        imgtk = ImageTk.PhotoImage(image=image)
        labels[i][2].configure(image=imgtk)
        labels[i][2].image = imgtk

        # Display the new name
        strs = name.split('_')  # A_B_C.pdf - [A,B,C.pdf]
        lastStr = strs[-1].split('.')[0]  # C.pdf - C
        firstPart = '_'.join(strs[:-1])  # [A,B] - A_B
        if (len(lastStr) < 10) and (len(lastStr) > 4):
            # add to the last position
            # A_B + _ + C + _ + score + .pdf - A_B_C_score.pdf
            newName = firstPart + '_' + lastStr + '_' + score + '.pdf'
        else:
            # replace the last postion
            # A_B + _ + score + .pdf - A_B_score.pdf
            newName = firstPart + '_' + score + '.pdf'
        labels[i][4].configure(text=newName)

    # Resize the scrolling region
    canvas.update_idletasks()
    canvasHeight = sum(labels[x][1].winfo_height() for x in range(no_files+1))
    canvas.config(height = canvasHeight)
    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

# Rename function
def click_rename():
    global folderpath, pdf_names
    for i, name in enumerate(pdf_names):
        i+=1
        # Display the new name
        score = labels[i][3].get('1.0','end').strip()

        strs = name.split('_') # A_B_C.pdf - [A,B,C.pdf]
        lastStr = strs[-1].split('.')[0] # C.pdf - C
        firstPart = '_'.join(strs[:-1]) # [A,B] - A_B
        if (len(lastStr) < 10) and (len(lastStr) > 4):
            # add to the last position
            # A_B + _ + C + _ + score + .pdf - A_B_C_score.pdf
            newName = firstPart+'_'+lastStr + '_' + score + '.pdf'
        else:
            # replace the last position
            # A_B + _ + score + .pdf - A_B_score.pdf
            newName = firstPart + '_' + score + '.pdf'

        labels[i][4].configure(text=newName)

        fullname = folderpath + '/' + newName
        oriName = folderpath + '/' + pdf_names[i-1]
        rename(oriName, fullname)


# CONSTANT VARIABLES
CROP_RATIO = 0.3
# Prediction
MNIST_MODEL = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
model = KerasLayer(MNIST_MODEL)

# GUI LAYOUT

# Overal window
win = Tk()
win.geometry("800x540")
win.grid_rowconfigure(0, weight=1)
win.columnconfigure(0, weight=1)
win.title('Grading Helper')

frame_main = Frame(win, bg="white")
frame_main.grid(sticky='news')

# Button to choose the folder
action = Button(frame_main,width = 20, text = 'Click me to choose folder', command = click_folder)
action.grid(column = 0, row = 0)

# Folder directory
label2 = Label(frame_main, text='Folder path', fg="blue", bg="white")
label2.grid(row=1, column=0, pady=(5, 0), sticky='nw')

# Rename button
rename_act = Button(frame_main,width = 20, text = 'Click to rename files',fg='red', command = click_rename)
rename_act.grid(column = 0, row = 3,pady=(5, 0), sticky='nw')

# Credit
label4 = Label(frame_main, text="Version:0.9 . Please send feedback to my email: thduong0811@gmail.com",bg="white")
label4.grid(row=4, column=0, pady=(5, 0), sticky='nw')

pbar = Progressbar(frame_main,orient='horizontal',length=800,mode="determinate",takefocus=True,maximum=100)
pbar.grid(row=5,column=0)

# Create a frame for the canvas with non-zero row&column weights
frame_canvas = Frame(frame_main)
frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw')
frame_canvas.grid_rowconfigure(0, weight=1)
frame_canvas.grid_columnconfigure(0, weight=1)

# Set grid_propagate to False to allow A-by-B buttons resizing later
frame_canvas.grid_propagate(False)

# Add a canvas in that frame
canvas = Canvas(frame_canvas)
canvas.grid(row=0, column=0, sticky="news")

# Link a scrollbar to the canvas
vsb = Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
vsb.grid(row=0, column=1, sticky='ns')
canvas.configure(yscrollcommand=vsb.set)

# Create a frame to contain the buttons
frame_labels = Frame(canvas, bg="blue")
canvas.create_window((0, 0), window=frame_labels, anchor='nw')

# Add ROWS-by-COLS labels to the frame
ROWS = 101
COLS = 5
labels = [[Label() for j in range(COLS)] for i in range(ROWS)]
colNames = ['Index','Original Name','Image','Prediction','New Name']
for i in range(0, ROWS):
    for j in range(0, COLS):
        if i == 0: # Header
            labels[i][j] = Label(frame_labels, text=colNames[j],font='Helvetica 10 bold')
        elif j != 3: # Labels
            labels[i][j] = Label(frame_labels, text=str(i))
        else: # Text box for correct label
            labels[i][j] = Text(frame_labels, width = 5, height = 0)
        labels[i][j].grid(row=i, column=j, sticky='news')

# Update buttons frames idle tasks to let tkinter calculate buttons sizes
frame_labels.update_idletasks()

# Resize the canvas frame
frame_canvas.config(width=win.winfo_width(),
                    height=win.winfo_height()-140)

# Set the canvas scrolling region
canvas.config(scrollregion=canvas.bbox("all"))

# Launch the GUI
win.mainloop()

