import numpy as np
import os
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject
from matplotlib import pyplot as plt
import cv2

GLADE_FILE = 'exemplo.glade'


class interface():

    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file(GLADE_FILE)
        self.builder.connect_signals(self)

        self.IMG_FILE = ''   # Original image
        self.IMG_FINAL = ''  # overlapping image
        self.IMG_HIST = ""   # histogram
        self.IMG_RES = ''    # vector image
        self.val = ''         # val reflec


        self.window = self.builder.get_object("main_window")
        self.imgOrigem = self.builder.get_object("imgOrigem")
        self.imgFinal = self.builder.get_object("imgFinal")
        self.imgHist = self.builder.get_object("imgHist")


        self.dialog = self.builder.get_object("frame_reflectancia")
        self.val_reflec = self.builder.get_object("val_reflectancia")

        self.window.connect("delete-event", Gtk.main_quit)
        self.window.show_all()

    def new_Files_activate(self, widget):
        dialog = Gtk.FileChooserDialog("Please choose a file", None,
                                       Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

        dialog.set_modal(True)
        self.add_filters(dialog)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.IMG_FILE = dialog.get_filename()
            self.imgOrigem.set_from_file(dialog.get_filename())

            #img = cv2.imread(self.IMG_FILE)
            #plt.hist(img.ravel(), 256, [0, 50])
            #plt.savefig("my_img.jpg")

            #img = cv2.imread("my_img.jpg")
            #img_scaled = cv2.resize(img, (800, 400), interpolation=cv2.INTER_AREA)
            #cv2.imwrite("scaled.jpg", img_scaled)
            #self.imgHist.set_from_file("scaled.jpg")
            #os.remove("scaled.jpg")
            #os.remove("my_img.jpg")

        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")

        dialog.destroy()

    def add_filters(self, dialog):
        filter_image = Gtk.FileFilter()
        filter_image.set_name("Image Files")
        filter_image.add_pattern("*.png")
        filter_image.add_pattern("*.jpg")
        filter_image.add_pattern("*.tif")
        filter_image.add_pattern("*.tiff")
        dialog.add_filter(filter_image)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)

    def save_as_activate(self, widget):
        dialog = Gtk.FileChooserDialog("Save file", None,
                                       Gtk.FileChooserAction.SAVE,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            save_file = dialog.get_filename()
            self.IMG_RES = vector_capture(self.IMG_FINAL)
            cv2.imwrite(save_file+".tif", self.IMG_RES)
        elif response == Gtk.ResponseType.CANCEL:
            dialog.destroy()

        dialog.destroy()

    def exit_activate(self, widget):
        self.window.destroy()
        exit(0)

    def watershed_activate(self, widget):
        self.dialog.show_all()

    def bt_reflectancia_clicked(self, widget):
        if self.val_reflec is not None:
            self.val = self.val_reflec.get_text()
            print(self.val)
            thresh, img = open(self.IMG_FILE, float(self.val))
            opening = noise(thresh)
            sure_bg = backg(opening)
            sure_fg = foreg(opening)
            self.IMG_FINAL = watershed(sure_fg, sure_bg, img)
            cv2.imwrite("temp.jpg", self.IMG_FINAL)
            self.imgFinal.set_from_file("temp.jpg")
            os.remove("temp.jpg")
        else:
            print('lixo')

        self.dialog.destroy()

    def help_about(self, widget):
        print("help")

def open(input, val_reflec):
    img = cv2.imread(input)
    # cv2.imshow("Input", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Input", gray)
    ret, thresh = cv2.threshold(gray,val_reflec, 255,cv2.THRESH_BINARY_INV | cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # cv2.imshow("thresh", thresh)
    return thresh,img

# noise
def noise(thresh):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.medianBlur(opening,13)
    # cv2.imshow("noise", opening)
    return opening

# bg
def backg(opening):
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    # cv2.imshow("back", sure_bg)
    return sure_bg

# fg
def foreg(opening):
    kernel = np.ones((3, 3), np.uint8)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(),255,0)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow("fore", sure_fg)
    return sure_fg

# finding region
def watershed(sure_fg, sure_bg, img):
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # cv2.imshow("sub", unknown)

    # markers
    ret, markers = cv2.connectedComponents(sure_fg)

    # add
    markers = markers+1

    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0,0,255]
    # cv2.imshow("result",img)
    return img

#Vector
def vector_capture(img):
    tolerancia = 0
    color = np.uint8([[[0,0,255]]]) #cor de referencia
    hsvrColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV) #hsv
    h= hsvrColor[0,0,0]
    lower = np.array([h-tolerancia, 255, 255])
    upper = np.array([h+tolerancia,255,255])

    hsv =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img,img, mask=mask)

    #Border
    proper = res.shape
    print(proper[0],proper[1],proper[2])
    BLACK = [0,0,0]
    for i in range(0, proper[1]): #up
        res[0, i] = BLACK

    for i in range(0, proper[1]): #down
        res[proper[0]-1, i] = BLACK

    for i in range(0, proper[0]): #left
        res[i, 0] = BLACK

    for i in range(0, proper[0]): #right
        res[i, proper[1]-1] = BLACK

    # cv2.imshow("vector",res)
    return res


def main():
    interface()
    Gtk.main()
    exit(0)

main()