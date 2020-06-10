from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageTk, Image, ImageGrab
import imageio
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.geometry("875x425+283+275")

        self.digit_model = keras.models.load_model("models/digit_model")
        self.symbol_model = keras.models.load_model('models/operator_model')

        pencil = ImageTk.PhotoImage(Image.open ("buttons/pencil.png"))
        self.pen_button = Button(self.root, image = pencil, command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        eraser = ImageTk.PhotoImage(Image.open("buttons/eraser.png"))
        self.eraser_button = Button(self.root, image = eraser, command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.process_button = Button(self.root, text='process', command = self.getter)
        self.process_button.grid(row=0, column = 4)

        self.clear_button = Button(self.root, text='clear', command = self.clear)
        self.clear_button.grid(row = 0, column = 3)

        self.c = Canvas(self.root, bg='white', width=842, height=282, highlightthickness = 1, highlightbackground = "black")
        self.c.grid(row=1, columnspan=5, padx = 10, pady = 10)

        self.digit1Label = Label(self.root, text = "", font=("Courier", 16))
        self.digit1Label.grid(row = 2, column = 0)
        self.symbolLabel = Label(self.root, text = "", font=("Courier", 16))
        self.symbolLabel.grid(row = 2, column = 2, sticky = E)
        self.digit2Label = Label(self.root, text = "", font=("Courier", 16))
        self.digit2Label.grid(row = 2, column = 4)
        self.equation = Label(self.root, text = "", font=("Courier", 20), foreground = 'red')
        self.equation.grid(row = 3, column = 0, columnspan = 5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 1
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button

        self.c.create_line(280, 0, 280, 281)
        self.c.create_line(562, 0, 562, 281)

        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def clear(self):
        self.c.delete("all")
        self.c.create_line(280, 0, 280, 281)
        self.c.create_line(562, 0, 562, 281)

    def paint(self, event):
        self.line_width = 10
        paint_color = 'white' if self.eraser_on else 'black'
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
        self.c.create_line(280, 0, 280, 281)
        self.c.create_line(562, 0, 562, 281)

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def getter(self):
        widget = self.c
        x11=self.root.winfo_x() + 303
        y11=self.root.winfo_y() + 423
        x12=x11+ 560
        y12=y11+ 560
        ImageGrab.grab().crop((x11,y11,x12,y12)).save("process/digit1.png")

        x21=x12 + 2
        y21=y11
        x22=x21 + 560
        y22=y21 + 560
        ImageGrab.grab().crop((x21,y21,x22,y22)).save("process/symbol.png")

        x31=x22 + 2
        y31=y21
        x32=x31 + 560
        y32=y31 + 560
        ImageGrab.grab().crop((x31,y31,x32,y32)).save("process/digit2.png")
        self.main()

    def read_image(self, path):
        image = Image.open(path)
        new_image = image.resize((28, 28))
        new_image.save(path)
        im = imageio.imread(path)
        # Turn the image into grayscale
        im = np.dot(im[...,:3], [0.299, 0.587, 0.114])
        # # Display the image to the user
        # plt.imshow(im, cmap = plt.get_cmap('gray'))
        # plt.show()
        # Reshape and normalize the image so that the model can "read" it
        im = im.reshape(1, 28, 28, 1)
        im /= 255
        return im
    
    def main(self):
        # Labels
        symbols = ["+", "-", "/", "*", "="]
        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # First digit
        digit1 = self.read_image("process/digit1.png")
        dPrediction1 = self.digit_model.predict(digit1)
        d1 = dPrediction1.argmax()
        d1_confidence = dPrediction1[0][dPrediction1.argmax()]
        self.digit1Label["text"] = "%d with %.2f%% confidence!" % (d1, d1_confidence * 100)

        # Symbol
        symbol = self.read_image("process/symbol.png")
        sPrediction = self.symbol_model.predict(symbol)
        s = symbols[sPrediction.argmax()]
        s_confidence = sPrediction[0][sPrediction.argmax()]
        self.symbolLabel["text"] = "%s with %.2f%% confidence!" % (s, s_confidence * 100)

        # Second digit
        digit2 = self.read_image("process/digit2.png")
        dPrediction2 = self.digit_model.predict(digit2)
        d2 = dPrediction2.argmax()
        d2_confidence = dPrediction2[0][dPrediction2.argmax()]
        self.digit2Label["text"] = "%d with %.2f%% confidence!" % (d2, d2_confidence * 100)

        # Get overall confidence
        confidence = d1_confidence * s_confidence * d2_confidence

        # # Perform the operation
        if s == "+":
            self.equation["text"] = "%d + %d = %d with overall %.2f%% confidence!" % (d1, d2, d1 + d2, confidence * 100)
        elif s == "-":
            self.equation["text"] = "%d - %d = %d with overall %.2f%% confidence!" % (d1, d2, d1 - d2, confidence * 100)
        elif s == "*":
            self.equation["text"] = "%d * %d = %d with overall %.2f%% confidence!" % (d1, d2, d1 * d2, confidence * 100)
        elif s == "/":
            self.equation["text"] = "%d / %d = %.2f with overall %.2f%% confidence!" % (d1, d2, d1 / d2, confidence * 100)
        elif s == "=":
            self.equation["text"] = "%d = %d with overall %.2f%% confidence! Your equation is %r!" % (d1, d2, confidence * 100, d1 == d2)
        


if __name__ == '__main__':
    Paint() 