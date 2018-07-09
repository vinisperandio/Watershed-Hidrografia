import gtk

class teste(gtk.Window):
    def __init__(self):
        super(teste, self).__init__()
        self.set_default_size(400,300)
        self.set_title("gtk")

        self.img = gtk.Image()
        self.img.set_from_file("ikonos1.jpg")

        self.box1 = gtk.VBox()
        self.box1.pack_start(self.img)

        self.add(self.box1)
        self.show_all()

teste()
gtk.main()