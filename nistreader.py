import struct

# Label file
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.

# Image file
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel


class NistReader:

    def __init__(self, name):
        self.img_file = open("mnist/" + name + "-images.idx3-ubyte", "rb")
        self.lbl_file = open("mnist/" + name + "-labels.idx1-ubyte", "rb")
        img_magic = struct.unpack('>i', self.img_file.read(4))[0]
        lbl_magic = struct.unpack('>i', self.lbl_file.read(4))[0]
        if img_magic != 2051:
            raise ValueError('Invalid image file magic')
        if lbl_magic != 2049:
            raise ValueError('Invalid label file magic')
        img_items = struct.unpack('>i', self.img_file.read(4))[0]
        lbl_items = struct.unpack('>i', self.lbl_file.read(4))[0]
        if img_items != lbl_items:
            raise ValueError('Image and label file counts mismatch')
        self.items = img_items
        self.rows = struct.unpack('>i', self.img_file.read(4))[0]
        self.cols = struct.unpack('>i', self.img_file.read(4))[0]
        self.im_size = self.rows * self.cols

    def close(self):
        self.img_file.close()
        self.lbl_file.close()

    def skip_items(self, n=1):
        self.lbl_file.read(n)
        self.img_file.read(self.im_size * n)

    def read_item(self):
        label = struct.unpack('b', self.lbl_file.read(1))[0]
        image = self.img_file.read(self.im_size)
        binary = [1 if x > 127 else 0 for x in image]
        return label, binary

    def read_acceptable(self, classes=[0, 1]):
        while True:
            label, image = self.read_item()
            if label in classes:
                return label, image

    def read_balanced(self, number, classes=[0, 1]):
        req = classes.copy()
        ret = {x: [] for x in classes}
        while True:
            label, image = self.read_acceptable(req)
            if len(ret[label]) < number:
                ret[label].append(image)
            else:
                req.remove(label)
            if len(req) == 0:
                break
        return ret

    def make_matrix(self, linear):
        mat = []
        for i in range(0, self.rows):
            mat.append(linear[self.cols * i : self.cols * (i+1)])
        return mat
