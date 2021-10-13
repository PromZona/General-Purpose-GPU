import argparse

from PIL import Image
import numpy as np

pal = [
	(0, 0, 0),			(128, 128, 128),	(192, 192, 192),	(255, 255, 255),
	(255, 0, 255),		(128, 0, 128),		(255, 0, 0),		(128, 0, 0),
	(205, 92, 92),		(240, 128, 128),	(250, 128, 114),	(233, 150, 122),
	(205, 92, 92),		(240, 128, 128),	(250, 128, 114),	(233, 150, 122),
	(173, 255, 47),		(127, 255, 0),		(124, 252, 0),		(0, 255, 0),
	(50, 205, 50),		(152, 251, 152),	(144, 238, 144),	(0, 250, 154),
	(0, 255, 127),		(60, 179, 113),		(46, 139, 87),		(34, 139, 34),
	(0, 128, 0),		(0, 100, 0),		(154, 205, 50),		(107, 142, 35),
	(128, 128, 0),		(85, 107, 47),		(102, 205, 170),	(143, 188, 143),
	(32, 178, 170),		(0, 139, 139),		(0, 128, 128)
]

def bin_to_image(src, dest):
    with open(src, 'rb') as f:
        arr = f.read()
        
    im = Image.open('Input.png')
    pixelMap = im.load()
        
    img = Image.new('RGBA', im.size)
    pixelsNew = img.load()
    
    w = int.from_bytes(arr[:4], byteorder='little')
    h = int.from_bytes(arr[4:8], byteorder='little')
    print(f'{w}x{h}')
    print(img.mode)
    
    dt = np.dtype([('R','u1'), ('G','u1'), ('B','u1'), ('A','u1')])
    data = np.frombuffer(arr, dtype=dt)
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixelsNew[i, j] = pal[data[i + j * w][3] + 3] + (255,)         
    
    im.close()
    img.save(dest)
    img.close()

def image_to_bin(src, dest):
    res = b''

    with Image.open(src) as img:
        w, h = map(lambda x: x.to_bytes(4, byteorder='little'), img.size)
        res += w + h

        for pixel in img.convert('RGBA').getdata():
            r, g, b, alpha = map(lambda x: x.to_bytes(1, byteorder='little'), pixel)
            res += r + g + b + alpha

    with open(dest, 'wb') as f:
        f.write(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode', action='store_true', help='Convert image to bin')
    parser.add_argument('--decode', action='store_true', help='Convert bin to image')
    parser.add_argument('--src', required=True)
    parser.add_argument('--dest', required=True)
    args = parser.parse_args()
    
    if args.encode and args.decode:
        print('Choose encode or decode')
    elif args.decode:
        bin_to_image(args.src, args.dest)
    elif args.encode:
        image_to_bin(args.src, args.dest)