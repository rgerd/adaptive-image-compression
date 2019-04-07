import cv2 as cv

def get_image(path):
    return cv.imread(path, cv.IMREAD_GRAYSCALE)

def save_image(path, img):
    return cv.imwrite(path, img);

if __name__ == '__main__':
    save_image("imgs/hexagons_gs.png", get_image("imgs/hexagons.png"))