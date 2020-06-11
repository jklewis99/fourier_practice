import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = 'C:/Users/jklew/OneDrive/Desktop/reu/opencv-projects/fourier'

def main():
    im = cv2.imread(PATH + '/madmax.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    w = int(im.shape[1] * 0.5)
    h = int(im.shape[0] * 0.5)
    dim = (w, h)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

    dft = cv2.dft(np.float32(im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    # cv2.cartToPolar() returns both magnitude and phase in single shot

    rows, cols = im.shape
    crow,ccol = rows//2 , cols//2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    plt.subplot(121),plt.imshow(im, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
