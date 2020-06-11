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

    fourier_transform = np.fft.fft2(im)
    fshift = np.fft.fftshift(fourier_transform)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    # plt.imshow(magnitude_spectrum)
    phase = np.angle(fshift)
    # plt.imshow(phase)
    plt.imshow(im)
    # inverse formula:
    # r*e^theta*i
    z = np.absolute(fshift) * np.exp(1j*phase)
    inverse = np.fft.ifft2(z)
    plt.imshow(np.absolute(inverse), cmap = 'gray')
    # cv2.imshow("name window", np.absolute(inverse))
    plt.show()

if __name__ == "__main__":
    main()
