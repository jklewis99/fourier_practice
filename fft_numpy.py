import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = 'C:/Users/jklew/OneDrive/Desktop/reu/opencv-projects/fourier'

def main():
    im1 = cv2.imread(PATH + '/johnWilliams.jpg')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread(PATH + '/randyNewman.jpg')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    im1_ft = np.fft.fft2(im1)
    im1_fshift = np.fft.fftshift(im1_ft)
    im2_ft = np.fft.fft2(im2)
    im2_fshift = np.fft.fftshift(im2_ft)

    im1_magnitude_spectrum = 20*np.log(np.abs(im1_fshift))
    im2_magnitude_spectrum = 20*np.log(np.abs(im2_fshift))
    
    im1_phase = np.angle(im1_fshift)
    im2_phase = np.angle(im2_fshift)

    # inverse formula:
    # r*e^theta*i
    z1 = np.absolute(im1_fshift) * np.exp(1j*im1_phase)
    im1_inverse = np.fft.ifft2(z1)
    z2 = np.absolute(im2_fshift) * np.exp(1j*im2_phase)
    im2_inverse = np.fft.ifft2(z2)
    
    #swap magnitude and phase
    im1_phase_im2_magnitude = np.absolute(im2_fshift) * np.exp(1j*im1_phase)
    im2_phase_im1_magnitude = np.absolute(im1_fshift) * np.exp(1j*im2_phase)
    swap_inverse1 = np.fft.ifft2(im1_phase_im2_magnitude)
    swap_inverse2 = np.fft.ifft2(im2_phase_im1_magnitude)

    #place to display images and compare
    plt.subplot(2, 5, 1),plt.imshow(im1, cmap = 'gray')
    plt.title('Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 6),plt.imshow(im2, cmap = 'gray')
    plt.title('Image 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 2),plt.imshow(im1_magnitude_spectrum, cmap = 'gray')
    plt.title('Image 1 Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 7),plt.imshow(im2_magnitude_spectrum, cmap = 'gray')
    plt.title('Image 2 Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 3),plt.imshow(im1_phase, cmap = 'gray')
    plt.title('Image 1 Phase'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 8),plt.imshow(im2_phase, cmap = 'gray')
    plt.title('Image 2 Phase'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 4),plt.imshow(np.absolute(im1_inverse), cmap = 'gray')
    plt.title('Image 1 Inverse'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 9),plt.imshow(np.absolute(im2_inverse), cmap = 'gray')
    plt.title('Image 2 Inverse'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 5),plt.imshow(np.absolute(im1_phase_im2_magnitude))
    plt.title('Phase image 1, magnitude image 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 10),plt.imshow(np.absolute(im2_phase_im1_magnitude))
    plt.title('Phase image 2, magnitude image 1'), plt.xticks([]), plt.yticks([])
    plt.show()
    

if __name__ == "__main__":
    main()
