import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    im1 = cv2.imread('./johnWilliams.jpg', 0)
    im2 = cv2.imread('./randyNewman.jpg', 0)

    im1_ft = np.fft.fft2(im1)
    im1_fshift = np.fft.fftshift(im1_ft)

    im2_ft = np.fft.fft2(im2)
    im2_fshift = np.fft.fftshift(im2_ft)

    im1_magnitude_spectrum = 20 * np.log(np.abs(im1_fshift))
    im2_magnitude_spectrum = 20 * np.log(np.abs(im2_fshift))

    im1_phase = np.angle(im1_fshift)
    im2_phase = np.angle(im2_fshift)

    # inverse formula:
    # r*e^theta*i
    z1 = np.absolute(im1_fshift) * np.exp(1j*im1_phase)
    im1_inverse = np.fft.ifft2(z1)

    z2 = np.absolute(im2_fshift) * np.exp(1j*im2_phase)
    im2_inverse = np.fft.ifft2(z2)

    # swap magnitude and phase
    im1_phase_im2_magnitude = np.absolute(im2_fshift) * np.exp(1j*im1_phase)
    im2_phase_im1_magnitude = np.absolute(im1_fshift) * np.exp(1j*im2_phase)

    swap_inverse1 = np.fft.ifft2(im1_phase_im2_magnitude)
    swap_inverse2 = np.fft.ifft2(im2_phase_im1_magnitude)

    # place to display images and compare
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 5, 1)
    show_image(im1, 'Image 1')
    plt.subplot(2, 5, 6)
    show_image(im2, 'Image 2')

    plt.subplot(2, 5, 2)
    show_image(im1_magnitude_spectrum, 'Image 1 Magnitude')
    plt.subplot(2, 5, 7)
    show_image(im2_magnitude_spectrum, 'Image 2 Magnitude')

    plt.subplot(2, 5, 3)
    show_image(im1_phase, 'Image 1 Phase')
    plt.subplot(2, 5, 8)
    show_image(im2_phase, 'Image 2 Phase')

    plt.subplot(2, 5, 4)
    show_image(np.absolute(im1_inverse), 'Image 1 Inverse')
    plt.subplot(2, 5, 9)
    show_image(np.absolute(im2_inverse), 'Image 2 Inverse')

    plt.subplot(2, 5, 5)
    show_image(np.absolute(im1_phase_im2_magnitude),
               'Phase image 1, magnitude image 2')
    plt.subplot(2, 5, 10)
    show_image(np.absolute(im2_phase_im1_magnitude),
               'Phase image 2, magnitude image 1')

    plt.savefig('output.png')
    plt.show()


def show_image(img, title="Image"):
    img = np.absolute(img)
    plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])


if __name__ == "__main__":
    main()
