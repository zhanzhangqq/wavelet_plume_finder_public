import math
import pywt
import numpy as np
import matplotlib.pyplot as plt
from Image_proc_functions import *



# This function does image pre-processing by setting strong signals to consistent values
# scale: scaling factor that linearly affects the threshold (default = 2)
def consist_signals(image, scale):
    output_image = image.copy()
    mu, sigma = np.mean(output_image.flatten()), np.std(output_image.flatten())
    threshold = mu + scale * sigma
    output_image[output_image > threshold] = np.max(output_image.flatten())  # signals higher than thresholds are set as the max pixel value of the image

    return output_image


# This function applies wavelet transform and modify coefficients
# wavelet: The wavelet used (suggest 'haar')
# level: The level of wavelet transform
# set_zero: In coefficients (cA, (cH, cV, cD)), if each type is set to zero.
# (1, 0, 0, 0) means approximations are set as zero. (0, 1, 1, 1) means details are set as zero.
def modify_wt_coeffs(image, wavelet, level, set_zero):
    coeffs = pywt.wavedec2(image, wavelet, level=level, mode='symmetric')
    
    # Extract approximation and detail coefficients
    approximations = coeffs[0]    # Approximation at level n
    details = coeffs[1:]          # Detail coefficients at levels 1, 2, ..., n

    # Modify approximations and details coefficients
    (approx_zero, detail0_zero, detail1_zero, detail2_zero) = set_zero
    modified_details = []
    if approx_zero:
        approximations = np.zeros_like(approximations)
    for (cH, cV, cD) in details:
        cH = np.zeros_like(cH) if detail0_zero else cH
        cV = np.zeros_like(cV) if detail1_zero else cV
        cD = np.zeros_like(cD) if detail2_zero else cD
        modified_details.append((cH, cV, cD))
    
    modified_coeffs = (approximations, *modified_details)
    modified_coeffs = list(modified_coeffs)
    return modified_coeffs







# This function compares the approximation coeff shape to one of the detail coeffs and 
# truncates the last element along the axis to match dimensions if necessary
# Source: https://github.com/PyWavelets/pywt/blob/main/pywt/_multilevel.py Line 445
def _match_coeff_dims(a_coeff, d_coeff):
    size_diffs = np.subtract(a_coeff.shape, d_coeff.shape)
    if np.any((size_diffs < 0) | (size_diffs > 1)):
        print(size_diffs)
        raise ValueError("incompatible coefficient array sizes")
    return a_coeff[tuple(slice(s) for s in d_coeff.shape)]






# This function applies image denoising using wavelet transform
# wavelet: The wavelet used for denoising (suggest biorthogonal: 'bior4.4')
# noiseSigma: A parameter of the soft threshold
def denoise_image(image, wavelet, noiseSigma):
    row, col = image.shape
    n = math.floor(np.log2(row))

    # Appy wavelet transform to get coefficients
    coeffs = pywt.wavedec2(image, wavelet, level=n, mode='symmetric')
    approximation = coeffs[0]
    details = coeffs[1:]

    # Apply soft thresholding to detail coefficients
    threshold = noiseSigma * np.sqrt(2 * np.log2(row * col))
    denoised_details = []
    for detail in details:
        denoised_detail = pywt.threshold(detail, threshold, mode='soft')
        denoised_details.append(tuple(denoised_detail))

    # Reconstruct the denoised image using modified coefficients
    denoised_coeffs = (approximation, *denoised_details)
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet, mode='symmetric')
    denoised_image = _match_coeff_dims(denoised_image, image) # Match dimensions

    return denoised_image






# This function does image processing to L3 data based on wavelet transform
# Example: set_zero = (1, 0, 0, 0):
# It first extracts detail coefficients to form a reconstructed image, where 
# approximations are set as zero (flat image) to preserve the high-frequency 
# information only. Then the reconstructed image is subtracted from the input image. 
# The resulting subtract image contains much less noise. Note that the input image 
# is pre-processed by setting strong signals to consistent values. In this way, the 
# strong signals are more "stable" and thus saved in subtract image. Lastly, do 
# wavelet thresholding to further denoise the image.
def wavelet_processing(image, wavelet, denoise_wavelet, scale, level, noiseSigma, set_zero):
    # Step 0: Pre-processing.
    input_image = consist_signals(image, scale)

    # Step 1: Wavelet transform with modified coefficients.
    modified_coeffs = modify_wt_coeffs(input_image, wavelet, level, set_zero)

    # Step 2: Inverse wavelet transform
    reconstructed_image = pywt.waverec2(modified_coeffs, wavelet, mode='symmetric')
    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image = _match_coeff_dims(reconstructed_image, input_image) # Match dimensions

    # Step 3: Subtract the reconstructed image from the input image
    subtract_image = input_image - reconstructed_image
    subtract_image[subtract_image < 0] = 0

    # Step 4: Denoise image using wavelet thresholding
    denoised_image = denoise_image(subtract_image, denoise_wavelet, noiseSigma)
    
    return input_image, reconstructed_image, subtract_image, denoised_image







# This function plots the decomposed coefficients of a level of wavelet transform
def plot_decomposed_coeffs(image, level):
    row, col = image.shape
    if level == 0:
        print('Error: Level should be higher than 0.')
    elif level > math.floor(np.log2(row)):
        print('Error: Level should not exceed the highest level.')
    coeffs = pywt.wavedec2(image, 'haar', level=level, mode='symmetric')
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    plt.figure(figsize=(10, 10))

    # Approximation image (cA)
    plt.subplot(2, 2, 1)
    plt.imshow(LL, cmap='gray')
    plt.title('Approximation Image (cA, LL), n = {}'.format(str(level)))
    plt.colorbar()

    # Horizontal detail image (cH)
    plt.subplot(2, 2, 2)
    plt.imshow(LH, cmap='gray')
    plt.title('Horizontal Detail Image (cH, LH)')
    plt.colorbar()

    # Vertical detail image (cV)
    plt.subplot(2, 2, 3)
    plt.imshow(HL, cmap='gray')
    plt.title('Vertical Detail Image (cV, HL)')
    plt.colorbar()

    # Diagonal detail image (cD)
    plt.subplot(2, 2, 4)
    plt.imshow(HH, cmap='gray')
    plt.title('Diagonal Detail Image (cD, HH)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    print('min: {}'.format(str(np.min(LL.flatten()))))
    print('max: {}'.format(str(np.max(LL.flatten()))))
