from Wavelet_functions import *

# Read GeoTIFF file
image_file = 'example.tif'
image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
image_data = image_dataset.GetRasterBand(1).ReadAsArray()
rows, cols = image_data.shape

# Define the wavelet type
wavelet = 'haar'
denoise_wavelet = 'bior3.5'

# Define the number of levels
level = math.floor(np.log2(rows) // 2)

# Define other parameters
scale                = 2
noiseSigma           = 16 
label_mask_threshold = 500
gaussian_sigma       = 3

# Image processing using wavelet transform
input_image, reconstructed_image, subtract_image, denoised_image = wavelet_processing(
        image_data, wavelet, denoise_wavelet, scale, level, noiseSigma, (1, 0, 0, 0))

# Plot images
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

images = [
    image_data,
    input_image,
    reconstructed_image,
    subtract_image,
    denoised_image
]

titles = [
    "Original Image",
    "Input Image",
    "Reconstructed Image",
    "Subtracted Image",
    "Denoised Image"
]

for ax, img, title in zip(axes, images, titles):
    if title == "Reconstructed Image":
        im = ax.imshow(img, cmap='viridis', vmin=0, vmax=700)
    else:
        im = ax.imshow(img, cmap='viridis', vmin=1500, vmax=2500)
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
