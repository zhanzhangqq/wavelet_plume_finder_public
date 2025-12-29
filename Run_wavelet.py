import os, sys
from Image_proc_functions import *
from Wavelet_functions import *
import argparse
import warnings
warnings.filterwarnings("ignore")


def run_wavelet(inputpath, outputpath, flight, file_name, scale, sigma_factor):
    directory_path = os.path.join(inputpath, flight)

    # Read and covert the original NetCDF to a GeoTIFF file
    initial_file = nc2tif(directory_path, outputpath, flight, file_name)

    # Create input GeoTIFF file, filling NAN values with median + gaussian noise
    input_file = initial_file.split('.tif')[0] + '_filled.tif'
    if not os.path.isfile(input_file):
        fill_nan(initial_file)
    print('GeoTIFF file created: ', input_file)

    # Open the input GeoTIFF file
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    if dataset is None:
        print("Error: Failed to open the input GeoTIFF file.")

    # Read the image data from the GeoTIFF file
    image_data = dataset.GetRasterBand(1).ReadAsArray()
    rows, cols = image_data.shape
    geo_transform = dataset.GetGeoTransform()
    min_x = geo_transform[0]
    max_y = geo_transform[3]

    # Define the wavelet type
    wavelet = 'haar'
    denoise_wavelet = 'bior3.5'

    # Define the number of levels
    n = math.floor(np.log2(rows) // 2)

    # Define other parameters
    noiseSigma = 16  # factor of the denoising soft threshold (range: 16, 32, 64, a higher value helps remove more small masks that are likely to be noise)
    label_mask_threshold = 100  # = 10p*10p = 100m*100m (avoid high values: may miss small-shape plumes)
    gaussian_sigma = 3  # sigma of Gaussian filtering (avoid low values (<1): masks may be too "shattered")

    # Image processing using wavelet transform
    input_image, reconstructed_image, subtract_image, denoised_image = wavelet_processing(
        image_data, wavelet, denoise_wavelet, scale, n, noiseSigma, (1, 0, 0, 0))

    # Add Gaussian filter to denoised image
    denoised_image = gaussian_filter(denoised_image, sigma=gaussian_sigma)

    # Create masks of original image based on denoised image
    masked_image = plume_masking(denoised_image, image_data, sigma_factor, label_mask_threshold)
    masked_image = masked_image.filled(np.nan)
    masked_image[np.isnan(image_data)] = np.nan

    # Fill NAN holes within masks
    masked_image_geotransform = (min_x, max_y, rows, cols)
    masked_image = remove_small_clumps(masked_image, label_mask_threshold)
    masked_image = fill_nan_within_mask(masked_image, masked_image_geotransform, initial_file)

    # Save output to GeoTIFF files
    denoised_file = initial_file.split('.tif')[0] + '_denoised.tif'
    create_geotiff(dataset, denoised_image, denoised_file)

    masked_file = initial_file.split('.tif')[0] + '_masked.tif'
    create_geotiff(dataset, masked_image, masked_file)
    print('Wavelet processing done.')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wavelet processing.")
    parser.add_argument("inputpath", nargs='?', type=str, default='/your_path/to/NetCDF/files/', 
                    help="Input path to the NetCDF files.")
    parser.add_argument("outputpath", nargs='?', type=str, default='/your_path/to/NetCDF/files/', 
                    help="Output path for GeoTIFF files.")
    parser.add_argument("flight", type=str, help="Flight name.")
    parser.add_argument("file_name", type=str, help="NetCDF file name.")
    parser.add_argument("scale", nargs='?', type=float, default=2, help="Factor of the consistent signal threshold, try 1.5/2/2.5")
    parser.add_argument("sigma_factor", nargs='?', type=float, default=1.5, help="Factor of the plume masking threshold, try 2/2.5/3")
    
    # Run the function with the provided arguments
    args = parser.parse_args()
    run_wavelet(args.inputpath, args.outputpath, args.flight, args.file_name, args.scale, args.sigma_factor)
