import os, sys
from netCDF4 import Dataset
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from osgeo import gdal
from scipy.ndimage import gaussian_filter
import numpy as np
import numpy.ma as ma
import cv2
import geojson
from shapely.geometry import shape, Point, Polygon



# This function does plume masking
# array: The array based on whose pixel values masks are created
# apply_array: The array masks are applied to
# sigma_factor: Sigma scaling factor for the masking
# label_mask_threshold: Min count of pixels in a clump
def plume_masking(array, apply_array, sigma_factor, label_mask_threshold):
    # Mask out pixels based on threshold (mu + sigma * s)
    mu, sigma = np.nanmean(array), np.nanstd(array)
    threshold = mu + sigma_factor * sigma
    
    if np.array_equal(array, apply_array):
        masked_array = ma.masked_less(array, threshold)
    else:
        temp_masked_array = ma.masked_less(array, threshold)
        masked_array = ma.masked_array(apply_array, temp_masked_array.mask)
    
    # Generate clump labels, further mask out clumps based on min pixel count
    binary_mask = np.logical_not(masked_array.mask).astype(np.uint8)  # Generate a binary mask (0 for masked values, 1 for unmasked values)
    num_labels, labeled_image = cv2.connectedComponents(binary_mask)
    sizes = np.bincount(labeled_image.ravel())
    mask_sizes = sizes > label_mask_threshold
    mask_sizes[0] = 0  # Ignore background (label 0)
    filtered_image = mask_sizes[labeled_image]
    masked_array[filtered_image == 0] = np.ma.masked
    
    return masked_array


# This function masks out small clumps based on min pixel count
def remove_small_clumps(masked_image, label_mask_threshold):
    masked_image_copy = masked_image.copy()
    
    masked_image_binary = ~np.isnan(masked_image).astype(bool)
    num_labels, labels = cv2.connectedComponents(masked_image_binary.astype(np.uint8))
    sizes = np.bincount(labels.ravel())
    mask_sizes = sizes > label_mask_threshold
    mask_sizes[0] = 0  # Ignore background (label 0)
    filtered_image = mask_sizes[labels]
    masked_image_copy[filtered_image == 0] = np.nan

    return masked_image_copy


# This functin creates a new GeoTIFF file of the image array based on the given dataset
# dataset: The dataset with GeoTransform information of the image
def create_geotiff(dataset, image, output_file):
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_band = output_dataset.GetRasterBand(1)

    output_array = np.ma.filled(image, fill_value=np.nan)  # Convert masked array to array with NaNs

    output_band.WriteArray(output_array)
    output_band.FlushCache()
    output_band.SetNoDataValue(np.nan)

    # Close the datasets
    dataset = None
    output_dataset = None

    print(f"Geotiff file generated sucessfully. Output file: {output_file}")

# This function crops a GeoTIFF image to a smaller one based on the center coordinates and box size
def crop_geotiff(image_array, input_file, output_file, longitude, latitude, half_box_size):
    center_x, center_y = longitude, latitude

    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()

    min_x = geo_transform[0]
    max_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = -1 * geo_transform[5]

    min_x_crop = center_x - half_box_size
    max_x_crop = center_x + half_box_size
    min_y_crop = center_y - half_box_size
    max_y_crop = center_y + half_box_size

    start_x = round((min_x_crop - min_x) / pixel_width)
    start_y = round((max_y - max_y_crop) / pixel_height)
    width = int((max_x_crop - min_x_crop) / pixel_width)
    height = int((max_y_crop - min_y_crop) / pixel_height)
    
    start_x = 0 if start_x < 0 else start_x
    start_y = 0 if start_y < 0 else start_y
    if start_y+height > rows:
        height -= (start_y+height - rows)
    if start_x+width > cols:
        width -= (start_x+width - cols)

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_file, width, height, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform((min_x_crop, pixel_width, 0, max_y_crop, 0, -1*pixel_height))
    dst_ds.SetProjection(dataset.GetProjection())
    
    dst_ds.GetRasterBand(1).WriteArray(image_array[start_y:start_y+height, start_x:start_x+width])

    # Close datasets
    src_ds = None
    dataset = None

# This function crops a GeoTIFF image to a smaller one based on the center coordinates and box size
def crop_geotiff_image(image_array, input_file, longitude, latitude, half_box_size):
    center_x, center_y = longitude, latitude

    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    geo_transform = dataset.GetGeoTransform()

    min_x = geo_transform[0]
    max_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = -1 * geo_transform[5]

    min_x_crop = center_x - half_box_size
    max_x_crop = center_x + half_box_size
    min_y_crop = center_y - half_box_size
    max_y_crop = center_y + half_box_size

    start_x = round((min_x_crop - min_x) / pixel_width)
    start_y = round((max_y - max_y_crop) / pixel_height)
    width = int((max_x_crop - min_x_crop) / pixel_width)
    height = int((max_y_crop - min_y_crop) / pixel_height)
    
    start_x = 0 if start_x < 0 else start_x
    start_y = 0 if start_y < 0 else start_y
    if start_y+height > rows:
        height -= (start_y+height - rows)
    if start_x+width > cols:
        width -= (start_x+width - cols)

    output_array = image_array[start_y:start_y+height, start_x:start_x+width]
    
    return output_array, start_x, start_y, width, height


# This function reads a GeoTIFF file and extract GeoTransform information
# Used for setting X and Y axis to coordinates in plots
def geotransform_info(dataset):
    image_data = dataset.GetRasterBand(1).ReadAsArray()
    row, col = image_data.shape

    geo_transform = dataset.GetGeoTransform()
    min_x = geo_transform[0]
    max_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = -1 * geo_transform[5]
    max_x = min_x + col * pixel_width
    min_y = max_y - row * pixel_height
    extent = (min_x, max_x, min_y, max_y)

    return extent


# This function fills NAN values of a GeoTIFF file with median + gaussian noise
def fill_nan(input_file):
    filled_file = input_file.split('.tif')[0] +'_filled.tif'
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # Replace NAN values with histogram peak + gaussian noise
    non_nan_values = array[~np.isnan(array)]
    if not non_nan_values.any():
        print('The cropped image is empty. Stop generating filled image.')
    else:
        hist, bins = np.histogram(non_nan_values, bins=50)
        peak_value = bins[np.argmax(hist)]
        noise = np.random.normal(0, np.std(non_nan_values), array.shape)
        filled_array = np.where(np.isnan(array), peak_value + noise, array)

        # Create a new GeoTIFF file with the filtered array
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(filled_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
        output_dataset.SetProjection(dataset.GetProjection())
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())
        output_band = output_dataset.GetRasterBand(1)
        output_band.WriteArray(filled_array)
        output_band.FlushCache()
        output_band.SetNoDataValue(np.nan)
        
        # Close the datasets
        dataset = None
        output_dataset = None
        print(f"Filled NAN values with median + gaussian noise. Output file: {filled_file}")


# This function changes the filled NAN values back to NAN
def reverse_fill_nan(image, initial_file):
    dataset_initial = gdal.Open(initial_file, gdal.GA_ReadOnly)  # Read the initial image data
    if dataset_initial is None:
        print("Error: Failed to open the initial GeoTIFF file.")
    initial_image = dataset_initial.GetRasterBand(1).ReadAsArray()
    image[np.isnan(initial_image)] = np.nan

    return image

# This function creates masks of hotspots based on pixel values of the original image
def hotpost_masking(array, threshold, label_mask_threshold):
    # Mask out pixel values lower than threshold and pixels with NAN values
    hotspot_array = ma.masked_less(array, threshold)
    hotspot_array = ma.masked_invalid(hotspot_array)
    
    # Generate clump labels, further mask out clumps based on min pixel count
    binary_mask = np.logical_not(hotspot_array.mask).astype(np.uint8)  # Generate a binary mask (0 for masked values, 1 for unmasked values)
    num_labels, labeled_image = cv2.connectedComponents(binary_mask)
    sizes = np.bincount(labeled_image.ravel())
    mask_sizes = sizes >= label_mask_threshold
    mask_sizes[0] = 0  # Ignore background (label 0)
    filtered_image = mask_sizes[labeled_image]
    hotspot_array[filtered_image == 0] = np.ma.masked
    hotspot_array = hotspot_array.filled(np.nan)
    
    return hotspot_array

# This function creates a new GeoTIFF based on the masked image and hotspot image, preserving the 
# masked clumps that contain hotspots only
def preserve_hotspot(masked_image, hotspot_image):
    # Load images
    image1 = masked_image
    image2 = hotspot_image

    # Convert to binary images for connected component labeling
    image1_binary = ~np.isnan(image1).astype(bool)
    image2_binary = ~np.isnan(image2).astype(bool)

    # Identify clumps in each image
    _, labeled_image1 = cv2.connectedComponents(image1_binary.astype(np.uint8))
    _, labeled_image2 = cv2.connectedComponents(image2_binary.astype(np.uint8))

    # Generate overlap mask
    overlap = cv2.bitwise_and(labeled_image1.astype(bool).astype(np.uint8), labeled_image2.astype(bool).astype(np.uint8)).astype(bool)

    # Preserve whole clump in image 1 that contains overlapping pixels with image 2
    preserved_image = np.zeros_like(image1)
    for label_value in np.unique(labeled_image1):
        if label_value == 0:  # Background
            continue
        if np.any(overlap[labeled_image1 == label_value]):
            preserved_image[labeled_image1 == label_value] = image1[labeled_image1 == label_value]
    preserved_image[preserved_image == 0] = np.nan

    return preserved_image

# This function shrinks a 10m*10m image to a 30m*30m image
def shrink_image(input_image):
    # Pad the image with NaN values to ensure dimensions are divisible by 3
    rows, cols = input_image.shape
    padded_rows = (rows + 2) // 3 * 3  # Calculate the padded rows
    padded_cols = (cols + 2) // 3 * 3  # Calculate the padded cols
    padded_image = np.full((padded_rows, padded_cols), np.nan)
    padded_image[:rows, :cols] = input_image
    
    # Resize the padded image to 1/3 of its original size
    resized_image = cv2.resize(padded_image, (cols // 3, rows // 3), interpolation=cv2.INTER_LINEAR)
    
    # Replace NaN values in the resized image with NaN values again
    resized_image[np.isnan(resized_image)] = np.nan
    
    return resized_image

# This function converts an original NetCDF file to a GeoTIFF file
def nc2tif(directory_path, outputpath, flight, file_name):
    output_path = outputpath + flight + '/Output_wavelet/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Output directory '{output_path}' created.")

    if os.path.isfile(os.path.join(directory_path, file_name)) and file_name.endswith('_10m.nc'):
        # Print the file name
        print("File name:", file_name)

        # Read the file
        nc_file = Dataset(os.path.join(directory_path, file_name), 'r')

        # Read data
        lat = nc_file.variables['lat'][:]
        lon = nc_file.variables['lon'][:]
        xch4 = nc_file.variables['xch4'][:]
        num_samples = nc_file.variables['num_samples'][:]

        # Replace certain values with NaN
        xch4 = np.where(xch4 > 1e35, np.nan, xch4)
        xch4 = np.where(num_samples < 0.7, np.nan, xch4)

        # Reverse the longitude grid if needed
        if lat[0] < lat[-1]:
            print('Flip the data array vertically')
            lat = lat[::-1]
            xch4 = np.flip(xch4, axis=0)

        # Calculate the geotransform parameters (assuming regular grid)
            lon_min, lon_max = lon.min(), lon.max()
            lat_min, lat_max = lat.min(), lat.max()
            pixel_width = (lon_max - lon_min) / len(lon)
            pixel_height = (lat_max - lat_min) / len(lat)
            transform = from_origin(lon_min, lat_max, pixel_width, pixel_height)

            # Create a GeoTIFF file
            tif_file = output_path + '/' + flight + '.tif'
            with rasterio.open(tif_file, 'w', driver='GTiff', width=len(lon), height=len(lat),
                            count=1, dtype=xch4.dtype, crs='EPSG:4326', transform=transform) as dst:
                dst.write(xch4, 1)
            print("GeoTIFF file created:", tif_file)

            # Close the NetCDF file
            nc_file.close()
        
        return tif_file

# This function converts a point index location in a GeoTIFF image to the coordinate location
def index_2_latlon(geotransform, x, y):
    lon = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    lat = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    
    return lon, lat

# This function converts a coordinate location in a GeoTIFF image to the point index location
def latlon_2_index(geotransform, lat, lon):
    y = round((lon - geotransform[0]) / geotransform[1])
    x = round((lat - geotransform[3]) / geotransform[5])

    return x, y

# This function converts a labeled image into GeoJSON polygons
def labeled_image_to_geojson(num_labels, labels, geotransform):
    features = []
    for label in range(1, num_labels):  # Exclude background label 0
        mask = labels == label
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = contours[0]  # Assume one contour per label
            contour_points = contour.squeeze()
            coordinates = [(index_2_latlon(geotransform, point[0], point[1])) for point in contour_points]
            geometry = geojson.Polygon([coordinates])
            properties = {"label": label}
            feature = geojson.Feature(geometry=geometry, properties=properties)
            features.append(feature)
    
    return geojson.FeatureCollection(features)

# This function fills the NAN holes within the plume masks
# masked_image_geotransform: (min_lon, max_lat, rows, cols)
def fill_nan_within_mask(masked_image, masked_image_geotransform, original_file):
    # Crop the original image based on the geotransform info of the masked image
    image_dataset = gdal.Open(original_file, gdal.GA_ReadOnly)
    original_image = image_dataset.GetRasterBand(1).ReadAsArray()
    
    cols = image_dataset.RasterXSize
    rows = image_dataset.RasterYSize
    geo_transform = image_dataset.GetGeoTransform()

    min_x = geo_transform[0]
    max_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = -1 * geo_transform[5]
    
    min_x_crop, max_y_crop, crop_rows, crop_cols = masked_image_geotransform
    
    start_x = round((min_x_crop - min_x) / pixel_width)
    start_y = round((max_y - max_y_crop) / pixel_height)
    width, height = crop_cols, crop_rows
    
    start_x = 0 if start_x < 0 else start_x
    start_y = 0 if start_y < 0 else start_y
    if start_y+height > rows:
        height -= (start_y+height - rows)
    if start_x+width > cols:
        width -= (start_x+width - cols)
    
    sub_original_image = original_image[start_y:start_y+height, start_x:start_x+width]

    # Create polygons based on the masked image
    masked_image_binary = ~np.isnan(masked_image).astype(bool)
    num_labels, labels = cv2.connectedComponents(masked_image_binary.astype(np.uint8))
    polygons_data = labeled_image_to_geojson(num_labels, labels, geo_transform)
    polygons = [shape(feature['geometry']) for feature in polygons_data['features']]
    
    # Recreate new masks based on polygon boundaries
    with rasterio.open(original_file) as src:
        original_image = src.read(1)
    polygon_mask = geometry_mask(polygons, out_shape=sub_original_image.shape, transform=src.transform, invert=True)
    sub_original_image[~polygon_mask] = np.nan
    
    # If there are NANs within the masks that are also in the original image, preserve them
    new_mask = np.isnan(sub_original_image) & (~np.isnan(masked_image))
    sub_original_image[new_mask] = masked_image[new_mask]

    return sub_original_image

# Define a function to find the closest point to a polygon and calculate the distance
def closest_point_to_polygon(polygon, points):
    min_distance = float('inf')
    closest_point = None
    for point in points:
        dist = polygon.distance(point)
        if dist < min_distance:
            min_distance = dist
            closest_point = point
    return min_distance, closest_point

# Define a function to find the closest polygon to a point and calculate the distance
def closest_polygon_to_point(polygons, point):
    min_distance = float('inf')
    closest_point = None
    closest_index = 0
    for polygon_index, polygon in enumerate(polygons):
        dist = polygon.distance(point)
        if dist < min_distance:
            min_distance = dist
            closest_polygon = polygon
            closest_index = polygon_index
    return min_distance, closest_index, closest_polygon

# Define a function to find the distance between a point and its farthest polygon vertex 
def farthest_dist_point_polygon(polygon, point):
    distances_to_vertices = [point.distance(Point(vertex)) for vertex in polygon.exterior.coords]
    farthest_distance = max(distances_to_vertices)
    return farthest_distance

# Define a function to find the closest point to a point and calculate the distance
def closest_point_to_point(points, fixed_point):
    min_distance = float('inf')
    closest_point = None
    for (point_x, point_y) in points:
        point = Point(point_x, point_y)
        dist = point.distance(fixed_point)
        if dist < min_distance:
            min_distance = dist
            closest_point = point
    return min_distance, closest_point

# Define a function to find all the points within a certain distance to a polygon
def all_points_to_polygon(polygon, points, dist_thred):
    points_list = []
    for point in points:
        dist = polygon.distance(point)
        if dist < dist_thred:
            points_list.append((point.x, point.y))
    return points_list
