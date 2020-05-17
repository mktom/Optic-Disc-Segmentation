#COMP9517 Assignment 2 - Individual Component 
#Mark Thomas z5194597

import cv2, os, numpy, sys, random, collections, math, time
import matplotlib.pyplot as pypl
from PIL import Image

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.filters import prewitt, unsharp_mask
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.draw import circle_perimeter
from skimage import exposure
from skimage.feature import canny
from skimage.morphology import erosion, closing, label

file_seperator = ["\\\\" if sys.platform == "win32" else '/'][0]
file_path = file_seperator.join(os.getcwd().split(file_seperator))
working_directory = file_path + file_seperator + "Data_Individual_Component"
random.seed()
rescaled_height, rescaled_length = 470, 707

(os.mkdir(file_seperator.join([working_directory, "result_optic_discs"])) 
            if not os.path.isdir(file_seperator.join([working_directory, "result_optic_discs"])) else None)

image_files = os.listdir(file_seperator.join([working_directory, "original_retinal_images"]))
image_rgb = [cv2.cvtColor(cv2.imread(file_seperator.join([working_directory, "original_retinal_images", image]), 1), cv2.COLOR_BGR2RGB) for image in image_files]
image_averaged = [cv2.blur(image, (6,6)) for image in image_rgb] #Mean Filter - 6x6 Window

mask_files = os.listdir(file_seperator.join([working_directory, "optic_disc_segmentation_masks"]))
mask_grey = [cv2.cvtColor(cv2.imread(file_seperator.join([working_directory, "optic_disc_segmentation_masks", image]), 1), cv2.COLOR_BGR2GRAY) for image in mask_files]
mask_resize = [cv2.resize(image, (rescaled_length, rescaled_height)) for image in mask_grey]

#Mean histogram - Regular window 485x485: Rescaled to fit 80x80 window by dividing by 6 to dimensions(707, 470)
image_resized = [cv2.resize(image, (rescaled_length, rescaled_height)) for image in image_averaged]

#vrtns Folder - Convert images to grey-scale then check if > 127 to get brightest and darkest images
#To get mid-range find images close to mean brightness of images
image_grey = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in image_averaged] #Grey-scale for comparision
#Images with pixel intensities larger than 127 are treated as brighter
variation_dictionary = collections.OrderedDict((files, numpy.where(image > 127)[0].shape[0]) for files, image in list(zip([i for i in range(len(image_files))], image_grey))) #Store all indexes of image_grey as indexes and the count of pixels > 127 as values
sorted_dictionary = collections.OrderedDict((key, variation_dictionary[key]) for key in sorted(variation_dictionary, key=variation_dictionary.get)) # Sorted dictionary so that the darkest images are at the start and the lightest at the end

average_count = numpy.mean(list(sorted_dictionary.values())) #Find average brightness images
average_index = numpy.argsort(numpy.abs(numpy.array(list(sorted_dictionary.values())) - average_count), axis=0)[:4] #Indicies of the dictionary.values list for the first four images closest to the mean
medium_index = [key for key in sorted_dictionary.keys() for value in list(numpy.array(list(sorted_dictionary.values()))[average_index]) if sorted_dictionary[key] == value] #Indicies for the medium images in image_averaged
dark_index, light_index = (list(sorted_dictionary.keys())[:4], list(sorted_dictionary.keys())[-4:]) #Indicies for the both dark and light images in image_averaged

dark_images, medium_images, light_images = ([image_averaged[id] for id in dark_index], [image_averaged[id] for id in medium_index], [image_averaged[id] for id in light_index]) # The dark, medium and light images

#Manually set 80x80 window for optic disc in Images 2, 11, 25, 35 for dark
dark_files = [files for files in os.listdir(file_seperator.join([working_directory, "80x80_opt_dsc\\vrtns\\drk\\temp"])) if not files.endswith(".png")]
dark_rgb = [cv2.cvtColor(cv2.imread(file_seperator.join([working_directory, "80x80_opt_dsc\\vrtns\\drk\\temp", im]), 1), cv2.COLOR_BGR2RGB) for im in dark_files]
dark_mean_image = numpy.mean(dark_rgb, axis=0)
                                                                                        
#Manually set 80x80 window for optic disc in Images 18, 22, 41, 52 for medium
medium_files = [files for files in os.listdir(file_seperator.join([working_directory, "80x80_opt_dsc\\vrtns\\mdm\\temp"])) if not files.endswith(".png")] 
medium_rgb = [cv2.cvtColor(cv2.imread(file_seperator.join([working_directory, "80x80_opt_dsc\\vrtns\\mdm\\temp", im]), 1), cv2.COLOR_BGR2RGB) for im in medium_files]
medium_mean_image = numpy.mean(medium_rgb, axis=0)

#Manually set 80x80 window for optic disc in Images 8, 16, 46, 53 for light
light_files = [files for files in os.listdir(file_seperator.join([working_directory, "80x80_opt_dsc\\vrtns\\lht\\temp"])) if not files.endswith(".png")] 
light_rgb = [cv2.cvtColor(cv2.imread(file_seperator.join([working_directory, "80x80_opt_dsc\\vrtns\\lht\\temp", im]), 1), cv2.COLOR_BGR2RGB) for im in light_files]
light_mean_image = numpy.mean(light_rgb, axis=0)

dark_histogram_red, dark_histogram_green, dark_histogram_blue = [numpy.unique(numpy.array(dark_mean_image[:, :, i], numpy.uint8), return_counts=True) for i in range(3)] #Unique values and counts for all 3 channels in the dark_mean_image
dark_array_red, dark_array_green, dark_array_blue = [numpy.zeros((256, ), numpy.int32) for _ in range(3)] #Numpy array to store all counts for values 0 - 255 for dark_mean_image
dark_array_red[dark_histogram_red[0]], dark_array_green[dark_histogram_green[0]], dark_array_blue[dark_histogram_blue[0]] = dark_histogram_red[1], dark_histogram_green[1], dark_histogram_blue[1] #Setting all non-zero values in the respective R, G, B arrays for dark_mean_image

medium_histogram_red, medium_histogram_green, medium_histogram_blue = [numpy.unique(numpy.array(medium_mean_image[:, :, i], numpy.uint8), return_counts=True) for i in range(3)] #Unique values and counts for all 3 channels in the medium_mean_image
medium_array_red, medium_array_green, medium_array_blue = [numpy.zeros((256, ), numpy.int32) for _ in range(3)] #Numpy array to store all counts for values 0 - 255 for medium_mean_image
medium_array_red[medium_histogram_red[0]], medium_array_green[medium_histogram_green[0]], medium_array_blue[medium_histogram_blue[0]] = medium_histogram_red[1], medium_histogram_green[1], medium_histogram_blue[1] #Setting all non-zero values in the respective R, G, B arrays for medium_mean_image

light_histogram_red, light_histogram_green, light_histogram_blue = [numpy.unique(numpy.array(light_mean_image[:, :, i], numpy.uint8), return_counts=True) for i in range(3)] #Unique values and counts for all 3 channels in the light_mean_image
light_array_red, light_array_green, light_array_blue = [numpy.zeros((256, ), numpy.int32) for _ in range(3)] #Numpy array to store all counts for values 0 - 255 for light_mean_image
light_array_red[light_histogram_red[0]], light_array_green[light_histogram_green[0]], light_array_blue[light_histogram_blue[0]] = light_histogram_red[1], light_histogram_green[1], light_histogram_blue[1] #Setting all non-zero values in the respective R, G, B arrays for light_mean_image

def optic_disc_localize(original_image, histograms, window_size): 
    depth = window_size // 2
    histogram_red, histogram_green, histogram_blue = histograms
    copied = numpy.zeros((original_image.shape[:2]), numpy.float32)
    for row_value, column_value in numpy.ndindex(original_image.shape[:2]): #Iterate through the array to get indices
            current_neighbourhood = original_image[max((row_value - depth), 0) : min((row_value + depth), original_image.shape[0]), max((column_value - depth), 0) : min((column_value + depth), original_image.shape[1])] #Set 80x80 window in the image at C(i, j)
            #Following process is to obtain the correlation coefficent:
            current_histogram_red, current_histogram_green, current_histogram_blue = [numpy.unique(current_neighbourhood[:, :, j], return_counts=True) for j in range(3)] #Unique values and counts for all 3 channels in the current window
            current_array_red, current_array_green, current_b = [numpy.zeros((256, ), numpy.uint32) for _ in range(3)] #Numpy array to store all counts for values 0 - 255 in the current window
            current_array_red[current_histogram_red[0]], current_array_green[current_histogram_green[0]], current_b[current_histogram_blue[0]] = current_histogram_red[1], current_histogram_green[1], current_histogram_blue[1] #Setting all non-zero values in the respective R, G, B arrays in the current window
            c_r, c_g, c_b = [(1 /(1 + ((current_array_red - histogram_red) ** 2).sum())), (1 /(1 + ((current_array_green - histogram_green) ** 2).sum())), (1 /(1 + ((current_b - histogram_blue) ** 2).sum()))] # coefficents for the R, G, B channels calcualted without excluding any values
            copied[row_value, column_value] = (0.5 * c_r) + (2 * c_g) + (1 * c_b) # Setting the centre pixel value
    return copied

dark_copied_image = dark_mean_image.copy()
dark_copied_image[:, :, 0][dark_copied_image[:, :, 0] < 200] = 0
dark_copied_image[:, :, 1][dark_copied_image[:, :, 1] > 50] = 0
dark_copied_image[:, :, 2][dark_copied_image[:, :, 2] < 4] = 0

medium_copied_image = medium_mean_image.copy()
medium_copied_image[:, :, 0][medium_copied_image[:, :, 0] > 175] = 0
medium_copied_image[:, :, 1][medium_copied_image[:, :, 1] > 75] = 0
medium_copied_image[:, :, 2][medium_copied_image[:, :, 2] > 50] = 0

light_copied_image = light_mean_image.copy()
light_copied_image[:, :, 0][light_copied_image[:, :, 0] < 250] = 0
light_copied_image[:, :, 1][light_copied_image[:, :, 1] < 150] = 0
light_copied_image[:, :, 2][light_copied_image[:, :, 2] < 77] = 0

dark_histogram_red, dark_histogram_green, dark_histogram_blue = [numpy.unique(numpy.array(dark_copied_image[:, :, i], numpy.uint8), return_counts=True) for i in range(3)]
dark_array_red, dark_array_green, dark_array_blue = [numpy.zeros((256, ), numpy.int32) for _ in range(3)] 
dark_array_red[dark_histogram_red[0][1:]], dark_array_green[dark_histogram_green[0][1:]], dark_array_blue[dark_histogram_blue[0][1:]] = dark_histogram_red[1][1:], dark_histogram_green[1][1:], dark_histogram_blue[1][1:]

medium_histogram_red, medium_histogram_green, medium_histogram_blue = [numpy.unique(numpy.array(medium_copied_image[:, :, i], numpy.uint8), return_counts=True) for i in range(3)]
medium_array_red, medium_array_green, medium_array_blue = [numpy.zeros((256, ), numpy.int32) for _ in range(3)] 
medium_array_red[medium_histogram_red[0][1:]], medium_array_green[medium_histogram_green[0][1:]], medium_array_blue[medium_histogram_blue[0][1:]] = medium_histogram_red[1][1:], medium_histogram_green[1][1:], medium_histogram_blue[1][1:]

light_histogram_red, light_histogram_green, light_histogram_blue = [numpy.unique(numpy.array(light_copied_image[:, :, i], numpy.uint8), return_counts=True) for i in range(3)]
light_array_red, light_array_green, light_array_blue = [numpy.zeros((256, ), numpy.int32) for _ in range(3)] 
light_array_red[light_histogram_red[0][1:]], light_array_green[light_histogram_green[0][1:]], light_array_blue[light_histogram_blue[0][1:]] = light_histogram_red[1][1:], light_histogram_green[1][1:], light_histogram_blue[1][1:]

full_dark_mean = numpy.mean(dark_images, axis=0)
full_medium_mean = numpy.mean(medium_images, axis=0)
full_light_mean = numpy.mean(light_images, axis=0) 

temporary_dictionary = {"dark" : (full_dark_mean, (dark_array_red, dark_array_green, dark_array_blue)), "medium" : (full_medium_mean, (medium_array_red, medium_array_green, medium_array_blue)), "light" : (full_light_mean, (light_array_red, light_array_green, light_array_blue))}
temporary_index = [numpy.argmin([numpy.abs(numpy.count_nonzero(cv2.cvtColor(values[0].astype(numpy.uint8), cv2.COLOR_RGB2GRAY) > 127) - numpy.where(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 127)[0].shape[0]) for values in list(temporary_dictionary.values())]) for image in image_averaged]
temporary_values = [list(temporary_dictionary.values())[idx][1] for idx in temporary_index]
# copied_images = [optic_disc_localize(oimage, hst, 80) for oimage, hst in list(zip(image_resized, temporary_values))]
copied_images = [cv2.imread(f"Data_Individual_Component\\result_optic_discs\\cpy_imgs\\Correlated_Img_{i}.jpg", 0) for i in range(1, 55)]

def number_of_connected_components_min(image, threshold_factor):
    threshold = threshold_factor * image.max()    
    thresholded_image = image.copy()               
    thresholded_image[image > threshold], thresholded_image[image < threshold] = 255, 0
    shapes, labels = cv2.connectedComponents(thresholded_image.astype(numpy.uint8)) 
    if threshold_factor < 0.95:
        if shapes != 2:                                              
            threshold_factor += 0.05
            threshold_factor, thresholded_image, labels = number_of_connected_components_min(image, threshold_factor)
    return (threshold_factor, thresholded_image, labels) 

def number_of_connected_components_max(image, threshold_factor):
    threshold = threshold_factor * image.max()    
    thresholded_image = image.copy()               
    thresholded_image[image > threshold], thresholded_image[image < threshold] = 255, 0
    shapes, labels = cv2.connectedComponents(thresholded_image.astype(numpy.uint8)) 
    if threshold_factor < 0.95:
        if shapes != 1:                                              
            threshold_factor += 0.05
            threshold_factor, thresholded_image, labels = number_of_connected_components_max(image, threshold_factor)
    return (threshold_factor, thresholded_image, labels) 

def final_threshold(image, min_threshold_factor, max_threshold_factor, portion_of_range):
    new_threshold_factor = min_threshold_factor + ((max_threshold_factor - min_threshold_factor) * portion_of_range)
    threshold = new_threshold_factor * image.max()    
    thresholded_image = image.copy()               
    thresholded_image[image > threshold], thresholded_image[image < threshold] = 255, 0
    return (new_threshold_factor, thresholded_image)

threshold_images_min = [number_of_connected_components_min(cp_im[50:425, :], 0.3) for cp_im in copied_images] 
threshold_images_max = [number_of_connected_components_max(cp_im[50:425, :], 0.3) for cp_im in copied_images]
threshold_images = [final_threshold(image[50:425, :], min_values[0], max_values[0], 0.5) for image, min_values, max_values in list(zip(copied_images, threshold_images_min, threshold_images_max))]

def largest_circular_component(thresholded_image):
    number_of_shapes, labeled_image = cv2.connectedComponents(thresholded_image.astype(numpy.uint8))
    largest_circles_in_each_shape = []
    for shape in range(1, number_of_shapes):    
        row_array, column_array = numpy.where(labeled_image == shape)
        if row_array.shape[0] >= 4 or column_array.shape[0] >= 4:
            normalized_rows = row_array - min(row_array)
            normalized_columns = column_array - min(column_array)
            row_length = max(row_array) - min(row_array)
            column_length = max(column_array) - min(column_array)
            mask = numpy.zeros((row_length + 1, column_length + 1), numpy.uint8)        
            mask[normalized_rows, normalized_columns] = 255
            radius_min = numpy.min([row_length / 2, column_length / 2]) / 2
            radius_max = numpy.max([row_length / 2, column_length / 2])
            if radius_min >= 1 and radius_max >= 1: 
                radius = numpy.arange(int(radius_min), int(radius_max))
                circles  = hough_circle(mask, radius)
                acummalator, coordinates_x, coordinates_y, detected_radii = hough_circle_peaks(circles, radius, total_num_peaks=1)
                largest_circles_in_each_shape.append(numpy.hstack((acummalator, coordinates_x, coordinates_y, column_array.min(), row_array.min(), detected_radii)))
    largest_circle_array = numpy.array(largest_circles_in_each_shape, numpy.float16)
    accumalator_index = numpy.argmax(largest_circle_array[:, 0])
    accumalator_all_max = numpy.where(largest_circle_array[:, 0] == largest_circle_array[accumalator_index][0])
    radii_max = numpy.max(largest_circle_array[accumalator_all_max][:, -1])
    radii_variables = largest_circle_array[largest_circle_array[:, -1] == radii_max][0] 
    return radii_variables


def optic_disc_window(coordinates, original_image):
    centre_x, centre_y, depth = int(coordinates[1:5][0::2].sum()), int(coordinates[1:5][1::2].sum()) + 50, 250 // 2
    optic_window = original_image[max((centre_y - depth), 0) : min((centre_y + depth), original_image.shape[0]), max((centre_x - depth), 0) : min((centre_x + depth), original_image.shape[1])]
    return optic_window

radius_images = [largest_circular_component(thresh_image[1]) for thresh_image in threshold_images]
optic_windows = [optic_disc_window(coords, image) for coords, image in list(zip(radius_images, image_resized))]

vessel_element = numpy.full((1, 37), 127).astype(numpy.float32)

def rotate_image(grey_image, degree):      
    y_length, x_length = grey_image.shape 
    compliment_degree = 90 - degree     
    new_x_length = math.fabs(x_length * (math.cos(math.radians(degree)))) + math.fabs(y_length * (math.cos(math.radians(compliment_degree))))
    new_y_length = math.fabs(x_length * (math.sin(math.radians(degree)))) + math.fabs(y_length * (math.sin(math.radians(compliment_degree))))
    rotation_matrix = cv2.getRotationMatrix2D((new_x_length / 2, new_y_length / 2), degree, 1)
    translation_matrix = numpy.dot(rotation_matrix, numpy.array([(new_x_length - x_length) / 2, (new_y_length - y_length)/2, 0]))
    rotation_matrix[0, 2] += translation_matrix[0] 
    rotation_matrix[1, 2] += translation_matrix[1]
    rotated_image = cv2.warpAffine(grey_image, rotation_matrix, (math.ceil(new_x_length), math.ceil(new_y_length)))
    rotated_image[rotated_image != 0] = 127
    return rotated_image

grey_elements = [vessel_element] + [rotate_image(vessel_element, iteration_degree) for iteration_degree in range(15, (12*15) + 1, 15)]

def blood_vessel_removal(rgb_image, grey_array):
    channel_image = numpy.zeros((rgb_image.shape), numpy.uint8)
    for row_value, column_value in numpy.ndindex(rgb_image.shape):
        grey_dictionary = collections.OrderedDict([(key, (0, 0)) for key in range(0, (12*15) + 1, 15)])
        for array_index in range(len(grey_elements)):
                grey_row, grey_column = numpy.ceil(numpy.array(grey_elements[array_index].shape) * 0.5).astype(numpy.uint8)
                grey_neighbourhood = rgb_image[max((row_value - grey_row), 0) : min((row_value + grey_row), rgb_image.shape[0]), max((column_value - grey_column), 0) : min((column_value + grey_column), rgb_image.shape[1])]       
                image_neighbourhood = cv2.copyMakeBorder(grey_neighbourhood.copy(), [grey_row if (row_value - grey_row) < 0 else 0][0], [grey_row if (row_value + grey_row) > rgb_image.shape[0] else 0][0], [grey_column if (column_value - grey_column) < 0 else 0][0], [grey_column if (column_value + grey_column) > rgb_image.shape[1] else 0][0], cv2.BORDER_CONSTANT, value=0)
                grey_coordinates = numpy.where(grey_elements[array_index] == 127)
                values = image_neighbourhood[grey_coordinates]
                grey_dictionary[array_index*15] = ((values.max() - values.min()), values.max())
        channel_image[row_value, column_value] = max(list(grey_dictionary.values()), key=lambda x:x[0])[1]        
    return channel_image

red_images = [blood_vessel_removal(image[:, :, 0], grey_elements) for image in optic_windows] 
green_images = [blood_vessel_removal(image[:, :, 1], grey_elements) for image in optic_windows]

red_histogram_equalized = [exposure.equalize_hist(image[29:-30, 29:-30]) for image in red_images]
green_histogram_equalized = [exposure.equalize_hist(image[29:-30, 29:-30]) for image in green_images]
green_sharpened = [unsharp_mask(image, 50, 1) for image in green_histogram_equalized]

red_canny_edges = [canny(image, 0.5, 0.05, 0.5) for image in red_histogram_equalized]
green_canny_edges = [canny(image, 0.5, 0.05, 0.35) for image in green_sharpened]

red_canny_edges = [image.astype(numpy.uint8) for image in red_canny_edges]     
green_canny_edges = [image.astype(numpy.uint8) for image in green_canny_edges] 

for image in red_canny_edges:
    image[image == 0], image[image == 1] = 255, 0 

for image in green_canny_edges:
    image[image == 0], image[image == 1] = 255, 0 

def circular_kernel(radius):
    circular_mask = numpy.zeros((2*radius+1, 2*radius+1), numpy.uint8)
    y_array, x_array = numpy.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x_array**2 + y_array**2 <= radius**2 
    circular_mask[mask] = 1
    return circular_mask

red_erode = [erosion(image, circular_kernel(5 // 2)) for image in red_canny_edges] 
green_erode = [erosion(image, circular_kernel(5 // 2)) for image in green_canny_edges] 

red_close = [closing(image, circular_kernel(3 // 2)) for image in red_erode]
green_close = [closing(image, circular_kernel(3 // 2)) for image in green_erode]

def optic_disc_component(closed_image):
    row_coordinate_min, row_coordinate_max = (closed_image.shape[1] / 2) - 38, (closed_image.shape[1] / 2) + 38
    column_coordinate_min, column_coordinate_max = (closed_image.shape[1] / 2) - 38, (closed_image.shape[1] / 2) + 38
    number_of_shapes, labeled_image = cv2.connectedComponents(closed_image.astype(numpy.uint8))
    largest_circles_in_each_shape = []
    for shape in range(1, number_of_shapes):     
        row_array, column_array = numpy.where(labeled_image == shape)
        if row_array.shape[0] >= 4 or column_array.shape[0] >= 4:
            normalized_rows = row_array - min(row_array)
            normalized_columns = column_array - min(column_array)
            row_length = max(row_array) - min(row_array)
            column_length = max(column_array) - min(column_array)
            mask = numpy.zeros((row_length + 1, column_length + 1), numpy.uint8)        
            mask[normalized_rows, normalized_columns] = 255
            radius_min = numpy.min([row_length / 2, column_length / 2]) / 2
            radius_max = numpy.max([row_length / 2, column_length / 2])
            radius = numpy.arange(int(radius_min), int(radius_max))
            circles  = hough_circle(mask, radius)
            acummalator, coordinates_x, coordinates_y, detected_radii = hough_circle_peaks(circles, radius, total_num_peaks=1)
            if coordinates_x + column_array.min() >= column_coordinate_min and coordinates_x + column_array.min() <= column_coordinate_max and coordinates_y + row_array.min() >= row_coordinate_min and coordinates_y + row_array.min() <= row_coordinate_max:
                largest_circles_in_each_shape.append(numpy.hstack((acummalator, coordinates_x, coordinates_y, column_array.min(), row_array.min(), detected_radii)))
    largest_circle_array = numpy.array(largest_circles_in_each_shape, numpy.float16)
    accumalator_index = numpy.argmax(largest_circle_array[:, 0])
    accumalator_all_max = numpy.where(largest_circle_array[:, 0] == largest_circle_array[accumalator_index][0])
    radii_max = numpy.max(largest_circle_array[accumalator_all_max][:, -1])
    radii_variables = largest_circle_array[largest_circle_array[:, -1] == radii_max][0] 
    return radii_variables

def best_segmentation(radii_pair, red_green_images):
        best_circle = numpy.array(radii_pair, numpy.float16)
        accumalator_index = numpy.argmax(best_circle[:, 0])
        accumalator_all_max = numpy.where(best_circle[:, 0] == best_circle[accumalator_index][0])
        radii_max = numpy.max(best_circle[accumalator_all_max][:, -1])
        radii_index = numpy.where(best_circle[:, -1] == radii_max)
        radii_variables = best_circle[radii_index][0] 
        best_image = red_green_images[radii_index[0][0]]
        flood_mask = numpy.zeros(numpy.array(best_image.shape) + 2, numpy.uint8)
        coordinates = (int(radii_variables[1:5][0::2].sum()), int(radii_variables[1:5][1::2].sum()))
        final_image = cv2.floodFill(best_image.astype(numpy.uint8), flood_mask, coordinates, 255)
        return flood_mask[1:-1, 1:-1]

def final_mask(radii_parameters, flood_image, optic_window):
    orginal_optic_window = numpy.zeros((optic_window.shape[:2]), numpy.uint8)
    orginal_optic_window[29:-30, 29:-30] = flood_image
    optic_mask = numpy.zeros((rescaled_height, rescaled_length), numpy.uint8)
    centre_x, centre_y, depth = int(radii_parameters[1:5][0::2].sum()), int(radii_parameters[1:5][1::2].sum()) + 50, 250 // 2
    optic_mask[max((centre_y - depth), 0) : min((centre_y + depth), rescaled_height), max((centre_x - depth), 0) : min((centre_x + depth), rescaled_length)] = orginal_optic_window
    return optic_mask

red_radius = [optic_disc_component(image) for image in red_close]
green_radius = [optic_disc_component(image) for image in green_close]

window_masks = [best_segmentation(radii_pair, red_green_pair) for radii_pair, red_green_pair in list(zip(list(zip(red_radius, green_radius)), list(zip(red_close, green_close))))]
# mask_images = [final_mask(radius, flood, optic_mask) for radius, flood, optic_mask in list(zip(radius_images, window_masks, optic_windows))]

# for image in mask_images:
#     image[image == 1] = 255

# def recolor_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     image[:, :, 1], image[:, :, 2] = 0, 0
#     return image

# mask_rgb = [recolor_image(image) for image in mask_images]
mask_images = [cv2.imread(f"Data_Individual_Component\\result_optic_discs\\generated_masks\\Mask_{i}.jpg", 0) for i in range(1, 55)]

for image in mask_images:
    image[image != 0] = 1

for image in mask_resize:
    image[image != 0] = 1

jaccard_metric = [jaccard_score(gen_mask.flatten(), truth_mask.flatten()) for gen_mask, truth_mask in list(zip(mask_images, mask_resize))] 

# for image in range(len(mask_images)):
#     fig = pypl.figure(frameon=False, figsize=(707/120, 470/120), dpi=120)
#     ax = fig.add_axes([0, 0, 1,1])     
#     ax.axis("off"), fig.tight_layout()
#     ax.imshow(copied_images[image]), pypl.savefig(f"Data_Individual_Component\\result_optic_discs\\generated_masks\\Mask_{image + 1}.jpg", bbox_inches="tight", pad_inches=0, dpi=120)       
