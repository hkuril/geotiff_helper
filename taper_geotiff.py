# Import standard-library modules.
import argparse

# Import third-party modules.
import numpy as np
from osgeo import gdal

# Import modules from this repo.
from handling_geotiff import load_geotiff, save_geotiff

def get_pixel_distance_to_nearest_edge(n_x, n_y, x_indices, y_indices):
    '''
    For an array of shape (n_x, n_y), returns an integer array with the same shape where the values are the distances (in pixels) to the nearest edge, taking either a horizontal or vertical path (no diagonal distances).
    '''

    # Find the distance from each pixel to the nearest horizontal edge.
    distance_from_left_edge = x_indices
    distance_from_right_edge = n_x - 1 - x_indices
    distance_from_horz_edge = np.min([distance_from_left_edge,
                                      distance_from_right_edge], axis = 0)

    # Find the distance from each pixel to the nearest vertical edge.
    distance_from_top_edge = y_indices
    distance_from_bottom_edge = n_y - 1 - y_indices
    distance_from_vert_edge = np.min([distance_from_top_edge,
                                      distance_from_bottom_edge], axis = 0)
     
    # Find the distance from each pixel to the nearest edge.
    distance_from_edge = np.min([distance_from_horz_edge,
                                      distance_from_vert_edge], axis = 0)
    
    return distance_from_edge

def get_indices_of_nearest_nontapered_pixel(n_x, n_y, x_indices, y_indices, taper_width):
    '''
    For each pixel, if that pixel is within `taper_width` of the edge, find the indices of the nearest pixel which is not within `taper_width` of the edge.
    '''

    # Set the indices which define the left and right margins (I1, I2) and the
    # bottom and top margins (J1, J2).
    I1 = taper_width
    I2 = n_x - 1 - taper_width
    J1 = taper_width
    J2 = n_y - 1 - taper_width

    # Divide the pixels into nine regions:
    # top_lft   top_mid     top_rgt
    # mid_lft   mid_mid     mid_rgt
    # bot_lft   bot_mid     bot_rgt
    # Where the widths of the outer eight regions are given by `taper_width`.
    i_lt_I1 = (x_indices < I1)
    i_gt_I2 = (x_indices > I2)

    j_lt_J1 = (y_indices < J1)
    j_gt_J2 = (y_indices > J2)
    
    lft_col = i_lt_I1
    mid_col = ~i_lt_I1 & ~i_gt_I2
    rgt_col = i_gt_I2

    top_row = j_lt_J1
    mid_row = ~j_lt_J1 & ~j_gt_J2
    bot_row = j_gt_J2

    top_lft = top_row & lft_col
    mid_lft = mid_row & lft_col
    bot_lft = bot_row & lft_col

    top_mid = top_row & mid_col
    mid_mid = mid_row & mid_col
    bot_mid = bot_row & mid_col

    top_rgt = top_row & rgt_col
    mid_rgt = mid_row & rgt_col
    bot_rgt = bot_row & rgt_col

    # Prepare the output array.
    nearest_i_index = np.zeros((n_x, n_y), dtype = np.int64)
    nearest_j_index = np.zeros((n_x, n_y), dtype = np.int64)

    # Define lists of indices increasing along the edges of the non-tapered
    # region.
    I_mid_span = np.array(range(I1, I2 + 1), dtype = np.int64) 
    J_mid_span = np.array(range(J1, J2 + 1), dtype = np.int64) 
    
    # Assign the nearest indices to the margin regions.
    # Top row.
    nearest_i_index[top_lft] = I1
    nearest_j_index[top_lft] = J1
    #
    nearest_i_index[top_mid] = np.tile(I_mid_span, (taper_width, 1)).T.flatten()
    nearest_j_index[top_mid] = J1
    #
    nearest_i_index[top_rgt] = I2 
    nearest_j_index[top_rgt] = J1
    # 
    # Middle row.
    nearest_i_index[mid_lft] = I1
    nearest_j_index[mid_lft] = np.tile(J_mid_span, (1, taper_width)).flatten()
    #
    nearest_i_index[mid_mid] = 0
    nearest_j_index[mid_mid] = 0 
    #
    nearest_i_index[mid_rgt] = I2 
    nearest_j_index[mid_rgt] = np.tile(J_mid_span, (1, taper_width)).flatten()
    #
    # Bottom row.
    nearest_i_index[bot_lft] = I1
    nearest_j_index[bot_lft] = J2
    #
    nearest_i_index[bot_mid] = np.tile(I_mid_span, (taper_width, 1)).T.flatten()
    nearest_j_index[bot_mid] = J2
    #
    nearest_i_index[bot_rgt] = I2 
    nearest_j_index[bot_rgt] = J2

    return nearest_i_index, nearest_j_index

def taper_func(x, x_edge, width, dist):
    '''
    A function defining how the elevation is tapered between the non-tapered region and the outer edge.
    '''

    # The outer fraction of the tapered region is flat.
    sub_width = 0.1

    # f is the fractional distance of the given pixel between the edge and
    # the non-tapered region.
    f = (dist - width) / width + 1.0

    # The flat region around the very edge.
    if f < sub_width:

        x_new = x_edge

    # The tapered region, increasing linearly from the flat region inwards.
    else:

        f_adjusted = (f - sub_width) / (1.0 - sub_width)

        d_x = x - x_edge
        x_new = x_edge + (d_x * f_adjusted)

    return x_new

def main():

    # Parsing command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_geotiff_in", help = "File path to input GeoTIFF file.")
    parser.add_argument("path_geotiff_out", help = "File path to output GeoTIFF file.")
    parser.add_argument("taper_width_px", help = "Width of taper zone in pixels (must be less than half the width and height of GeoTIFF.")
    parser.add_argument("taper_depth", help = "The amount by which the outermost value of the tapered GeoTIFF is lower than the lowest point in the original GeoTIFF (in the units of the original GeoTIFF.")
    #
    args = parser.parse_args()
    #
    path_geotiff_in  = args.path_geotiff_in
    path_geotiff_out = args.path_geotiff_out
    taper_width      = int(args.taper_width_px)
    taper_depth      = int(args.taper_depth)

    # Load the geotiff.
    data_object, data_array, data_transform, proj_wkt = \
        load_geotiff(path_geotiff_in)

    # Lowest value
    if taper_depth > 0:

        data_min = np.min(data_array)
        taper_edge_value = data_min - taper_depth

    else:

        data_max = np.max(data_array)
        taper_edge_value = data_max - taper_depth
    
    # Determine the shape of the geotiff.
    n_x, n_y = data_array.shape

    # Get the indices of each pixel.
    x_indices = np.tile(range(n_x), (n_y, 1)).T
    y_indices = np.tile(range(n_y), (n_x, 1))
    
    # Find the distance from each pixel to the nearest edge.
    distance_from_edge = get_pixel_distance_to_nearest_edge(
                            n_x, n_y, x_indices, y_indices)
    
    # For the pixels in the tapered region, find the indices of the nearest
    # non-tapered pixel.
    nearest_i_index, nearest_j_index = \
        get_indices_of_nearest_nontapered_pixel(
            n_x, n_y, x_indices, y_indices, taper_width)

    # Prepare the output array.
    tapered_array = np.zeros((n_x, n_y))
    
    # Loop over pixels, and taper those which are within the taper width of
    # the edge.
    for i in range(n_x):

        for j in range(n_y):

            # Case 1: The pixel is not close to the edge.
            if distance_from_edge[i, j] >= taper_width:
                
                # In this case, the pixel is not changed.
                tapered_array[i, j] = data_array[i, j]

            # Case 2: The pixel is close to the edge.
            else:

                # In this case, the pixel is assigned a new value, depending
                # on
                # 1) the value of its nearest non-tapered pixel;
                # 2) the distance from the edge;
                # 3) the target value on the edge ('taper_depth'); and
                # 4) the choice of taper function.
                ii = nearest_i_index[i, j]
                jj = nearest_j_index[i, j]
                ref_value = data_array[ii, jj]
                tapered_array[i, j] = taper_func(ref_value, taper_edge_value, 
                                        taper_width, distance_from_edge[i, j])
    
    # Save the tapered geotiff.
    data_type = data_object.GetRasterBand(1).DataType
    save_geotiff(path_geotiff_out, tapered_array, proj_wkt, data_transform,
            data_type)

    return

if __name__ == '__main__':

    main()
