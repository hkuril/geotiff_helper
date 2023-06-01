# Import standard-library modules.
import argparse

# Import third-party modules.
import numpy as np
from osgeo import gdal
from scipy.signal import fftconvolve

# Import modules from this repo.
from handling_geotiff import load_geotiff, save_geotiff

def gaussian_blur(in_array, size):
    '''
    Copied from
    https://gis.stackexchange.com/a/10467/215740
    '''

    # expand in_array to fit edge of kernel
    padded_array = np.pad(in_array, size, 'symmetric')
    print(padded_array)
    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    print(x)
    print(y)
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    print(g)
    g = (g / g.sum()).astype(in_array.dtype)
    print(g)

    # do the Gaussian blur
    return fftconvolve(padded_array, g, mode = 'valid')

def main():

    # Parsing command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_geotiff_in", help = "File path to input GeoTIFF file.")
    parser.add_argument("path_geotiff_out", help = "File path to output GeoTIFF file.")
    parser.add_argument("blur_size_px", help = "Half-size (in pixels) of the blurring kernel.")
    #
    args = parser.parse_args()
    #
    path_geotiff_in  = args.path_geotiff_in
    path_geotiff_out = args.path_geotiff_out
    blur_size_px = int(args.blur_size_px)

    # Load the geotiff.
    data_object, data_array, data_transform, proj_wkt = \
        load_geotiff(path_geotiff_in)


    print(np.min(data_array), np.max(data_array))

    
    # Blur the geotiff.
    data_array_blurred = gaussian_blur(data_array.astype(np.float), blur_size_px)
    data_array_blurred = (np.round(data_array_blurred)).astype(int)
    print(np.min(data_array_blurred), np.max(data_array_blurred))
    
    # Save the blurred geotiff.
    data_type = data_object.GetRasterBand(1).DataType
    save_geotiff(path_geotiff_out, data_array_blurred, proj_wkt, data_transform,
            data_type)

    return

if __name__ == '__main__':

    main()
