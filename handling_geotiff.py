import numpy as np
from osgeo import gdal
from scipy.signal import fftconvolve

def load_geotiff(path_geotiff):
    '''
    See https://drr.ikcest.org/tutorial/k8023 for data types.
    '''

    # Make GDAL raise exceptions when an error occurs (to be consistent
    # with default Python function behaviour).
    gdal.UseExceptions()

    # Read the data, the transform and the projection.
    data_object = gdal.Open(path_geotiff)
    data_array = data_object.ReadAsArray()
    data_transform = data_object.GetGeoTransform()
    # Projection is in WKT format.
    proj_wkt = data_object.GetProjection()

    return data_object, data_array, data_transform, proj_wkt

def save_geotiff(path_out, array, projection, geo_transform, datatype):

    rows, cols = array.shape
    print("Saving to {:}".format(path_out))

    driver = gdal.GetDriverByName("GTiff")
    print(datatype)
    outdata = driver.Create(path_out, cols, rows, 1, datatype)#, gdal.GDT_UInt16)
    print(outdata)

    outdata.SetGeoTransform(geo_transform) 
    outdata.SetProjection(projection)
    outdata.GetRasterBand(1).WriteArray(array)

    # Saves to disk.
    outdata.FlushCache()

    return

def get_geotiff_pixel_coords(shape, data_transform):

    #xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()
    #xoffset, px_w, rot1, yoffset, px_h, rot2
    #https://stackoverflow.com/a/50196761/6731244
    xoffset, px_w, rot1, yoffset, rot2, px_h = data_transform
    n_x, n_y = shape
    #n_x = data_object.RasterXSize
    #n_y = data_object.RasterYSize
    i = np.array(range(n_x + 1), dtype = np.int32)
    j = np.array(range(n_y + 1), dtype = np.int32)
    I, J = np.meshgrid(i, j)

    X_edge = (px_w * I) + (rot1 * J) + xoffset
    Y_edge = (rot2 * I) + (px_h * J) + yoffset

    X_centre = (X_edge[:-1, :-1] + X_edge[1:, 1:]) / 2.0
    Y_centre = (Y_edge[:-1, :-1] + Y_edge[1:, 1:]) / 2.0

    return X_centre, Y_centre, X_edge, Y_edge

def smooth_raster(path_geotiff_in, path_geotiff_out, blur_size, treat_water = 'remove_below'):

    assert treat_water in ['remove_below', 'remove_above']

    data_object, data_array, data_transform, proj_wkt = load_geotiff(path_geotiff_in)
    data_type = data_object.GetRasterBand(1).DataType
    
    if treat_water == 'remove_below':

        index_below_water = np.where(data_array < 0)
        data_array[index_below_water] = 0.0

    elif treat_water == 'remove_above':

        index_above_water = np.where(data_array >= 0)
        data_array[index_above_water] = 0.0

    array_float = data_array.astype(np.float)
    data_blurred = gaussian_blur(array_float, blur_size)

    data_blurred = data_blurred.astype(np.int64)

    if treat_water == 'remove_below':

        data_blurred[index_below_water] = -1

    elif treat_water == 'remove_above':

        data_blurred[index_above_water] = 1 

    save_geotiff(path_geotiff_out, data_blurred, proj_wkt, data_transform, data_type)

    return

def gaussian_blur(in_array, size):
    
    # expand in_array to fit edge of kernel
    padded_array = np.pad(in_array, size, 'symmetric')
    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)
    # do the Gaussian blur

    return fftconvolve(padded_array, g, mode='valid')
