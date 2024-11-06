"""Helper functions for various tasks without a clear category


References
----------
"""
import numpy as np
import datetime
import time
from functools import wraps
import numpy as np

def cart2pol(x, y):
    """Transforms the provided cartesian coordinates to polar coordinates
    
    Parameters
    ----------
    x : float or ndarray of shape (num_points,)
        x coordinate
    y : float or ndarray of shape (num_points,)
        y coordinate

    Returns
    -------
    r : float or ndarray of shape (num_points,)
        radius
    angle : float or ndarray of shape (num_points,)
        angle in radians. 0 is the x-direction, pi/2 is the y-direction
    
    """
    r = np.hypot(x, y)
    angle = np.arctan2(y, x)
    return (r, angle)

def pol2cart(r, angle):
    """Tranforms the provided polar coordinates to cartesian coordinates
    
    Parameters
    ----------
    r : float or ndarray of shape (num_points,)
        radius
    angle : float or ndarray of shape (num_points,)
        angle in radians. 0 is the x-direction, pi/2 is the y-direction
    
    Returns
    -------
    x : float or ndarray of shape (num_points,)
        x coordinate
    y : float or ndarray of shape (num_points,)
        y coordinate
    """
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return (x, y)

def cart2spherical(cart_coord):
    """Transforms the provided cartesian coordinates to spherical coordinates

    Parameters
    ----------
    cart_coord : ndarray of shape (num_points, 3)

    Returns
    -------
    r : ndarray of shape (num_points, 1)
        radius of each point
    angle : ndarray of shape (num_points, 2)
        angle[:,0] is theta, the angle in the xy plane, where 0 is x direction, pi/2 is y direction
        angle[:,1] is phi, the zenith angle, where 0 is z direction, pi is negative z direction
    """
    r = np.linalg.norm(cart_coord, axis=1)
    r_xy = np.linalg.norm(cart_coord[:,:2], axis=1)

    theta = np.arctan2(cart_coord[:,1], cart_coord[:,0])
    phi = np.arctan2(r_xy, cart_coord[:,2])
    angle = np.concatenate((theta[:,None], phi[:,None]), axis=1)
    return (r, angle)

def spherical2cart(r, angle):
    """Transforms the provided spherical coordinates to cartesian coordinates
    
    Parameters
    ----------
    r : ndarray of shape (num_points, 1) or (num_points,)
        radius of each point
    angle : ndarray of shape (num_points, 2)
        the angles in radians
        angle[:,0] is theta, the angle in the xy plane, where 0 is x direction, pi/2 is y direction
        angle[:,1] is phi, the zenith angle, where 0 is z direction, pi is negative z direction
    
    Returns
    -------
    cart_coord : ndarray of shape (num_points, 3)
        the cartesian coordinates
    """
    num_points = r.shape[0]
    cart_coord = np.zeros((num_points,3))
    cart_coord[:,0] = np.squeeze(r) * np.cos(angle[:,0]) * np.sin(angle[:,1])
    cart_coord[:,1] = np.squeeze(r) * np.sin(angle[:,0]) * np.sin(angle[:,1])
    cart_coord[:,2] = np.squeeze(r) * np.cos(angle[:,1])
    return cart_coord


def db2mag(db):
    """Transforms the provided decibel value to magnitude"""
    return 10 ** (db / 20)

def mag2db(amp):
    """Transforms the provided magnitude to decibel"""
    return 20 * np.log10(amp)

def db2pow(db):
    """Transforms the provided decibel value to power"""
    return 10 ** (db / 10)

def pow2db(power):
    """Transforms the provided power to decibel"""
    return 10 * np.log10(power)




def measure_time(name):
    """
    Use as decorator on a function to measure the time the function takes
    use as @measure_time('print_this_as_name')
    """
    def measure_internal(func):
        @wraps(func)
        def _time_it(*args, **kwargs):
            start = int(round(time.time() * 1000))
            try:
                return func(*args, **kwargs)
            finally:
                end_ = int(round(time.time() * 1000)) - start
                print(name + f" execution time: {end_ if end_ > 0 else 0} ms")
        return _time_it
    return measure_internal



def flatten_dict(dict_to_flatten, parent_key="", sep="~"):
    items = []
    for key, value in dict_to_flatten.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    new_dict = dict(items)
    return new_dict


def restack_dict(dict_to_stack, sep="~"):
    """Only accepts dicts of depth 2.
    All elements must be of that depth."""
    extracted_data = {}
    for multi_key in dict_to_stack.keys():
        key_list = multi_key.split(sep)
        if len(key_list) > 2:
            raise NotImplementedError
        if key_list[0] not in extracted_data:
            extracted_data[key_list[0]] = {}
        extracted_data[key_list[0]][key_list[1]] = dict_to_stack[multi_key]
    return extracted_data



def get_time_string(detailed=False):
    """Returns a string with the current time in the format 'year_month_day_hour_minute'
    
    Parameters
    ----------
    detailed : bool
        If True, seconds and microseconds will be included in the string

    Returns
    -------
    time_str : str
        The time string
    """
    tm = datetime.datetime.now()
    time_str = (
        str(tm.year)
        + "_"
        + str(tm.month).zfill(2)
        + "_"
        + str(tm.day).zfill(2)
        + "_"
        + str(tm.hour).zfill(2)
        + "_"
        + str(tm.minute).zfill(2)
    )
    if detailed:
        time_str += "_" + str(tm.second).zfill(2)
        time_str += "_" + str(tm.microsecond).zfill(2)
    return time_str


def get_unique_folder(prefix, parent_folder, detailed_naming=False):
    """Returns a unique folder name in the parent folder with the prefix and the current time

    The folder name has the form parent_folder / prefix_year_month_day_hour_minute_0. If multiple folders are created
    within the same minute, the number is incremented by 1 for each new folder. 

    Parameters
    ----------
    prefix : str
        The prefix for the folder name
    parent_folder : Path
        The parent folder where the new folder should be created, as a Path object (from pathlib)
    detailed_naming : bool
        If True, the folder name will include seconds and microseconds. 
        If used with multithreading, it is a good idea to set this to True. 
        In that case, uniqueness is not guaranteed, but it reduces the risk of clashes significantly.
    
    Returns
    -------
    folder_name : Path
        The full path to the new folder. The folder is not created by this function.
    """
    file_name = prefix + get_time_string(detailed=detailed_naming)
    file_name += "_0"
    folder_name = parent_folder / file_name
    if folder_name.exists():
        idx = 1
        folder_name_len = len(folder_name.name) - 2
        while folder_name.exists():
            new_name = folder_name.name[:folder_name_len] + "_" + str(idx)
            folder_name = folder_name.parent / new_name
            idx += 1
    return folder_name