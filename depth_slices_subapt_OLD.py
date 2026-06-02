"""
depth_slices_subapt_OLD.py
Written by Ruth

DO NOT USE THIS FILE FOR GENERATING THE FOCAL STACK!! 
This is old code!
SUBAPETURE VIEWS ARE VERY LOW RESOLUTION!

Generates depth focal stacks
Modified from PlenopticToolbox2.0 in that it doesn't use disparity map
(which, when I tried running, took an extremely long time to generate disparity map)
Instead we do a simpler method: combine subapeture views similar to CS 166 HW 3 no 2 
"""

import numpy as np
import matplotlib.pyplot as plt
from xml.etree import ElementTree
import lens_grid as rtxhexgrid
import lens as rtxlens
import scipy.interpolate as sinterp
from scipy.interpolate import griddata
from scipy.ndimage import shift
import cv2 as cv2
import time
from tqdm import tqdm
from PIL import Image

from PIL import Image, ImageDraw

gif_frames = []

def get_subaperture_image(lenslet_img_list, lenslet_pc_list, u, v):
    """
    Generate subapeture image at the specified u,v location.
    Inputs:
        lenslet_img_list: [lenslet_0_img, lenslet_1_img, ...]
        lenslet_pc_list: [lenslet_0_center, lenslet_1_center,...]
            where lenslet_0_center = [lenslet_0_x_center_in_pixel_coords, lenslet_0_y_center_in_pixel_coords]
            This tells us: "When we take a pixel from lenslet_i_img, where in our
                subapeture image do we put it?"
            (We get this info from the corresponding physical lenslet location!)
        u, v: by row, col, which pixel to take from each lenslet_i_img to create the 
            subapeture image
            u, v should be ints in range [0, lenslet_pixels_per_side)
    Outputs:
        subapt_img: Subapeture image made from pixel (u,v) of every lenslet image
    """
    verbose = False    # print time of execution
    start_time = time.perf_counter()

    points = []   # combined list of all relevant pixel center coords
    xs = [pc[0] for pc in lenslet_pc_list]   # all pixel center x coords
    ys = [pc[1] for pc in lenslet_pc_list]   # all pixel center y coords

    # each relevant lenslet point will be chucked into the corresponding channel 
    colors_r = []
    colors_g = []
    colors_b = []
    
    # iterate over each lenslet, grabbing the relevant pixel and coordinates
    for patch, pc in zip(lenslet_img_list, lenslet_pc_list):
        points.append(pc)
        colors_r.append(patch[u,v,0])
        colors_g.append(patch[u,v,1])
        colors_b.append(patch[u,v,2])

    grid_x, grid_y = np.meshgrid(np.arange(int(min(xs)), int(max(xs)) + 1), np.arange(int(min(ys)), int(max(ys)) + 1))

    # Interpolate between points to fill in the color gaps
    R = griddata(points, colors_r, (grid_x, grid_y), method='linear')
    G = griddata(points, colors_g, (grid_x, grid_y), method='linear')
    B = griddata(points, colors_b, (grid_x, grid_y), method='linear')
    subapt_img = np.dstack([R,G,B])

    end_time = time.perf_counter()
    if verbose:
        execution_time = end_time - start_time
        print(f"Took {execution_time:.6f} seconds to generate sub-apt image at {u}, {v}.")

    # Clean up nan
    subapt_img = np.nan_to_num(subapt_img)
    return subapt_img

def generate_depth_slices(plenoptic_image_name, focal_planes, xml_file_path="plenoptic_images_raw/R29.xml"):

    """ 
    Input: 
        plenoptic_image_name: Name of the image to genearte depth slices at
            We will then examine the image at plenoptic_images_raw/plentoptic_image_name.png
        focal_planes: List of floats; chooses depth to focus at. Floats between 0 and 1. 
    Output:
        Depth images in plenoptic_images_depth_slices/plenoptic_image_name/
            as depth-00.png, depth-01.png... for all focal planes specified

    The images are from a R29 Raytrix Camera: https://data.mendeley.com/datasets/t6czryg5nw/1
    """
    '''
    1. Parse calibration file which tells us physical info about the lenslets

        Referenced: https://github.com/freerafiki/PlenopticToolbox2.0/blob/master/python/plenopticIO/imgIO.py 
            specifically, the function read_calibration(filename)   
    '''

        ############################################# XML HELPER FUNCTIONS ########
    def _float_from_element(parent, element):
        return float(parent.find(element).text)
         
    def _floats_from_section(parent, section, elements):
        sec = parent.findall(section)
        vals = dict()
        if len(sec) != 1:
            raise ValueError("Number of {0} entries != 1".format(section))
        sec = sec[0]
    
        for e in elements:
            res = sec.findall(e)
            if len(res) != 1:
                raise ValueError("Number of {0} entries != 1".format(e))
            vals[e] = float(res[0].text)
        return vals
    
    def _section_with_attrib(parent, section, attr, val): 
        elements = parent.findall(section)
        result = None

        for e in elements:
            if e.attrib[attr] == val:
                result = e
                break
        return result

    ############################################# XML PARSING ########
    tree = ElementTree.parse(xml_file_path)
    root = tree.getroot()

    calib = dict()
    
    calib["offset"] = _floats_from_section(root, "offset", ["x", "y"])
    calib["diameter"] = _float_from_element(root, "diameter")
    calib["rotation"] = _float_from_element(root, "rotation")
    calib["lens_border"] = _float_from_element(root, "lens_border")
    calib["lens_base_x"] = _floats_from_section(root, "lens_base_x", ["x", "y"])
    calib["lens_base_y"] = _floats_from_section(root, "lens_base_y", ["x", "y"])
    calib["sub_grid_base"] = _floats_from_section(root, "sub_grid_base", ["x", "y"])

    lens_types = []
    i = 0
    sec = _section_with_attrib(root, "lens_type", "id", str(i))
    while sec is not None:
        sub_dict = dict()
        sub_dict["offset"] = _floats_from_section(sec, "offset", ["x", "y"])
        sub_dict["depth_range"] = _floats_from_section(sec, "depth_range", ["min", "max"])
        lens_types.append(sub_dict)
        i += 1
        sec = _section_with_attrib(root, "lens_type", "id", str(i))
    calib["lens_types"] = lens_types

    '''
    2. Calculate lenslet info from calibration file constants, 
        so we know what our lenslets look like and where their centers are

        Referenced: https://github.com/freerafiki/PlenopticToolbox2.0/blob/master/python/plenopticIO/imgIO.py 
            specifically, the class MLACalibration(object)
    '''
    lens_diameter = calib['diameter']   # the diameter of a single lens in pixels
    rot_angle = calib['rotation']  # rotation angle of the grid, counter-clockwise

        # pixel coordinates of the center lens
        # raw image origin (0, 0) is upper left corner
        # lower right corner is (h, w)
    x, y = calib['offset']['x'], calib['offset']['y']
    offset = np.array((-y, x))

        # lens bases in lens units: X
    x, y = calib['lens_base_x']['x'], calib['lens_base_x']['y']
    v = (x, y)
    lens_base_x = np.array([-v[1], v[0]])           # reflect
        # lens bases in lens units: Y
    x, y = calib['lens_base_y']['x'], calib['lens_base_y']['y']
    v = (x, y)
        # Calculate lbasis
    lens_base_y = np.array([-v[1], v[0]])
    lbx = lens_base_x
    lby = -lens_base_y + lens_base_x
    lbasis = np.vstack((lby, lbx)).T    # axial grid basis vectors in lens units

    # Image shape info
    processed_image_file_path = "plenoptic_images_raw/"+plenoptic_image_name+".png"
    img = plt.imread(processed_image_file_path)
    img_shape = np.asarray(img.shape[0:2])
    print(img_shape)
    
    # Lens coordinate info
        # center coords: array of (y, x) lens center coordinates in pixels
    coords, ny, nx, sy, sx, img_c = rtxhexgrid.hex_lens_grid_plus(img_shape, lens_diameter, rot_angle, offset, lbasis)
        # each row spans -1*radius to 1*radius, num steps = lens_diameter 
    local_grid = rtxlens.LocalLensGrid(lens_diameter)

    # smooth 2D surface of RGB channels
    # because refocusing requires sampling at fractional pixel coordinates
    gridy, gridx = range(img.shape[0]), range(img.shape[1]) 
    interp_r = sinterp.RectBivariateSpline(gridy, gridx, img[:,:,0])
    interp_g = sinterp.RectBivariateSpline(gridy, gridx, img[:,:,1])
    interp_b = sinterp.RectBivariateSpline(gridy, gridx, img[:,:,2])

    '''
    3. Grab lenslet locations and lenslet images
        (These will be used to create sub-apeture views later)
    '''
    # Loop through each lenslet to grab each lenslet and make it square

    lenslet_pixels_per_side = int(np.ceil(lens_diameter))
    lenslet_img_list = []  # A list of all the lenslet images
    lenslet_pc_list = []   # A list of all the lenslet centers in pixel coords

    max_pc_x = 0
    max_pc_y = 0
    min_pc_x = 10000000000
    min_pc_y = 10000000000

    for lc in coords:
        
        # Grab lens center
        pc = coords[lc] # pc = pixel coordinates
        pc_x, pc_y = pc[1], pc[0]

        # Get the image from one lens
        # local_grid helps us index in the neighborhood around pixel center
            # mask away pixels not within the circular len
        mask = 1 #(local_grid.xx**2 + local_grid.yy**2 <= (lens_diameter/2)**2)
        r = interp_r(local_grid.y + pc_y, local_grid.x + pc_x)*mask
        g = interp_g(local_grid.y + pc_y, local_grid.x + pc_x)*mask
        b = interp_b(local_grid.y + pc_y, local_grid.x + pc_x)*mask
        patch = np.dstack([r,g,b])

        # Save to rectangular arrays
        lenslet_img_list.append(patch)
        lenslet_pc_list.append([pc_x, pc_y])

        # Check min/max coordinates (used for lenslet_pc compression in next step)
        max_pc_x = max(max_pc_x, pc_x)
        min_pc_x = min(min_pc_x, pc_x)
        max_pc_y = max(max_pc_y, pc_y)
        min_pc_y = min(min_pc_y, pc_y)

    '''
    3.5: Compress lenslet_pc_list
        Reason 1: This is because lenslet_pc_list tells us where the lenslets physically are,
            but when we use this list to generate sub-apeture views,
            we can place the pixel from each lenslet image much closer together
            than the lenslets are physically spaced.
        Reason 2: The closer we place the pixels, the less interpolation get_subaperture_image 
            has to do. The output image is smaller too, and this speeds up our overall pipeline.
    '''
    space_factor = 0.9   # to prevent us from compressing lesnlet locations to <1 pixel apart
    x_compression_factor = space_factor*(max_pc_x - min_pc_x)/nx
    y_compression_factor = space_factor*(max_pc_y - min_pc_y)/ny
    lenslet_pc_list = np.array(lenslet_pc_list)
    lenslet_pc_list[:, 0] = lenslet_pc_list[:, 0]/x_compression_factor
    lenslet_pc_list[:, 1] = lenslet_pc_list[:, 1]/y_compression_factor
    lenslet_pc_list = lenslet_pc_list.tolist()
    output_image_shape = (int(max_pc_y/y_compression_factor)-int(min_pc_y/y_compression_factor)+1, 
                          int(max_pc_x/x_compression_factor)-int(min_pc_x/x_compression_factor)+1,3)

    '''
    4. Combine into focal stacks! 
    '''
    verbose = True    # progress bars for generating focal stacks

    for i in range(len(focal_planes)):
        d = focal_planes[i]
        i_str = "0"+str(i) if i<10 else str(i)
        output_str = "plenoptic_images_depth_slices/"+plenoptic_image_name+"/depth--"+i_str+".png"

        # Output depth image
        output_img = np.zeros(output_image_shape)

        num_valid_pixels = 0
        with tqdm(total=lenslet_pixels_per_side**2, desc=f"Focal Stack for d={d}") as pbar:
            for u in range(lenslet_pixels_per_side):
                if u % 2 == 0:
                    v_range = range(lenslet_pixels_per_side)
                else:
                    v_range = reversed(range(lenslet_pixels_per_side))

                for v in v_range:

                    pbar.update(1)

                    center_corrected_u = u - (lenslet_pixels_per_side-1)//2
                    center_corrected_v = v - (lenslet_pixels_per_side-1)//2
                    if (center_corrected_u**2 + center_corrected_v**2 > (0.8*lens_diameter/2)**2):
                        continue  # We only conider a circular lens
                    else:
                        num_valid_pixels += 1

                    subapt_img = get_subaperture_image(lenslet_img_list, lenslet_pc_list, u, v)

                    ''' GIF generation, sanity check, AI helped me write this
                    subapt_uint8 = np.uint8(255 * subapt_img.clip(0, 1))

                    frame = Image.fromarray(subapt_uint8)

                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((5, 5, 120, 35), fill=(0, 0, 0))
                    draw.text((10, 10), f"u={u}, v={v}", fill=(255, 255, 255))

                    gif_frames.append(frame)
                    #'''

                    s_shift = d*center_corrected_u   # Can be a float, scipy handles interpolation
                    t_shift = d*center_corrected_v

                    shifted_arr = shift(subapt_img, shift=(s_shift, t_shift,0), mode='nearest')
                    output_img += shifted_arr

        print(f"Saving {output_str}")
        # Normalize before saving
        output_img /= num_valid_pixels
        Image.fromarray(np.uint8(255 * output_img.clip(0, 1)**(1/1))).save(output_str)

        if len(gif_frames) > 0:
            gif_frames[0].save(
                "subaperture_sweep.gif",
                save_all=True,
                append_images=gif_frames[1:],
                duration=1000,   # milliseconds = 1 second
                loop=0
            )

generate_depth_slices("University_Processed", [0.3])       