"""
depth_slices_from_depth_map.py
Written by Ruth

Generates depth focal stacks from a given (colored) plenoptic image and depth map.
Modified from PlenopticToolbox2.0 in that it doesn't use disparity map
(which, when I tried running, took an extremely long time to generate), but infers
disparity from the provided depth map png.
"""

import numpy as np
import matplotlib.pyplot as plt
from xml.etree import ElementTree
import lens_grid as rtxhexgrid
import lens as rtxlens
import scipy.interpolate as sinterp
from scipy.interpolate import griddata
import scipy.ndimage
from scipy.ndimage import shift
import cv2 as cv2
import time
from tqdm import tqdm
from PIL import Image, ImageDraw

def _hex_focal_type(c):
    """
    Calculates the focal type for the three lens hexagonal grid
    """

    focal_type = ((-c[0] % 3) + c[1]) % 3

    return focal_type

def smoothstep(x, goingup, goingdown, curvingrange):

    y = np.zeros_like(x)
    ones_indices = np.where((x > (goingup + curvingrange) ) & (x < (goingdown - curvingrange)) )
    y[ones_indices] = 1
    
    curveup = np.where((x >= (goingup - curvingrange) ) & (x <= (goingup + curvingrange)) )
    if goingup < curvingrange:
        y[curveup] = 1
    else:
        y[curveup] = 0.5 + (x[curveup] - goingup  ) / (x[curveup[0][-1]] - x[curveup[0][0]])
    curvedown = np.where((x >= (goingdown - curvingrange) ) & (x <= (goingdown + curvingrange)) )
    if goingdown > x[-1] - curvingrange:
        y[curvedown] = 1
    else:
        y[curvedown] = 0.5 + ( x[curvedown] - goingdown ) / (x[curvedown[0][0]] - x[curvedown[0][-1]])

    return y

# The scipy ndimage filter adapted for 3-4 channels usage
def median_filter(img, filter_size, footprint=None, changeColorSpace=False):

    if len(img.shape) == 2:

        # normal filter
        filtered_img = scipy.ndimage.filters.median_filter(img, filter_size, footprint)

    elif len(img.shape) == 3:

        # channels
        filtered_img = np.zeros(img.shape)
        for i in range (3):
            filtered_img[:,:,i] = scipy.ndimage.filters.median_filter(img[:,:,i], filter_size, footprint)
        if filtered_img.shape[2] == 4:
            filtered_img[:,:,3] = 1

    else:

        # wrong
        print("is the first argument really an image?")
        filtered_img = np.zeros(img.shape)

    return filtered_img

    # resolution should be correlated with number of lenses more than 
    
def generate_depth_slices_new(plenoptic_image_name, depth_image_name, focal_planes, fmin = 0.2, fmax = 1.2, xml_file_path="plenoptic_images_raw/R29.xml"):

    """ 
    Input: 
        plenoptic_image_name: Name of the image to genearte depth slices at
            We will then examine the image at plenoptic_images_raw/plentoptic_image_name.png
        depth_image_name: We will use this to pull a "disparity map"
            Expecting the depth image at plenoptic_images_raw/depth_image_name.png
        focal_planes: List of floats; chooses depth to focus at. Floats between 0 and 1. 
        fmin: Minimum expected focal plane. Disparity 0 stuff is focused here when focal_plane = fmin
        fmax: Maximum expected focal plane. Disparity 1 stuff is focused here when focal_plane = fmax
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
    
    # Lens coordinate info
        # center coords: array of (y, x) lens center coordinates in pixels
    coords, ny, nx, sy, sx, img_c = rtxhexgrid.hex_lens_grid_plus(img_shape, lens_diameter, rot_angle, offset, lbasis)
        # each row spans -1*radius to 1*radius, num steps = lens_diameter 
    local_grid = rtxlens.LocalLensGrid(lens_diameter)

    print("Done loading lens grid and calibration info.")

    # get depth image and resize to size of actual image
    depth_image_file_path = "plenoptic_images_raw/"+depth_image_name+".png"
    depth_img = plt.imread(depth_image_file_path)[:, :, :3]  # RGB-alpha, only care about RGB
    depth_img = cv2.resize(depth_img,(img.shape[1], img.shape[0]),interpolation=cv2.INTER_LINEAR)

    # then, for the depth image,
    # smooth 2D surface of RGB channels for grabbing disparity from a patch
    gridy, gridx = range(img.shape[0]), range(img.shape[1]) 
    depth_interp_r = sinterp.RectBivariateSpline(gridy, gridx, depth_img[:,:,0])
    depth_interp_g = sinterp.RectBivariateSpline(gridy, gridx, depth_img[:,:,1])
    depth_interp_b = sinterp.RectBivariateSpline(gridy, gridx, depth_img[:,:,2])

    '''
    3. Grab lenslet locations and lenslet images
        (These will be used to create sub-apeture views later)
    '''
    # Loop through each lenslet to grab each lenslet and make it square

    lenslet_pixels_per_side = int(np.ceil(lens_diameter))
    lenslet_pc_list = []   # A list of all the lenslet centers in pixel coords
    disparity_per_lenslet = [] # A list of average disparity at each lenslet
    cmap = plt.cm.jet(np.linspace(0, 1, 256))[:, :3] # Colormap to compare depth map colors to
    
    for lc in coords:
        
        # Grab lens center
        pc = coords[lc] # pc = pixel coordinates
        pc_x, pc_y = pc[1], pc[0]

        # Get the depth color from one lenslet
        # local_grid helps us index in the neighborhood around pixel center
            # mask away pixels not within the circular len
        mask = (local_grid.xx**2 + local_grid.yy**2 <= (lens_diameter/2)**2)
        depth_r = depth_interp_r(local_grid.y + pc_y, local_grid.x + pc_x)*mask
        depth_g = depth_interp_g(local_grid.y + pc_y, local_grid.x + pc_x)*mask
        depth_b = depth_interp_b(local_grid.y + pc_y, local_grid.x + pc_x)*mask
        depth_patch = np.dstack([depth_r,depth_g,depth_b])

        # Get the disparity at this point
        depth_pixels = depth_patch[mask > 0]
        depth_rgb = depth_pixels.mean(axis=0)
        dist = np.sum((cmap - depth_rgb)**2, axis=1) # distance to every jet entry
        idx = np.argmin(dist)
        disp = 1.0 - idx / 255.0    # now 0-1
        #disp = min_disp + (max_disp - min_disp)*disp # rescale to desired range

        # Save to arrays
        lenslet_pc_list.append([pc_x, pc_y])
        disparity_per_lenslet.append(disp)

    # Improve disparity dynamic range
    mean = np.mean(disparity_per_lenslet)
    std = np.std(disparity_per_lenslet)
    lower_3sig = mean - 2 * std
    upper_3sig = mean + 2 * std
    disparity_per_lenslet = (disparity_per_lenslet - lower_3sig)/(upper_3sig - lower_3sig)
    disparity_per_lenslet = np.clip(disparity_per_lenslet, a_min=1e-10, a_max=1-1e-10)

    print("Done calculating disparity from depth.")

    '''
    4. Calculate interpolated maps for image and depth image
        
    Referenced https://github.com/freerafiki/PlenopticToolbox2.0/blob/master/python/plenopticIO/imgIO.py
    And I resized the depth image to match the disparity map format myself
    '''
    # the image grid in pixels used for the bivariate spline interpolation
    weights = np.array([0.3, 0.59, 0.11])
    data = np.sum([img[:, :, i] * weights[i] for i in range(3)], axis=0)
    data_col = img
    gridy, gridx = range(data.shape[0]), range(data.shape[1])
    data_interp = sinterp.RectBivariateSpline(gridy, gridx, data)
    data_interp_r = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,0])
    data_interp_g = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,1])
    data_interp_b = sinterp.RectBivariateSpline(gridy, gridx, data_col[:,:,2])
    
    '''
    5. Combine focal stack

    Largely borrowed from: https://github.com/freerafiki/PlenopticToolbox2.0/blob/master/python/rendering/render.py#L1612
    specifically, the function render_interp_img_at_focal_plane(imgs, interps, calibs, focal_plane, sam_per_lens, cut_borders)
    '''
    sam_per_lens = 15
    cut_borders = True

    for i in range(len(focal_planes)):
        focal_plane = focal_planes[i]
        i_str = "0"+str(i) if i<10 else str(i)
        output_str = "plenoptic_images_depth_slices/"+plenoptic_image_name+"/depth-"+i_str+".png"
        print(f"Starting to calculate focal image for d={focal_plane}...")

        # Setting resolution of rendered image
        hs = np.floor(sam_per_lens/2).astype(int)
        lens_types = 3
        reducing_factor = (lens_diameter / sam_per_lens) * 2 #* lens_types
        resolution = np.round(img_shape / reducing_factor).astype(int)
        print("\traw image is {}x{}, rendered image will be {}x{}".format(img_shape[0], img_shape[1], resolution[0], resolution[1]))
        rnd_img = np.zeros((resolution[0], resolution[1], 3, lens_types))
        rnd_cnt = np.zeros((resolution[0], resolution[1], 3, lens_types))
        coarse_d = np.zeros((resolution[0], resolution[1], lens_types))
        x, y = local_grid.x, local_grid.y
        
        patch_size_for_sampling = focal_plane * lens_diameter / 2
        effective_patch_sizes = []
        for j, lc in enumerate(coords):

            # pixel coordinates
            pc = coords[lc]
            ft = _hex_focal_type(lc)
            single_val_disp = disparity_per_lenslet[j] # previously np.mean(disp_interp(y+pc[0], x+pc[1]))
            #print(f"\tsingle_val_disp = {single_val_disp}, patch_size_for_sampling = {patch_size_for_sampling}")
            # sample the image at the correct position
            coords_resized = pc / reducing_factor

                # Try to improve blurring of non-in-focus plane

                # First, choose the expected disparity in the plane of focus, 
                # making note that smaller disparity = close to camera
                # and smaller focal plane = close to camera
            focus_disp = (focal_plane - fmin)/(fmax - fmin)
            focus_error = (single_val_disp - focus_disp)  # positive = farther away --> make patch smaller
            blur_strength = 10.0
            effective_patch_size = patch_size_for_sampling - blur_strength * focus_error
            effective_patch_size = max(effective_patch_size, 1)
            effective_patch_size = min(effective_patch_size, 0.95*(lens_diameter/2))
            effective_patch_sizes.append(effective_patch_size)

            intPCx = np.ceil(coords_resized[1]).astype(int)
            intPCy = np.ceil(coords_resized[0]).astype(int)
            if intPCx > sam_per_lens and resolution[1] - intPCx > sam_per_lens and intPCy > sam_per_lens and resolution[0] - intPCy > sam_per_lens:

                sampling_pattern = np.linspace(
                    -effective_patch_size,
                    effective_patch_size,
                    2*sam_per_lens + 1
                )          
                sampling_pattern_x = sampling_pattern
                sampling_pattern_y = sampling_pattern

                # extract the patch
                patch_values = np.dstack((data_interp_r(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                    data_interp_g(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1]),
                    data_interp_b(sampling_pattern_y+pc[0], sampling_pattern_x+pc[1])))
                #print("patch_values raw:", patch_values.shape)
                patch_values = np.clip(patch_values, 0, np.max(patch_values))
                assert patch_values.shape[0] == 2*sam_per_lens + 1
                assert patch_values.shape[1] == 2*sam_per_lens + 1
                #print("patch_values size {}".format(patch_values.shape))
                # interpolate the values
                interp_patch_r = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,0])
                interp_patch_g = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,1])
                interp_patch_b = sinterp.RectBivariateSpline(range(patch_values.shape[0]), range(patch_values.shape[1]), patch_values[:,:,2])

                sampling_pattern_for_patch_y = np.arange((intPCy-coords_resized[0]), (intPCy-coords_resized[0]+2*sam_per_lens+1), 1)
                sampling_pattern_for_patch_x = np.arange((intPCx-coords_resized[1]), (intPCx-coords_resized[1]+2*sam_per_lens+1), 1)

                # get the actual values for each channel
                r_channel = interp_patch_r(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
                g_channel = interp_patch_g(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)
                b_channel = interp_patch_b(sampling_pattern_for_patch_y, sampling_pattern_for_patch_x)

                # stack the 3 channels together
                rgb_interp_patch_img = np.dstack((r_channel, g_channel, b_channel))
                rgb_interp_patch_img = np.clip(rgb_interp_patch_img, 0, np.max(rgb_interp_patch_img))

                hsrgb = np.floor(rgb_interp_patch_img.shape[0]/2).astype(int)
                x = np.linspace(-hsrgb, hsrgb, rgb_interp_patch_img.shape[0])
                y = np.linspace(-hsrgb, hsrgb, rgb_interp_patch_img.shape[1])
                xx, yy = np.meshgrid(x,y)
                mask = np.zeros_like(xx)
                mask[xx**2 + yy**2 < hsrgb**2] = 1
                mask3c = np.dstack((mask, mask, mask))

                # fill the images
                rnd_img[intPCy-sam_per_lens:intPCy+sam_per_lens+1, intPCx-sam_per_lens:intPCx+sam_per_lens+1,:, ft] += rgb_interp_patch_img * mask3c
                rnd_cnt[intPCy-sam_per_lens:intPCy+sam_per_lens+1, intPCx-sam_per_lens:intPCx+sam_per_lens+1, :, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1], 3)) * mask3c
                coarse_d[intPCy-sam_per_lens:intPCy+sam_per_lens+1, intPCx-sam_per_lens:intPCx+sam_per_lens+1, ft] += np.ones((rgb_interp_patch_img.shape[0], rgb_interp_patch_img.shape[1])) * single_val_disp * mask

        #pdb.set_trace()
        img0vals0 = (rnd_cnt[:,:,:,0] == 0).astype(np.uint8)
        img0vals1 = (rnd_cnt[:,:,:,1] == 0).astype(np.uint8)
        img0vals2 = (rnd_cnt[:,:,:,2] == 0).astype(np.uint8)
        rnd0 = rnd_img[:,:,:,0] / (rnd_cnt[:,:,:,0] + img0vals0)
        rnd1 = rnd_img[:,:,:,1] / (rnd_cnt[:,:,:,1] + img0vals1)
        rnd2 = rnd_img[:,:,:,2] / (rnd_cnt[:,:,:,2] + img0vals2)
        coarse_d0 = coarse_d[:,:,0] / (rnd_cnt[:,:,0,0] + img0vals0[:,:,0])
        coarse_d1 = coarse_d[:,:,1] / (rnd_cnt[:,:,0,1] + img0vals1[:,:,0])
        coarse_d2 = coarse_d[:,:,2] / (rnd_cnt[:,:,0,2] + img0vals2[:,:,0])

        coarse_d_tot = (coarse_d0 + coarse_d1 + coarse_d2 ) / lens_types

        #pdb.set_trace()
        #plt.figure()
        filt_size = min(5, np.round(sam_per_lens/5).astype(int))

        # for the R29 dataset
        range_t0 = [0.400, 1]
        range_t1 = [0, 0.200]
        range_t2 = [0.200, 0.400]
        quantization_step = 0.01
        x = np.arange(0.0, 1.0, quantization_step)
        y_t0 = smoothstep(x, range_t0[0], range_t0[1], 0.05)
        y_t1 = smoothstep(x, range_t1[0], range_t1[1], 0.05)
        y_t2 = smoothstep(x, range_t2[0], range_t2[1], 0.05)

        weights_t0 = 1/lens_types + y_t0 / 3 * 2 - y_t1 / 3 * 1 - y_t2 /3 * 1
        weights_t1 = 1/lens_types - y_t0 / 3 * 1 + y_t1 / 3 * 2 - y_t2 /3 * 1
        weights_t2 = 1/lens_types - y_t0 / 3 * 1 - y_t1 / 3 * 1 + y_t2 /3 * 2

        weights = np.zeros_like(rnd0)
        coarse_d_tot = np.clip(coarse_d_tot, 0, 1)
        idisp = np.floor(coarse_d_tot / quantization_step).astype(np.uint8)


        weights[:,:,0] = weights_t0[idisp]
        weights[:,:,1] = weights_t1[idisp]
        weights[:,:,2] = weights_t2[idisp]

        #pdb.set_trace()

        #rnd_tot = rnd0f * weights + rnd1f * weights + rnd2f * weights
        weights0_w3c = np.dstack((weights[:,:,0], weights[:,:,0], weights[:,:,0]))
        weights1_w3c = np.dstack((weights[:,:,1], weights[:,:,1], weights[:,:,1]))
        weights2_w3c = np.dstack((weights[:,:,2], weights[:,:,2], weights[:,:,2]))
        rnd_nof = rnd0 * weights0_w3c + rnd1 * weights1_w3c + rnd2 * weights2_w3c

        filt_after = median_filter(rnd_nof, filt_size)

        rnd_img_final = np.clip(filt_after, 0, 1)
        padding = hs + np.floor(hs/2).astype(int)
        if cut_borders:
            rnd_img_final = rnd_img_final[padding:rnd_img_final.shape[0]-padding, padding:rnd_img_final.shape[1]-padding,:]
        
        print(f" patch sizes {min(effective_patch_sizes)} -- {max(effective_patch_sizes)}")
        Image.fromarray(np.uint8(255 * rnd_img_final.clip(0, 1)**(1/1))).save(output_str)
        print(f"\t...saved image {output_str}")

generate_depth_slices_new("University_Processed", "University_Depth", np.linspace(0.58, 0.92, 11).tolist(), 0.3, 1.0)       
# generate_depth_slices_new("Beers_Processed", "Beers_Depth", [0.1, 0.5, 1.0, 1.2])       