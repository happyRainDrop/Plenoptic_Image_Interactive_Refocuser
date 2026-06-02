"""
refocus_aids.py
Written by Ruth

Generates blurred and sharpened images,
and the numpy array to choose which focal stack to select based on position
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt

def generate_depth_selection_arr(path_to_focal_stack_folder):
    """
    Generates a numpy array that helps us select a focal stack for a given 
    cursor position.

    Inputs:
        path_to_focal_stack_folder: Path to the folder containing the focal stack images.
            We expect to find path_to_focal_stack_folder/depth-00.png ... 
                                path_to_focal_stack_folder/depth-10.png here.
            depth-00.png, depth-10.png, etc are all the same size
    Outputs:
        depth_chooser.npz: A npz of our output numpy array
        Output array format:
            If each of depth-00.png are H x W where W > H, 
            output array is of the shape (100, int(100*W/H))
            [[10, 10, 9, 8, ...]] means that the top left-most corner of the image
            is sharpest in focal stack 10 (depth-10.png), for example.
    """
    NUM_PNGS = 11
    focal_stack = []

    for i in range(NUM_PNGS):
        i_str = "0"+str(i) if i<10 else str(i)
        depth_img_name = path_to_focal_stack_folder + "/depth-" + i_str + ".png"
        img = cv2.imread(depth_img_name, cv2.IMREAD_GRAYSCALE)
        focal_stack.append(img)

    H, W = focal_stack[0].shape

    out_h = 250
    out_w = int(out_h * W / H)

    # Check local sharpness to choose a focal stack
    sharpness_maps = []
    for img in focal_stack:

        # Use all-in-focus method from HW 1:
        # compute high-frequency component,
        # then blur I_HF**2 with 1
        img_float = img.astype(np.float32) / 255.0
        blurred = gaussian_filter(img_float,sigma=2,mode="nearest")
        I_HF = img_float - blurred
        sharpness = gaussian_filter(I_HF * I_HF,sigma=8,mode="nearest")

        # smooth so we don't get noisy per-pixel depth
        sharpness = cv2.GaussianBlur(sharpness, (11, 11), 0)

        # Now make it the size of our output depth-stack indexing map
        sharpness_small = cv2.resize(sharpness, (out_w, out_h), interpolation=cv2.INTER_AREA)
        sharpness_maps.append(sharpness_small)


        # Now, choose the best focal depth pic for each location!
    # We don't naively choose the sharpest, because that can lead to big jumps
    # across the focal stack -- ideally, adjacent physical locations
    # should have close-together chosen focal stacks too (like 06 and 07)
    sharpness_volume = np.stack(sharpness_maps, axis=0)
    depth_chooser_a = np.argmax(sharpness_volume, axis=0).astype(np.uint8)
        # Smooth focal jumps
    # Lastly, Smooth focal jumps
    depth_chooser_a = cv2.medianBlur(depth_chooser_a.astype(np.uint8),11)
    np.savetxt("output_n.txt", depth_chooser_a, fmt="%d", delimiter=",")

    sharpness_maps = np.asarray(sharpness_maps)

    # First pass: Fill in highly-confident pixels
    depth_chooser = np.zeros((out_h, out_w))
    confidence_threshold = np.max(sharpness_maps)*0.005
    for r in range(out_h):
        for c in range(out_w):
            sharpnesses = sharpness_maps[:, r, c]
            
            # Get the two largest values
            largest_two = np.partition(sharpnesses, -2)[-2:]
            max_sharpness = largest_two[1]
            second_max_sharpness = largest_two[0]

            # Check confidence
            if (max_sharpness - second_max_sharpness > confidence_threshold):
                depth_chooser[r,c] = np.argmax(sharpnesses)
            else:
                depth_chooser[r,c] = -1

    np.savetxt("output_n.txt", depth_chooser, fmt="%d", delimiter=",")
    # Then, create a "reference" array of what things would look like
    # if every unconfident region just took the depth map of its nearest neighbor
    unconfident_mask = (depth_chooser == -1)      # True where pixels are "unconfident"
    # Get indices of nearest non-empty cell for every location
    _, indices = distance_transform_edt(unconfident_mask, return_distances=True, return_indices=True)
    # Fill empties with nearest neighbor values
    naive_depth_chooser = depth_chooser[tuple(indices)]

    # Second pass: Fill in neighboring low-confidence pixels
    SHARPNESS_DIFF_THRESH = confidence_threshold*0.1
    for r in range(out_h):
        for c in range(out_w):
            if (depth_chooser[r,c] == -1):
                
                sharpnesses = sharpness_maps[:, r, c]
                max_sharpness = np.max(sharpnesses)
                naive_chosen_depth = int(naive_depth_chooser[r,c])

                # Find the nearest to this selected depth!
                diff_from_naive_depth =  0
                while(diff_from_naive_depth < NUM_PNGS):
                    smaller_chosen_depth = max(0, naive_chosen_depth - diff_from_naive_depth)
                    larger_chosen_depth = min(NUM_PNGS-1, naive_chosen_depth + diff_from_naive_depth)

                    if (np.max(sharpnesses) - sharpnesses[smaller_chosen_depth] < SHARPNESS_DIFF_THRESH):
                        depth_chooser[r,c] = smaller_chosen_depth
                        break
                    if (np.max(sharpnesses) - sharpnesses[larger_chosen_depth] < SHARPNESS_DIFF_THRESH):
                        depth_chooser[r,c] = larger_chosen_depth
                        break

                    diff_from_naive_depth+=1
       
        # Lastly, Smooth focal jumps
    depth_chooser = cv2.medianBlur(depth_chooser.astype(np.uint8),11)
    np.savetxt("output.txt", depth_chooser, fmt="%d", delimiter=",")
    np.savez(path_to_focal_stack_folder + "/depth_chooser.npz", depth_chooser=depth_chooser)



def generate_blurred_focal_stack(path_to_focal_stack_folder):
    """
    Applies a simple Gaussian blurr to each focal stack and saves the blurred version.

    Inputs:
        path_to_focal_stack_folder: Path to the folder containing the focal stack images.
            We expect to find path_to_focal_stack_folder/depth-00.png ... 
                                path_to_focal_stack_folder/depth-10.png here.
            depth-00.png, depth-10.png, etc are all the same size
    Outputs:
        path_to_focal_stack_folder/blurred-00.png
        ...
        path_to_focal_stack_folder/blurred-10.png
    """
    NUM_PNGS = 11
    for i in range(NUM_PNGS):
        i_str = "0"+str(i) if i<10 else str(i)
        depth_img_name = path_to_focal_stack_folder + "/depth-" + i_str + ".png"
        img = cv2.imread(depth_img_name)

        min_img_dim = min(img.shape[0], img.shape[1])
        blurred = cv2.GaussianBlur(img, (int(min_img_dim/36), int(min_img_dim/36)), sigmaX=0)

        outname = path_to_focal_stack_folder + "/blurred-" + i_str + ".png"
        cv2.imwrite(outname, blurred)


def getYCbCrImageFromRGB(img):
    """
    Given an RGB image, return YCbCr 
    Inputs:
        img: RGB
    Outputs:
        img_out: YCbCr
    Formula referenced from: https://www.mir.com/DMG/ycbcr.html
    """

    '''
    Old method -- uses formulas in question statement with no scaling
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Y = 0.3*R + 0.6*G + 0.1*B
    Cb = (B-Y)/1.8
    Cr = (R-Y)/1.4

    img_out[:, :, 0] = Y
    img_out[:, :, 1] = Cb
    img_out[:, :, 2] = Cr
    '''

    img_out = np.zeros(np.shape(img))  # to be modified, just get right shape fo now
    coding_matrix = np.array([[0.299, 0.587, 0.114], 
                                [-0.168736, -0.331264, 0.5], 
                                [0.5, -0.418688, -0.081312]])

    # Scale RGB
    r, g, b = img.astype(np.float32)[:,:,0]/255.0, img.astype(np.float32)[:,:,1]/255.0, img.astype(np.float32)[:,:,2]/255.0

    # Apply matrix transform
    y = coding_matrix[0,0]*r + coding_matrix[0,1]*g + coding_matrix[0,2]*b
    cb = coding_matrix[1,0]*r + coding_matrix[1,1]*g + coding_matrix[1,2]*b
    cr = coding_matrix[2,0]*r + coding_matrix[2,1]*g + coding_matrix[2, 2]*b


    # Return as matrix
    img_out[:, :, 0] = y
    img_out[:, :, 1] = cb
    img_out[:, :, 2] = cr
    return img_out

def getRGBImageFromYCbCr(img):
    """
    Given an YCbCr image, return RBG 
    Inputs:
        img: YCbCr
    Outputs:
        img_out: RGB
    Formula referenced from: https://www.mir.com/DMG/ycbcr.html
    """
    coding_matrix = np.array([[1, 0, 1.402], 
                            [1, -0.344136, -0.714136], 
                            [1.0, 1.772, 0]])
    
    
    img_out = np.zeros(np.shape(img))  # to be modified, just get right shape fo now

    # Scale YCbCr
    y, cb, cr = img.astype(np.float32)[:,:,0], img.astype(np.float32)[:,:,1], img.astype(np.float32)[:,:,2]

    # Apply matrix transform
    r = coding_matrix[0,0]*y + coding_matrix[0,1]*cb + coding_matrix[0,2]*cr
    g = coding_matrix[1,0]*y + coding_matrix[1,1]*cb + coding_matrix[1,2]*cr
    b = coding_matrix[2,0]*y + coding_matrix[2,1]*cb + coding_matrix[2,2]*cr

    # Scale RGB
    R = r*255.0
    G = g*255.0
    B = b*255.0

    # Return as matrix
    img_out[:, :, 0] = R
    img_out[:, :, 1] = G
    img_out[:, :, 2] = B
    return img_out


def generate_sharpened_focal_stack(path_to_focal_stack_folder):
    """
    Applies sharpening and color correction to each focal stack and saves the sharp version.

    Inputs:
        path_to_focal_stack_folder: Path to the folder containing the focal stack images.
            We expect to find path_to_focal_stack_folder/depth-00.png ... 
                                path_to_focal_stack_folder/depth-10.png here.
            depth-00.png, depth-10.png, etc are all the same size
    Outputs:
        path_to_focal_stack_folder/sharper-00.png
        ...
        path_to_focal_stack_folder/sharper-10.png
    """
    NUM_PNGS = 11
    for i in range(NUM_PNGS):
        i_str = "0"+str(i) if i<10 else str(i)
        depth_img_name = path_to_focal_stack_folder + "/depth-" + i_str + ".png"
        img = cv2.imread(depth_img_name)
        img_float = img.astype(np.float32) / 255.0

        # Luminance enhancement
        img_YCbCr = getYCbCrImageFromRGB(img_float)
        y, cb, cr = img_YCbCr[:, :, 0].astype(np.float32), img_YCbCr[:, :, 1].astype(np.float32), img_YCbCr[:, :, 2].astype(np.float32)
        correct_y = y**0.95   # brighten
        correct_cb = cb         # no change
        correct_cr = cr         # no change

        new_img_YCbCr = np.zeros(np.shape(img_YCbCr)).astype(np.float32)
        new_img_YCbCr[:, :, 0] = correct_y
        new_img_YCbCr[:, :, 1] = correct_cb
        new_img_YCbCr[:, :, 2] = correct_cr
        
        img = getRGBImageFromYCbCr(new_img_YCbCr)

        # Mild gamma lift
        img = img**0.85

        # Sharpen by subtracting off blurred version
        k = 0.5    # determines how much of blurred version to subtract off
        blur = gaussian_filter(img, sigma=1.5)
        img = img + k*(img - blur)

        # Rescale
        img = 255*np.clip(img, 0, 1)

        outname = path_to_focal_stack_folder + "/sharper-" + i_str + ".png"
        cv2.imwrite(outname, img)


''' Chess
generate_depth_selection_arr("plenoptic_images_depth_slices/Chess")
generate_blurred_focal_stack("plenoptic_images_depth_slices/Chess")
generate_sharpened_focal_stack("plenoptic_images_depth_slices/Chess")
#'''

#''' University
generate_depth_selection_arr("plenoptic_images_depth_slices/University_Processed")
generate_blurred_focal_stack("plenoptic_images_depth_slices/University_Processed")
generate_sharpened_focal_stack("plenoptic_images_depth_slices/University_Processed")
#'''