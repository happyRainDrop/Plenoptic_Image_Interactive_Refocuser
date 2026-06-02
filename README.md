# Introduction

This is a Python project that interactively focuses parts of a displayed plenoptic image.

Run `interactive_viewer.py`, then hover your mouse over the image in the pop-up window. You should see an illuminated, focused region around the cursor. A dropdown menu allows you to select which image to interact with.

# Workflow / How to Run

1. Clone this GitHub repository.

2. If you want to use the pre-computed images included in the repository, skip to Step 6. Otherwise, continue to Step 3.

3. Download a plenoptic image dataset:

   * Navigate to the plenoptic dataset.
   * Open the **R29 Raytrix Camera** folder.
   * Download:

     * `[desired_img]_Processed.png`
     * `[desired_img]_Depth.png`
   * Place both files into the `plenoptic_images_raw/` directory.

4. Generate the depth stack by running:

   ```bash
   python depth_slices_from_depth_map.py
   ```

   Modify the function call at the bottom of `depth_slices_from_depth_map.py`:

   ```python
   generate_depth_slices_new(
       "University_Processed",
       "University_Depth",
       np.linspace(0.58, 0.92, 11).tolist(),
       0.3,
       1.0
   )
   ```

   Replace the parameters as follows:

   * `"University_Processed"` → `"[desired_img]_Processed"`

   * `"University_Depth"` → `"[desired_img]_Depth"`

   * `np.linspace(0.58, 0.92, 11)` → `np.linspace(MIN_DIST, MAX_DIST, 11)`

     Where:

     * `MIN_DIST` controls the closest object that can be brought into focus.
     * `MAX_DIST` controls the farthest object that can be brought into focus.

     Adjust these values as needed for the selected image.

   * `0.3, 1.0`

     Experiment with these values to control how strongly the foreground and background blur when other depth layers are in focus.

5. Generate the blurred/sharpened focal stacks and depth-selection array by running:

   ```bash
   python refocus_aids.py
   ```

   Replace the final three lines of `refocus_aids.py` with:

   ```python
   generate_depth_selection_arr(
       "plenoptic_images_depth_slices/[desired_img]_Processed"
   )

   generate_blurred_focal_stack(
       "plenoptic_images_depth_slices/[desired_img]_Processed"
   )

   generate_sharpened_focal_stack(
       "plenoptic_images_depth_slices/[desired_img]_Processed"
   )
   ```

6. Launch the interactive viewer:

   ```bash
   python interactive_viewer.py
   ```

   If you added a new image from the dataset, update the lists near the top of `interactive_viewer.py`:

   ```python
   stack_folders = [
       "plenoptic_images_depth_slices/University_Processed",
       "plenoptic_images_depth_slices/Chess",
       "plenoptic_images_depth_slices/[desired_img]"  # ADD THIS
   ]

   dropdown_items = [
       "University",
       "Chessboard",
       "[desired_img]"  # ADD THIS
   ]
   ```

7. Run `interactive_viewer.py` and select your image from the dropdown menu.
