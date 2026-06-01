"""
!!! THIS FILE WAS NOT WRITTEN BY RUTH

This file was written with ChatGPT. It blends pre-computed
enhanced and blurred focal stacks depending on where the mouse is.

"""

# interactive_focus_viewer.py

import os
import cv2
import numpy as np


STACK_FOLDER = "plenoptic_images_depth_slices/Chess"


# ------------------
# load images
# ------------------

sharp_stack = []
blurred_stack = []

for i in range(11):

    sharp = cv2.imread(
        os.path.join(
            STACK_FOLDER,
            f"sharper-{i:02d}.png"
        )
    )

    blur = cv2.imread(
        os.path.join(
            STACK_FOLDER,
            f"blurred-{i:02d}.png"
        )
    )

    sharp_stack.append(sharp)
    blurred_stack.append(blur)

chooser = np.load(
    os.path.join(
        STACK_FOLDER,
        "depth_chooser.npz"
    )
)["depth_chooser"]

H, W = sharp_stack[0].shape[:2]

chooser_H, chooser_W = chooser.shape

mouse_x = W // 2
mouse_y = H // 2


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y

    mouse_x = x
    mouse_y = y


cv2.namedWindow("Focus Viewer")
cv2.setMouseCallback(
    "Focus Viewer",
    mouse_callback
)

RADIUS = 100


while True:

    chooser_x = int(
        mouse_x * chooser_W / W
    )

    chooser_y = int(
        mouse_y * chooser_H / H
    )

    chooser_x = np.clip(
        chooser_x,
        0,
        chooser_W - 1
    )

    chooser_y = np.clip(
        chooser_y,
        0,
        chooser_H - 1
    )

    depth_idx = chooser[
        chooser_y,
        chooser_x
    ]

    sharp = sharp_stack[depth_idx]
    blur = blurred_stack[depth_idx]

    # ------------------
    # radial mask
    # ------------------

    yy, xx = np.mgrid[
        0:H,
        0:W
    ]

    dist2 = (
        (xx - mouse_x) ** 2 +
        (yy - mouse_y) ** 2
    )

    sigma = RADIUS / 2

    mask = np.exp(
        -dist2 /
        (2 * sigma * sigma)
    )

    mask = mask[..., None]

    display = (
        mask * sharp.astype(np.float32)
        +
        (1.0 - mask)
        * blur.astype(np.float32)
    )

    display = np.clip(
        display,
        0,
        255
    ).astype(np.uint8)

    cv2.imshow(
        "Focus Viewer",
        display
    )

    key = cv2.waitKey(10)

    if key == 27:
        break

cv2.destroyAllWindows()