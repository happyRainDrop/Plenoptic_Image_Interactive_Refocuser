"""
interactive_viewer.py
!!! THIS FILE WAS NOT WRITTEN BY RUTH

This file was written with ChatGPT. It blends pre-computed
enhanced and blurred focal stacks depending on where the mouse is.

"""
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# ============================================================
# CONFIGURE YOUR STACKS HERE
# ============================================================

stack_folders = [
    "plenoptic_images_depth_slices/University_Processed",
    "plenoptic_images_depth_slices/Chess"
]

dropdown_items = [
    "University",
    "Chessboard"
]

NUM_DEPTHS = 11
RADIUS = 100


# ============================================================
# LOAD STACK
# ============================================================

def load_stack(folder):

    sharp_stack = []
    blurred_stack = []

    for i in range(NUM_DEPTHS):

        sharp = cv2.imread(
            os.path.join(folder, f"sharper-{i:02d}.png")
        )

        blur = cv2.imread(
            os.path.join(folder, f"blurred-{i:02d}.png")
        )

        if sharp is None:
            raise RuntimeError(
                f"Could not load {folder}/sharper-{i:02d}.png"
            )

        if blur is None:
            raise RuntimeError(
                f"Could not load {folder}/blurred-{i:02d}.png"
            )

        sharp_stack.append(sharp)
        blurred_stack.append(blur)

    chooser = np.load(
        os.path.join(folder, "depth_chooser.npz")
    )["depth_chooser"]

    return sharp_stack, blurred_stack, chooser


# ============================================================
# APP
# ============================================================
class FocusViewer:

    def __init__(self, root):

        self.root = root

        root.title("Interactive Focus Viewer")

        root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        # -------------------------
        # Dropdown
        # -------------------------

        self.selection = tk.StringVar()

        self.dropdown = ttk.Combobox(
            root,
            textvariable=self.selection,
            values=dropdown_items,
            state="readonly"
        )

        self.dropdown.current(0)
        self.dropdown.pack(
            fill="x",
            padx=5,
            pady=5
        )

        self.dropdown.bind(
            "<<ComboboxSelected>>",
            self.change_stack
        )

        # -------------------------
        # Radius slider
        # -------------------------

        self.radius_var = tk.IntVar(value=100)

        slider_frame = tk.Frame(root)
        slider_frame.pack(
            fill="x",
            padx=5,
            pady=5
        )

        tk.Label(
            slider_frame,
            text="Focus Radius"
        ).pack(side="left")

        self.radius_slider = tk.Scale(
            slider_frame,
            from_=20,
            to=500,
            orient="horizontal",
            variable=self.radius_var
        )

        self.radius_slider.pack(
            side="left",
            fill="x",
            expand=True
        )

        # -------------------------
        # Canvas
        # -------------------------

        self.canvas = tk.Canvas(
            root,
            highlightthickness=0,
            bg="black"
        )

        self.canvas.pack(
            fill="both",
            expand=True
        )

        self.canvas.bind(
            "<Motion>",
            self.mouse_move
        )

        root.bind(
            "<Escape>",
            lambda e: self.close()
        )

        root.attributes(
            "-fullscreen",
            True
        )

        self.display_scale = 1.0
        self.display_w = 1
        self.display_h = 1

        self.load_stack_by_index(0)

        self.mouse_x = self.W // 2
        self.mouse_y = self.H // 2

        self.update_frame()

    # =================================================

    def load_stack_by_index(self, idx):

        self.sharp_stack, self.blurred_stack, self.chooser = \
            load_stack(stack_folders[idx])

        self.H, self.W = \
            self.sharp_stack[0].shape[:2]

        self.chooser_H, self.chooser_W = \
            self.chooser.shape

        # precompute coordinate grids once
        self.yy, self.xx = np.mgrid[
            0:self.H,
            0:self.W
        ]

    # =================================================

    def change_stack(self, event=None):

        idx = self.dropdown.current()

        self.load_stack_by_index(idx)

        self.mouse_x = self.W // 2
        self.mouse_y = self.H // 2

    # =================================================

    def mouse_move(self, event):

        if self.display_scale <= 0:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        offset_x = (
            canvas_w - self.display_w
        ) // 2

        offset_y = (
            canvas_h - self.display_h
        ) // 2

        img_x = (
            event.x - offset_x
        ) / self.display_scale

        img_y = (
            event.y - offset_y
        ) / self.display_scale

        self.mouse_x = np.clip(
            int(img_x),
            0,
            self.W - 1
        )

        self.mouse_y = np.clip(
            int(img_y),
            0,
            self.H - 1
        )

    # =================================================

    def make_focus_image(self):

        chooser_x = int(
            self.mouse_x *
            self.chooser_W /
            self.W
        )

        chooser_y = int(
            self.mouse_y *
            self.chooser_H /
            self.H
        )

        chooser_x = np.clip(
            chooser_x,
            0,
            self.chooser_W - 1
        )

        chooser_y = np.clip(
            chooser_y,
            0,
            self.chooser_H - 1
        )

        depth_idx = int(
            self.chooser[
                chooser_y,
                chooser_x
            ]
        )

        sharp = self.sharp_stack[depth_idx]
        blur = self.blurred_stack[depth_idx]

        dist2 = (
            (self.xx - self.mouse_x) ** 2
            +
            (self.yy - self.mouse_y) ** 2
        )

        radius = self.radius_var.get()

        sigma = max(
            1.0,
            radius / 2.0
        )

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

        return np.clip(
            display,
            0,
            255
        ).astype(np.uint8)

    # =================================================

    def update_frame(self):

        display = self.make_focus_image()

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 50:
            canvas_w = self.root.winfo_screenwidth()

        if canvas_h < 50:
            canvas_h = self.root.winfo_screenheight()

        # leave margin for controls
        canvas_h -= 40

        scale = min(
            canvas_w / self.W,
            canvas_h / self.H
        )

        # safety margin
        scale *= 0.95

        self.display_scale = scale

        new_w = max(
            1,
            int(self.W * scale)
        )

        new_h = max(
            1,
            int(self.H * scale)
        )

        self.display_w = new_w
        self.display_h = new_h

        display = cv2.resize(
            display,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        display = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(display)

        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")

        self.canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self.tk_img,
            anchor="center"
        )

        self.root.after(
            16,
            self.update_frame
        )

    # =================================================

    def close(self):

        self.root.destroy()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FocusViewer(root)

    root.mainloop()