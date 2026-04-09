import os
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.text import Text
from functools import partial

from CFGpy.behavioral._consts import (
    PARSED_ALL_SHAPES_KEY,
    PARSED_PLAYER_ID_KEY,
    EXPLOIT_KEY,
    VIS_SHAPE_COLOR,
    VIS_EXPLOIT_SHAPE_COLOR,
    VIS_SHAPE_BG_COLOR,
    VIS_GALLERY_BG_COLOR,
)

from .utils import get_shape_binary_matrix


# ---------- Fast helpers ----------

def _pad_binary_matrix(binary_mat, canvas_size=10):
    """Pad a binary matrix to a centered fixed-size square canvas."""
    nrow, ncol = binary_mat.shape
    if nrow > canvas_size or ncol > canvas_size:
        raise ValueError(f"Shape size {binary_mat.shape} exceeds canvas size {canvas_size}")

    pad_rows = (canvas_size - nrow) / 2
    pad_rows = (int(np.ceil(pad_rows)), int(np.floor(pad_rows)))
    pad_cols = (canvas_size - ncol) / 2
    pad_cols = (int(np.ceil(pad_cols)), int(np.floor(pad_cols)))

    return np.pad(binary_mat, (pad_rows, pad_cols))


def draw_binary_matrix(
    ax,
    binary_mat,
    *,
    is_gallery=False,
    is_exploit=False,
    title="",
    canvas_size=10,
):
    """
    Draw a shape directly onto an existing axis.
    Much faster than rendering to an intermediate figure/image buffer.
    """
    bg_color = VIS_GALLERY_BG_COLOR if is_gallery else VIS_SHAPE_BG_COLOR
    shape_color = VIS_EXPLOIT_SHAPE_COLOR if is_exploit else VIS_SHAPE_COLOR

    padded = _pad_binary_matrix(binary_mat, canvas_size=canvas_size)

    ax.imshow(
        padded,
        cmap=ListedColormap([bg_color, shape_color]),
        interpolation="none",
        vmin=0,
        vmax=1,
        origin="upper",
    )

    # gridlines
    ax.set_xticks(np.arange(-0.5, canvas_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, canvas_size, 1), minor=True)
    ax.grid(which="minor", color=bg_color, linestyle="-", linewidth=3)

    # hide axes labels/ticks
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)

    return ax


def show_binary_matrix(
    binary_mat,
    show=True,
    is_gallery=False,
    is_exploit=False,
    save_filename=None,
    title="",
    res=(750 / 100, 750 / 100),
    use_figure=None,
    ax=None,
):
    """
    Backward-compatible version that now draws directly instead of rendering to JPG/PIL.
    If ax is provided, draw into that axis.
    Otherwise create a new figure and axis.
    """
    created_fig = False

    if ax is None:
        if use_figure is None:
            fig, ax = plt.subplots(figsize=res, dpi=100)
            created_fig = True
        else:
            fig = use_figure
            fig.clear()
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    draw_binary_matrix(
        ax,
        binary_mat,
        is_gallery=is_gallery,
        is_exploit=is_exploit,
        title=title,
    )

    if save_filename:
        fig.savefig(save_filename, bbox_inches="tight")

    if show:
        plt.show()

    if created_fig:
        plt.close(fig)

    return ax


# ---------- Existing animation, updated to use direct axis drawing ----------

def animate_game(game, speed=1, output_dir_path="./", verbose=False):
    game_id = game[PARSED_PLAYER_ID_KEY]
    fig, ax = plt.subplots()
    fps = 20
    text_pos = (0.02, 0.95)  # axis coordinates
    interval = int((1 / fps) * 1000)

    time_text = ax.text(
        text_pos[0],
        text_pos[1],
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
    )

    def update(frame, show_time, tqdm_obj, verbose):
        if verbose:
            tqdm_obj.update(1)

        if not isinstance(frame, list):
            if show_time:
                time_text.set_text(str(frame))
            return

        ax.clear()
        shape = frame[0]
        is_gallery = frame[2] is not None
        shape = get_shape_binary_matrix(int(shape))

        draw_binary_matrix(
            ax,
            shape,
            is_gallery=is_gallery,
            is_exploit=False,
            title=f"{game_id}",
        )

        if show_time:
            text = np.round(frame[1], 2).astype(str)
            if is_gallery:
                text = np.round(frame[2], 2).astype(str)
            ax.text(
                text_pos[0],
                text_pos[1],
                s=text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
            )

    frames = []
    for action_index, action in enumerate(game[PARSED_ALL_SHAPES_KEY][:-1]):
        time_to_create = action[1]
        time_to_save = action[2]

        next_shape_create_time = game[PARSED_ALL_SHAPES_KEY][action_index + 1][1]
        if time_to_save is not None:
            dt_create = time_to_save - time_to_create
            total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
            frames += [[action[0], action[1], None]] + [
                np.round(float(time_to_create) + (i / fps) * speed, 2).astype(str)
                for i in range(1, total_frames_create)
            ]

            dt_save = next_shape_create_time - time_to_save
            total_frames_save = np.ceil(dt_save * fps / speed).astype(int)
            frames += [action] + [
                np.round(float(time_to_save) + (i / fps) * speed, 2).astype(str)
                for i in range(1, total_frames_save)
            ]
        else:
            dt_create = next_shape_create_time - time_to_create
            total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
            frames += [action] + [
                np.round(float(time_to_create) + (i / fps) * speed, 2).astype(str)
                for i in range(1, total_frames_create)
            ]

    action = game[PARSED_ALL_SHAPES_KEY][-1]
    last_time = 720
    time_to_create = action[1]
    time_to_save = action[2]
    if time_to_save is not None:
        dt_create = time_to_save - time_to_create
        total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
        frames += [[action[0], action[1], None]] + [
            np.round(float(time_to_create) + (i / fps) * speed, 2).astype(str)
            for i in range(1, total_frames_create)
        ]

        dt_save = last_time - time_to_save
        total_frames_save = np.ceil(dt_save * fps / speed).astype(int)
        frames += [action] + [
            np.round(float(time_to_save) + (i / fps) * speed, 2).astype(str)
            for i in range(1, total_frames_save)
        ]
    else:
        dt_create = last_time - time_to_create
        total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
        frames += [action] + [
            np.round(float(time_to_create) + (i / fps) * speed, 2).astype(str)
            for i in range(1, total_frames_create)
        ]

    update_func = partial(update, show_time=True, tqdm_obj=None, verbose=False)
    if verbose:
        tqdm_obj = tqdm.tqdm(total=len(frames))
        update_func = partial(update, show_time=True, tqdm_obj=tqdm_obj, verbose=True)

    ani = animation.FuncAnimation(fig=fig, func=update_func, frames=frames, interval=interval)
    os.makedirs(output_dir_path, exist_ok=True)
    path = os.path.join(output_dir_path, f"game_{game_id}.gif")
    ani.save(path)


# ---------- Faster plot_game ----------
def compute_layout_params(cols, gap_ratio=0.5, left_margin=0.05, right_margin=0.05):
    """
    Compute wspace and subplot region so that:
    - gap between subplots = gap_ratio * subplot width
    - right margin = one gap
    """

    # Effective usable width
    usable_width = 1 - left_margin - right_margin

    # Let subplot width = W
    # Total width = cols*W + (cols-1)*gap + right_gap
    # gap = gap_ratio * W
    # right_gap = gap_ratio * W

    # total = cols*W + (cols-1)*gap_ratio*W + gap_ratio*W
    #       = W * (cols + cols*gap_ratio)

    total_units = cols + cols * gap_ratio
    W = usable_width / total_units

    gap = gap_ratio * W

    # Convert to matplotlib wspace definition:
    # wspace = gap / W
    wspace = gap / W  # == gap_ratio

    # Adjust right margin to enforce final gap
    right = left_margin + cols * W + (cols - 1) * gap + gap

    return {
        "left": left_margin,
        "right": right,
        "wspace": wspace,
    }


def remove_duplicate_actions(game):
    cleaned_actions = [game[PARSED_ALL_SHAPES_KEY][0]]
    prev_action = game[PARSED_ALL_SHAPES_KEY][0]
    for action in game[PARSED_ALL_SHAPES_KEY][1:]:
        if action[2] is None and action[0] == prev_action[0]:
            continue
        else:
            cleaned_actions.append(action)
        prev_action = action

    return cleaned_actions

def plot_game(game, output_dir_path="./"):
    game_id = game[PARSED_PLAYER_ID_KEY]
    all_actions = game[PARSED_ALL_SHAPES_KEY]
    os.makedirs(output_dir_path, exist_ok=True)

    # ==========================================
    # 1. Prepare Data for the Graph (Top)
    # ==========================================
    exploit_shape_ranges = game[EXPLOIT_KEY]
    gallery_shape_times_and_indices = np.array([[idx, action[0], action[2]] for idx, action in enumerate(all_actions) if action[2] is not None])
    if gallery_shape_times_and_indices.size == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Player {game_id}\nNo gallery shapes", ha="center", va="center")
        ax.axis("off")
        fig.savefig(os.path.join(output_dir_path, f"game_{game_id}.png"), bbox_inches="tight")
        plt.close(fig)
        return

    gallery_save_indices = gallery_shape_times_and_indices[:, 0]
    gallery_save_shapes = gallery_shape_times_and_indices[:, 1]
    gallery_save_times = gallery_shape_times_and_indices[:, 2]
    gallery_shape_dts = np.diff(gallery_save_times, prepend=gallery_save_times[0])
    duplicate_actions_indices = [idx for idx in range(1, len(all_actions)) if all_actions[idx][2] is None and all_actions[idx][0] == all_actions[idx - 1][0]]
    corrected_gallery_save_indices = gallery_save_indices.copy()
    for indices in [np.where(gallery_save_indices > duplicate_action_index) for duplicate_action_index in duplicate_actions_indices]:
        corrected_gallery_save_indices[indices] -= 1

    steps_between_shapes = np.diff(corrected_gallery_save_indices, prepend=corrected_gallery_save_indices[0])

    # To avoid division by zero or near-zero, we can set a minimum threshold for delta_t when calculating velocity. If delta_t is below this threshold, we can set velocity to NaN or some predefined value.
    gallery_save_velocity = np.round(np.divide(steps_between_shapes, gallery_shape_dts, out=np.zeros_like(steps_between_shapes), where=gallery_shape_dts!=0), 2)
    gallery_shape_dts = np.round(gallery_shape_dts, 2)
    gallery_save_velocity[gallery_shape_dts == 0] = np.nan

    is_exploit = np.zeros_like(steps_between_shapes)
    is_start_of_exploit_phase = np.zeros_like(steps_between_shapes)
    
    exploit_bouts = []
    for exploit_range in exploit_shape_ranges:
        # Gets the new indices of the gallery saves that are in the current exploit range, and adds them as a bout
        exploit_range_indices = list(range(*exploit_range))
        exploit_bout = np.where(np.isin(gallery_save_indices, exploit_range_indices))[0]
        is_exploit[exploit_bout] = 1
        exploit_bouts.append(exploit_bout)
        if len(exploit_bout) > 0:
            is_start_of_exploit_phase[exploit_bout[0]] = 1

    # ==========================================
    # 2. Prepare Data for the Grid (Bottom)
    # ==========================================
    cleaned_actions = remove_duplicate_actions(game)
    cleaned_save_times = [action[2] for action in cleaned_actions if action[2] is not None]
    save_time_to_clean_idx = {save_time: idx for idx, save_time in enumerate(cleaned_save_times)}

    gallery_shapes = [
        (index, get_shape_binary_matrix(int(action[0])), action[2])
        for index, action in enumerate(all_actions)
        if action[2] is not None
    ]

    # ==========================================
    # 3. Figure & Subfigure Setup
    # ==========================================
    cols = max(int(np.ceil(np.sqrt(gallery_save_indices.size))), 2)
    rows = max(int(np.ceil(gallery_save_indices.size / cols)), 2)

    grid_width = cols * 2.4 + 2
    grid_height = rows * 2.4
    graph_height = 6.0 

    # Overall figure size accommodates both plots
    fig_width = max(10, grid_width)
    fig_height = graph_height + grid_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(f"Player {game_id}", fontsize=16, y=1.02)

    # Split the main figure into two isolated subfigures (Top and Bottom)
    subfigs = fig.subfigures(2, 1, height_ratios=[graph_height, grid_height])
    subfig_top = subfigs[0]
    subfig_bottom = subfigs[1]

    # ==========================================
    # 4. Plot Top: The Line Graph
    # ==========================================
    ax_graph = subfig_top.subplots()

    ax_graph.scatter(gallery_save_times, gallery_shape_dts, marker='*', color='orange', label='Explore Saves')

    for exploit_bout in exploit_bouts:
        ax_graph.plot(gallery_save_times[exploit_bout], gallery_shape_dts[exploit_bout], marker='o')
    
    ax_graph.set_xlabel('Save Time [$s$]')
    ax_graph.set_ylabel('$\\Delta$ t between gallery saves')
    ax_graph.set_title("Gallery Shape Saves Over Time vs $\\Delta$ t")

    # ==========================================
    # 5. Plot Bottom: The Image Grid
    # ==========================================
    ax_bottom = subfig_bottom.subplots(
        nrows=rows,
        ncols=cols,
        squeeze=False,
    )

    for save_index, shape in enumerate(gallery_save_shapes):
        binary_mat = get_shape_binary_matrix(int(shape))
        axis = ax_bottom.flat[save_index]
        is_shape_exploit = is_exploit[save_index]
        is_shape_exploit_start = is_start_of_exploit_phase[save_index]

        draw_binary_matrix(
            ax=axis,
            binary_mat=binary_mat,
            is_gallery=is_shape_exploit_start, # We're abusing the is_gallery parameter to indicate the start of an exploit phase
            is_exploit=is_shape_exploit,
            title="",
        )
        axis.set_xlabel(f"{np.round(gallery_save_times[save_index], 3)}")

    # Remove extra axes
    for axis in ax_bottom.flat[gallery_save_indices.size:]:
        axis.remove()

    # Isolate the layout adjustment to ONLY the bottom subfigure
    layout = compute_layout_params(cols, gap_ratio=0.5)
    subfig_bottom.subplots_adjust(
        left=layout["left"],
        right=layout["right"],
        top=0.9,
        bottom=0.05,
        wspace=layout["wspace"],
        hspace=layout["wspace"],
    )

    # Calculate text placement (Using subfig_bottom.text keeps your math valid)
    for counter in range(1, gallery_save_shapes.size):
        steps_between_shapes_for_shape = int(steps_between_shapes[counter])
        delta_t_for_shape = gallery_shape_dts[counter]
        velocity_for_shape = gallery_save_velocity[counter]

        pos = ax_bottom.flat[counter].get_position()
        prev_pos = ax_bottom.flat[counter - 1].get_position()

        if counter % cols != 0:
            x_pos = (pos.x0 + prev_pos.x1) / 2
            y_pos = (pos.y0 + prev_pos.y1) / 2
        else:
            x_pos = prev_pos.x1 + (prev_pos.x1 - prev_pos.x0) / 4
            y_pos = (prev_pos.y0 + prev_pos.y1) / 2

        subfig_bottom.text(
            x_pos,
            y_pos,
            f"v={velocity_for_shape}\nsbs={steps_between_shapes_for_shape}\ndt={delta_t_for_shape}",
            color="black",
            ha="center",
            va="center",
        )

    # ==========================================
    # 6. Save the Combined Figure
    # ==========================================
    plt.savefig(os.path.join(output_dir_path, f"game_{game_id}_combined.png"), bbox_inches="tight")
    plt.close(fig)