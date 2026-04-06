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

def plot_game(game, output_dir_path="./"):
    game_id = game[PARSED_PLAYER_ID_KEY]
    all_actions = game[PARSED_ALL_SHAPES_KEY]

    cleaned_actions = remove_duplicate_actions(game)
    cleaned_save_times = [action[2] for action in cleaned_actions if action[2] is not None]
    save_time_to_clean_idx = {save_time: idx for idx, save_time in enumerate(cleaned_save_times)}

    # Precompute exploit phase membership once
    exploit_phase_by_index = {}
    for phase_idx, exploit_slice in enumerate(game[EXPLOIT_KEY]):
        for idx in range(*exploit_slice):
            exploit_phase_by_index[idx] = phase_idx

    gallery_shapes = [
        (index, get_shape_binary_matrix(int(action[0])), action[2])
        for index, action in enumerate(all_actions)
        if action[2] is not None
    ]

    len_shapes = len(gallery_shapes)
    if len_shapes == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Player {game_id}\nNo gallery shapes", ha="center", va="center")
        ax.axis("off")
        if not os.path.isdir(output_dir_path):
            os.mkdir(output_dir_path)
        fig.savefig(os.path.join(output_dir_path, f"game_{game_id}.png"), bbox_inches="tight")
        plt.close(fig)
        return

    cols = max(int(np.ceil(np.sqrt(len_shapes))),2)
    rows = max(int(np.ceil(len_shapes / cols)), 2)

    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * 2.4 + 2, rows * 2.4),
        squeeze=False,
    )
    plt.suptitle(f"Player {game_id}")

    prev_exploit_phase = -1
    prev_index = gallery_shapes[0][0]
    delta_t_and_steps = []

    for counter, (index, shape, save_time) in enumerate(gallery_shapes):
        axis = ax.flat[counter]

        exploit_phase = exploit_phase_by_index.get(index, None)
        is_exploit = exploit_phase is not None
        is_new_exploit = is_exploit and (exploit_phase > prev_exploit_phase)

        if is_exploit:
            prev_exploit_phase = exploit_phase

        curr_save_time = all_actions[index][2]
        prev_save_time = all_actions[prev_index][2]

        if counter == 0:
            delta_t_and_steps.append((None, None))
        else:
            delta_t = curr_save_time - prev_save_time
            steps_between_shapes = (
                save_time_to_clean_idx[curr_save_time] - save_time_to_clean_idx[prev_save_time]
            )
            delta_t_and_steps.append((delta_t, steps_between_shapes))

        draw_binary_matrix(
            axis,
            shape,
            is_gallery=is_new_exploit,
            is_exploit=is_exploit,
            title="",
        )
        axis.set_xlabel(f"{np.round(save_time, 3)}")

        prev_index = index

    for axis in ax.flat[len_shapes:]:
        axis.remove()

    # fig.tight_layout()
    # # leave extra space on the right of the same size as between axes gap:
    # gap_ratio = (fig.subplotpars.right - fig.subplotpars.left) / cols
    # fig.subplots_adjust(right=1 - gap_ratio)
    layout = compute_layout_params(cols, gap_ratio=0.5)

    fig.subplots_adjust(
        left=layout["left"],
        right=layout["right"],
        top=0.9,
        bottom=0.05,
        wspace=layout["wspace"],
        hspace=layout["wspace"],  # same vertically
    )


    for counter in range(1, len(gallery_shapes)):
        delta_t, steps_between_shapes = delta_t_and_steps[counter]

        pos = ax.flat[counter].get_position()
        prev_pos = ax.flat[counter - 1].get_position()

        if counter % cols != 0:
            x_pos = (pos.x0 + prev_pos.x1) / 2
            y_pos = (pos.y0 + prev_pos.y1) / 2
        else:
            x_pos = prev_pos.x1 + (prev_pos.x1 - prev_pos.x0) / 4
            y_pos = (prev_pos.y0 + prev_pos.y1) / 2

        ratio = np.round(steps_between_shapes / delta_t, 2) if delta_t not in (0, None) else np.nan
        fig.text(
            x_pos,
            y_pos,
            f"v={ratio}\nsbs={steps_between_shapes}\ndt={np.round(delta_t, 2)}",
            color="black",
            ha="center",
            va="center",
        )
    os.makedirs(output_dir_path, exist_ok=True)


    plt.savefig(os.path.join(output_dir_path, f"game_{game_id}.png"), bbox_inches="tight")
    plt.close()


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

