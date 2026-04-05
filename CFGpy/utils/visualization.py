import io
import os
import tqdm

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.text import Text
from functools import partial

from CFGpy.behavioral._consts import PARSED_ALL_SHAPES_KEY, PARSED_PLAYER_ID_KEY, EXPLOIT_KEY, VIS_SHAPE_COLOR, VIS_EXPLOIT_SHAPE_COLOR, VIS_SHAPE_BG_COLOR, VIS_GALLERY_BG_COLOR

from .utils import get_shape_binary_matrix

def animate_game(game, speed=1, output_dir_path='./', verbose=False):
    game_id = game[PARSED_PLAYER_ID_KEY]
    fig = plt.figure()
    fps = 20
    text_pos = (10, 10)
    interval = int((1 / fps) * 1000)

    def update(frame, show_time, tqdm_obj, verbose):
        if verbose:
            tqdm_obj.update(1)
        if type(frame) is not list:
            if show_time:
                ax = plt.gca()
                for match in ax.findobj(lambda artist: isinstance(artist, Text) and artist.get_position() == text_pos):
                    match.remove()
                text = frame
                ax.text(text_pos[0], text_pos[1], s=text)
            return

        fig.clear()
        shape = frame[0]
        is_gallery = frame[2] is not None
        shape = get_shape_binary_matrix(int(shape))
        show_binary_matrix(shape, show=False, is_gallery=is_gallery, is_exploit=False, render=False, save_filename=None, title=f'{game_id}', res=None, use_figure=fig)
        ax = plt.gca()
        if show_time:
            text = np.round(frame[1], 2).astype(str)
            if is_gallery:
                text = np.round(frame[2], 2).astype(str)
            ax.text(text_pos[0], text_pos[1], s=text)

    frames = []
    for action_index, action in enumerate(game[PARSED_ALL_SHAPES_KEY][:-1]):
        time_to_create = action[1]
        time_to_save = action[2]

        next_shape_create_time = game[PARSED_ALL_SHAPES_KEY][action_index + 1][1]
        if time_to_save is not None:
            dt_create = time_to_save - time_to_create
            total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
            frames += [[action[0], action[1], None]] + [np.round(float(time_to_create) + (i/fps)*speed, 2).astype(str) for i in range(1, total_frames_create)]

            dt_save = next_shape_create_time - time_to_save
            total_frames_save = np.ceil(dt_save * fps / speed).astype(int)
            frames += [action] + [np.round(float(time_to_save) + (i/fps)*speed, 2).astype(str) for i in range(1, total_frames_save)]
        else:
            dt_create = next_shape_create_time - time_to_create
            total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
            frames += [action] + [np.round(float(time_to_create) + (i/fps)*speed, 2).astype(str) for i in range(1, total_frames_create)]

    action = game[PARSED_ALL_SHAPES_KEY][-1]
    last_time = 720
    time_to_create = action[1]
    time_to_save = action[2]
    if time_to_save is not None:
        dt_create = time_to_save - time_to_create
        total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
        frames += [[action[0], action[1], None]] + [np.round(float(time_to_create) + (i/fps)*speed, 2).astype(str) for i in range(1, total_frames_create)]

        dt_save = last_time - time_to_save
        total_frames_save = np.ceil(dt_save * fps / speed).astype(int)
        frames += [action] + [np.round(float(time_to_save) + (i/fps)*speed, 2).astype(str) for i in range(1, total_frames_save)]
    else:
        dt_create = last_time - time_to_create
        total_frames_create = np.ceil(dt_create * fps / speed).astype(int)
        frames += [action] + [np.round(float(time_to_create) + (i/fps)*speed, 2).astype(str) for i in range(1, total_frames_create)]

    update_func = partial(update, show_time=True, tqdm_obj=None, verbose=False)
    if verbose:
        tqdm_obj = tqdm(total=len(frames))
        update_func = partial(update, show_time=True, tqdm_obj=tqdm_obj, verbose=True)
    ani = animation.FuncAnimation(fig=fig, func=update_func, frames=frames, interval=interval)

    if not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)
    path = os.path.join(output_dir_path, 'game_{game_id}.gif'.format(game_id=game_id))
    ani.save(path)

def plot_game(game, output_dir_path='./'):
    game_id = game[PARSED_PLAYER_ID_KEY]
    cleaned_actions = np.array(remove_duplicate_actions(game))
    exploit_times = [range(*exploit_slice) for exploit_slice in game[EXPLOIT_KEY]]
    gallery_shapes = [[index, get_shape_binary_matrix(int(action[0])), action[2]] for index, action in enumerate(game[PARSED_ALL_SHAPES_KEY]) if action[2] is not None]
    len_shapes = len(gallery_shapes)
    cols = np.ceil(len_shapes**0.5).astype(int)
    fig, ax = plt.subplots(nrows=cols, ncols=cols, figsize = (16, 12))
    plt.suptitle('Player {player_id}'.format(player_id=game_id))
    prev_exploit_time = -1
    prev_index = gallery_shapes[0][0]
    delta_t_and_steps = []
    for counter, shape_and_index in enumerate(gallery_shapes):
        index, shape, save_time = shape_and_index
        is_new_exploit = False
        exploit_time_index = np.nonzero([index in exploit_time for exploit_time in exploit_times])[0]
        is_exploit = exploit_time_index.size == 1
        if is_exploit:
            is_new_exploit =  exploit_time_index[0] - prev_exploit_time > 0
            prev_exploit_time = exploit_time_index[0]

        curr_save_time = game[PARSED_ALL_SHAPES_KEY][index][2]
        prev_save_time = game[PARSED_ALL_SHAPES_KEY][prev_index][2]
        delta_t = curr_save_time - prev_save_time
        steps_between_shapes = np.where(cleaned_actions == curr_save_time)[0] - np.where(cleaned_actions == prev_save_time)[0]
        delta_t_and_steps.append([delta_t, steps_between_shapes[0]])
        res = (900/100, 900/100)
        shape_image = show_binary_matrix(shape, show=False, is_gallery=is_new_exploit, is_exploit=is_exploit, render=True, save_filename=None, title='', res=res)
        ax.flat[counter].imshow(shape_image)
        ax.flat[counter].set_xlabel('{}'.format(np.round(save_time, 3)))

        ax.flat[counter].set_xticklabels([])
        ax.flat[counter].set_yticklabels([])

        prev_index = index
    
    for axis in ax.flat[counter + 1:]:
        axis.remove()

    fig.tight_layout()
    for counter, _ in enumerate(gallery_shapes[1:]):
        delta_t, steps_between_shapes = delta_t_and_steps[counter + 1]
        pos = ax.flat[counter + 1].get_position()
        prev_pos = ax.flat[counter].get_position()
        if (counter + 1) % cols != 0:
            x_pos = (pos.x0 + prev_pos.x1) / 2
            y_pos = (pos.y0 + prev_pos.y1) / 2

        else:
            x_pos = (prev_pos.x1) + (prev_pos.x1 - prev_pos.x0) / 4
            y_pos = (prev_pos.y0 + prev_pos.y1) / 2

        fig.text(x_pos, y_pos, 'v={ratio}\nsbs={sbs}\ndt={dt}'.format(ratio=np.round(steps_between_shapes/delta_t, 2), dt=np.round(delta_t, 2), sbs=steps_between_shapes), color='black', ha='center', va='center')
    
    fig.set_size_inches(fig.get_size_inches()[0] + 2, fig.get_size_inches()[1])
    if not os.path.isdir(output_dir_path):
        os.mkdir(output_dir_path)

    fig.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(output_dir_path, 'game_{game_id}.png'.format(game_id=game_id)), bbox_inches='tight')
    plt.close()

    return

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

def show_binary_matrix(binary_mat, show=True, is_gallery=False, is_exploit=False, render=False, save_filename=None, title='', res=(750/100, 750/100), use_figure=None):
    """
    Displays the binary matrix representation of a shape.
    :param binary_mat: a binary matrix representation of a shape.
    :param is_gallery: True iff this a gallery shape. affects background color.
    :param save_filename: a filename to save the image, or None (to avoid saving).
    """
    bg_color = VIS_GALLERY_BG_COLOR if is_gallery else VIS_SHAPE_BG_COLOR
    shape_color = VIS_SHAPE_COLOR if not is_exploit else VIS_EXPLOIT_SHAPE_COLOR
    nrow, ncol = binary_mat.shape
    pad_rows = (10 - nrow) / 2
    pad_rows = (np.ceil(pad_rows).astype('int'), np.floor(pad_rows).astype('int'))
    pad_cols = (10 - ncol) / 2
    pad_cols = (np.ceil(pad_cols).astype('int'), np.floor(pad_cols).astype('int'))
    binary_mat = np.pad(binary_mat, (pad_rows, pad_cols))

    if use_figure is None:
        dpi = 100
        fig = plt.figure(figsize=res, dpi=dpi)

    ax = plt.gca()
    ax.matshow(binary_mat, cmap=ListedColormap([bg_color, shape_color]))

    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))

    ax.set_xticklabels(['' for i in np.arange(1, 11, 1)])
    ax.set_yticklabels(['' for i in np.arange(1, 11, 1)])

    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which='minor', color=bg_color, linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title(title)

    if save_filename:
        plt.savefig(save_filename)
    
    if show:
        plt.show()

    if render:
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img

    if use_figure is None:
        plt.close(fig)