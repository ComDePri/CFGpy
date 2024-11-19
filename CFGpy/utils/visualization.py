import os
import tqdm

from rectpack import newPacker
import numpy as np
import sys
sys.path.append('C:\\Users\\Yogevhen\\Desktop\\Project\\simCFG')
sys.path.append('D:\\ComDePri\\ComDePy')
import simCFG
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.offsetbox import OffsetImage, AnnotationBbox, BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import partial
from matplotlib.text import Text

def animate_game(game, game_number, speed=1):
    fig = plt.figure()
    fps = 20
    text_pos = (10, 10)
    interval = int((1 / fps) * 1000)

    def update(frame, show_time, tqdm_obj):
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
        shape = simCFG.utils.get_shape_binary_matrix(int(shape))
        simCFG.utils.show_binary_matrix(shape, show=False, is_gallery=is_gallery, is_exploit=False, render=False, save_filename=None, title='', res=None, use_figure=fig)
        ax = plt.gca()
        if show_time:
            text = np.round(frame[1], 2).astype(str)
            if is_gallery:
                text = np.round(frame[2], 2).astype(str)
            ax.text(text_pos[0], text_pos[1], s=text)

    frames = []
    for action_index, action in enumerate(game['actions'][:-1]):
        time_to_create = action[1]
        time_to_save = action[2]

        next_shape_create_time = game['actions'][action_index + 1][1]
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

    action = game['actions'][-1]
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

    tqdm_obj = tqdm(total=len(frames))
    tqdm_update = partial(update, show_time=True, tqdm_obj=tqdm_obj)
    ani = animation.FuncAnimation(fig=fig, func=tqdm_update, frames=frames, interval=interval)

    if not os.path.isdir('games'):
        os.mkdir('games')
    path = 'games/game_{game_number}.gif'.format(game_number=game_number)
    ani.save(path)

def plot_game(game, game_number):
    cleaned_actions = np.array(remove_duplicate_actions(game))
    exploit_times = [range(*exploit_slice) for exploit_slice in game['exploit']]
    gallery_shapes = [[index, simCFG.utils.get_shape_binary_matrix(int(action[0])), action[2]] for index, action in enumerate(game['actions']) if action[2] is not None]
    len_shapes = len(gallery_shapes)
    cols = np.ceil(len_shapes**0.5).astype(int)
    fig, ax = plt.subplots(nrows=cols, ncols=cols, figsize = (16, 12))
    plt.suptitle('Game {game_number} Player {player_id}'.format(game_number=game_number, player_id=game['id']))
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

        curr_save_time = game['actions'][index][2]
        prev_save_time = game['actions'][prev_index][2]
        delta_t = curr_save_time - prev_save_time
        steps_between_shapes = np.where(cleaned_actions == curr_save_time)[0] - np.where(cleaned_actions == prev_save_time)[0]
        delta_t_and_steps.append([delta_t, steps_between_shapes[0]])
        res = (900/100, 900/100)
        shape_image = simCFG.utils.show_binary_matrix(shape, show=False, is_gallery=is_new_exploit, is_exploit=is_exploit, render=True, save_filename=None, title='', res=res)
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
    if not os.path.isdir('games'):
        os.mkdir('games')
    plt.savefig('games/game_{game_number}.png'.format(game_number=game_number))
    plt.close()
    return

# Move somewhere else when I finish with this
def show_community(community, community_number, subfolder=''):
    # unique_community_shapes = set([shape for community_set in community for shape in community_set])
    unique_community_shapes = community
    unique_community_shapes = [simCFG.utils.get_shape_binary_matrix(int(shape)) for shape in unique_community_shapes]

    len_community = len(unique_community_shapes)

    cols = np.ceil(len_community**0.5).astype(int)
    fig, ax = plt.subplots(nrows=cols, ncols=cols, figsize = (16, 12))
    for counter, shape in enumerate(unique_community_shapes):
        res = (900/100, 900/100)
        shape_image = simCFG.utils.show_binary_matrix(shape, show=False, is_gallery=False, is_exploit=False, render=True, save_filename=None, title='', res=res)
        ax.flat[counter].imshow(shape_image)
        x_position = 305 * (counter % cols)
        y_position = 305 * (counter // cols)
        ax.flat[counter].set_xticklabels([])
        ax.flat[counter].set_yticklabels([])
    
    for axis in ax.flat[counter + 1:]:
        axis.remove()
    
    plt.tight_layout()
    folder = os.path.normpath(os.path.join('communities', subfolder))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig('{folder}\\community_{community_number}.png'.format(folder=folder, community_number=community_number))
    plt.close()
    return

def remove_duplicate_actions(game):
    cleaned_actions = [game['actions'][0]]
    prev_action = game['actions'][0]
    for action in game['actions'][1:]:
        if action[2] is None and action[0] == prev_action[0]:
            continue
        else:
            cleaned_actions.append(action)    
        prev_action = action
    
    return cleaned_actions

def show_shape_from_size_dict(shapes_dict):
    min_size = 3
    shapes = [
        simCFG.utils.show_binary_matrix(simCFG.utils.get_shape_binary_matrix(int(shape)), show=False, is_gallery=False, is_exploit=False, render=True, save_filename=None, title='', res=(min_size + (size/20), min_size + (size/20)))
        for shape, size in shapes_dict.items()
    ]

    packer = newPacker()
    for im in shapes:
        packer.add_rect(*im.size)

    bin_size = np.sum(sorted([im.size[0] for im in shapes])[::-1][:int(len(shapes_dict) ** 0.5) + 1])
    packer.add_bin(bin_size, bin_size)
    packer.pack()
    bin = packer[0]
    rect_arr = np.array(packer.rect_list())
    rightmost_shape = rect_arr[np.argmax(rect_arr[:, 1])]
    highest_shape = rect_arr[np.argmax(rect_arr[:, 2])]
    fig_width = (rightmost_shape[1] + rightmost_shape[3])/100
    fig_height = (highest_shape[2] + highest_shape[4])/100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for index, pil_obj in enumerate(shapes):
        rect = [bin[index].x, bin[index].y, bin[index].width, bin[index].height]
        bbox = Bbox.from_bounds(*rect)
        bbox_image = BboxImage(bbox)
        bbox_image.set_data(pil_obj)
        fig.add_artist(bbox_image)
    
    plt.axis('off')
    
    return fig

def save_plot(fig, file_name, path, subfolder='', close=True):
    folder = os.path.normpath(os.path.join(path, subfolder))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    full_path = os.path.normpath(os.path.join(folder, file_name))

    fig.savefig(full_path)
    if close:
        plt.close(fig)
