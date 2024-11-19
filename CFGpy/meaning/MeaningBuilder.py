import sys
sys.path.append('C:\\Users\\Yogevhen\\Desktop\\Project\\simCFG')
sys.path.append('D:\\ComDePri\\ComDePy')
import os
import simCFG
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.text import Text

import pandas as pd
import numpy as np
import itertools
import networkx as nx
from networkx.readwrite import json_graph
import json

from tqdm import tqdm
from itertools import groupby, combinations
from collections import Counter
from .. import behavioral
from ..behavioral import _consts as consts
from .. import utils
from functools import partial

CLUSTER_GRAPH = r'cluster_grpah.json'
CONNECTED_CLUSTERS = r'connected_clusters.json'
MIN_INTERSECT_TO_CONNECT_CLUSTERS = 4
MIN_SHAPES_FOR_CLUSTER = 3
MIN_COUNT_TO_SHOW = 5
N_TOP_COMMUNITIES = 8

def animate_game(game, game_number, speed=1):
    fig = plt.figure()
    fps = 20
    text_pos = (10, 10)
    interval = int((1 / fps) * 1000)
    total_frames = 12*60 * fps / speed

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
        # missing last guy
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


def get_all_clusters(games):
    '''
        Takes a list of CFG games, outputs a list of lists of clusters per game
        Where cluster is a list of gallery shapes saved during an exploit bout, if the amount of shapes saved is above a minimal level.
    '''
    all_clusters = []
    for game in games:
        actions = game[consts.PARSED_ALL_SHAPES_KEY]
        
        exploit_actions = [
            actions[cluster_range[0]:cluster_range[1]] for cluster_range in game[consts.EXPLOIT_KEY]
        ]

        clusters = [
            cluster[~pd.isna(cluster[:, consts.SHAPE_SAVE_TIME_IDX])][:, consts.SHAPE_ID_IDX] for cluster in exploit_actions
        ]
        clusters = [
            cluster.tolist() for cluster in clusters if cluster.shape[0] > MIN_SHAPES_FOR_CLUSTER
        ]
        all_clusters += clusters

    return all_clusters

def get_all_shapes_in_games(games):
    all_shapes = []
    for game in games:
        actions = np.array(game[consts.PARSED_ALL_SHAPES_KEY])
        all_shapes += actions[:, consts.SHAPE_ID_IDX].tolist()
    
    all_shapes = set(all_shapes)

    return all_shapes

def get_all_shapes():
    return set(range(1, utils.N_ALL_SHAPES))

def get_all_gallery_shapes(games):
    all_gallery_shapes = []
    for game in games:
        actions = game[consts.PARSED_ALL_SHAPES_KEY]
        exploit_actions = [
            actions[cluster_range[0]:cluster_range[1]] for cluster_range in game[consts.EXPLOIT_KEY]
        ]
        clusters = [
            cluster[~pd.isna(cluster[:,consts.SHAPE_SAVE_TIME_IDX])][:, consts.SHAPE_ID_IDX] for cluster in exploit_actions
        ]

        all_gallery_shapes += [int(shape) for cluster in clusters for shape in cluster]
    
    all_gallery_shapes = set(all_gallery_shapes)

    return all_gallery_shapes

def find_connected_clusters(clusters):
    pairs = [
        (tuple(pair[0]),tuple(pair[1])) for pair in itertools.product(clusters,clusters)
        if np.intersect1d(pair[0], pair[1]).shape[0] >= MIN_INTERSECT_TO_CONNECT_CLUSTERS and pair[0] != pair[1]
    ]

    return pairs

def find_largest_connected_component(clusters):
    return

def build_meaning_from_cluster_community(sorted_cluster_community):
    top_communities = sorted_cluster_community[:N_TOP_COMMUNITIES]
    return

def find_common_shapes_in_community(community, n):
    unraveled_community = [
        int(shape) for subcommunity in community for node in subcommunity for shape in node
    ]
    shape_counter = Counter(unraveled_community)

    [shape for shape in shape_counter if shape_counter[shape] >= n]

def count_shapes_in_many_communities(communities):
    counters = []
    all_shapes = []
    for community in communities:
        unraveled_community = [
            int(shape) for shapes in community for shape in shapes
        ]
        shape_counter = Counter(unraveled_community)
        all_shapes += list(shape_counter.keys())
        counters.append(shape_counter)
    
    all_shapes = set(all_shapes)

    return {
        shape: [counter[shape] for counter in counters] for shape in all_shapes
    }

def is_core_shape(shape_counts):
    return np.nonzero(shape_counts)[0].shape[0] == 1

def build_gallery_graph(games, flatten):
    '''
        Build a graph from the exploit clusters,
        Vertices are the shapes, if two shapes are adjacent to one another in a cluster, they have an edge between them
        Example:
            clusters: [(1, 5, 19, 100), (17, 19, 1, 42)]
            So the vertices/nodes are [1, 5, 17, 19, 42, 100]
            And the edges are [(1, 5), (5, 19), (19, 100), (17, 19), (19, 1), (1, 42)]
    '''
    edge_list = []
    for game in games:
        game_clusters = get_all_clusters([game])
        if flatten:
            flattened_shapes = [
                shape for cluster in game_clusters for shape in cluster
            ]
            edge_list += list(zip(flattened_shapes, flattened_shapes[1:]))
        else:
            for cluster in game_clusters:
                edge_list += list(zip(cluster, cluster[1:]))

    return nx.from_edgelist(edge_list)

def build_meaning(all_galleries_connected, gallery_meaning_scores, gallery_graph_connected, distance_normalizer=10):
    # all_galleries_connected = gallery shapes
    # gallery_meaning_scores = core shapes and their meaning scores
    # gallery_graph_connected = graph we're doing the search on

    all_shapes_meanings = []
    for shape in all_galleries_connected:
        core_meaning_and_distance_from_shape = [
            [gallery_meaning_scores[core_shape], nx.shortest_path_length(gallery_graph_connected, core_shape, shape)]
            for core_shape in gallery_meaning_scores
        ]
        core_meaning_and_distance_from_shape = sorted(core_meaning_and_distance_from_shape, key=lambda x: x[1])
        distances = groupby(core_meaning_and_distance_from_shape, lambda x: x[1])
        shape_meaning = []
        for distance, group in distances:
            meaning_scores_for_distance = [core_shape[0] for core_shape in group]
            factor = (distance_normalizer**-distance) / len(meaning_scores_for_distance)
            shape_meaning.append(factor * np.sum(meaning_scores_for_distance, axis=0))
        shape_meaning = np.sum(shape_meaning, axis=0)
        all_shapes_meanings.append([shape, shape_meaning])
    
    return all_shapes_meanings

def load_connected_clusters(verbose=True):
    try:
        with open(CONNECTED_CLUSTERS, 'r') as f:
            connected_clusters = json.load(f)
            return [(tuple(connected_cluster[0]), tuple(connected_cluster[1])) for connected_cluster in connected_clusters]

    except FileNotFoundError:
        if verbose:
            print('Connected clusters not found, recalculating')
        with open('fixed_vanilla.json', 'r') as fp:
            parsed_vanilla = json.load(fp)

        for game in parsed_vanilla:
            game['actions'] = np.array([action for action in game['actions']])
        clusters = get_all_clusters(parsed_vanilla) # Move from parsed data form to exploit cluster form
        connected_clusters = find_connected_clusters(clusters) # Each exploit cluster is treated like a vertex, a node

        if verbose:
            print('Saving connected clusters calculation')
        with open(CONNECTED_CLUSTERS, 'w') as f:
            json.dump(connected_clusters, f)
    
    return connected_clusters

def load_cluster_graph(verbose=False):
    '''
        Builds a graph from exploit clusters/bouts, an edge exists between clusters that share shapes
        The weight is the amount of shared shapes
    '''
    try:
        with open(CLUSTER_GRAPH, 'r') as f:
            G = json_graph.node_link_graph(json.load(f))
            return G

    except FileNotFoundError:
        if verbose:
            print('Clusters graph not found, recalculating')
        with open('fixed_vanilla.json', 'r') as fp:
            parsed_vanilla = json.load(fp)

        clusters = get_all_clusters(parsed_vanilla)
        G = nx.Graph()
        for pair in itertools.product(clusters,clusters):
            weight = np.intersect1d(pair[0], pair[1]).shape[0]
            if pair[0] != pair[1] and weight > 0:
                G.add_edge(tuple(pair[0]), tuple(pair[1]), weight=weight)

        if verbose:
            print('Saving connected clusters calculation')
        with open(CLUSTER_GRAPH, 'w') as fp:
            json.dump(json_graph.node_link.node_link_data(G), fp)

    return G

def build_shared_cluster_graph(parsed_data):
    clusters = get_all_clusters(parsed_data)
    G = nx.Graph()
    counter = 0
    
    for cluster in tqdm(clusters):
        for edge in combinations(cluster, 2):
            counter += 1
            if G.has_edge(*edge):
                number_of_shared_clusters = G.get_edge_data(*edge).get('shared_clusters')
                number_of_shared_clusters += 1
                attrs = {
                    edge: {'shared_clusters': number_of_shared_clusters}
                }
                nx.set_edge_attributes(G, attrs)
            else:
                G.add_edge(*edge, shared_clusters=1)

    return G

def test_community_finding_by_shape(parsed_data):
    shared_clusters_graph = build_shared_cluster_graph(parsed_data)
    shared_clusters = nx.get_edge_attributes(shared_clusters_graph, 'shared_clusters')
    for edge in shared_clusters_graph.edges:
        weight = shared_clusters[edge]
        attrs = {
            edge: {'nonlinear_weight': 10**(weight-1)}
        }
        nx.set_edge_attributes(shared_clusters_graph, attrs)
    shared_clusters_graph.remove_edges_from([edge for edge in shared_clusters_graph.edges if shared_clusters[edge] == 1])
    numbers = [2699, 12919, 16384, 19464, 3887, 16145, 3985, 8511, 5657, 2860, 12005, 3900, 2673]
    number_edges = combinations(numbers, 2)
    communities = [community for community in nx.community.louvain_communities(shared_clusters_graph, resolution=1, weight='nonlinear_weight')]
    offset = 0
    print('Saving all communities')
    for ind, community in tqdm(enumerate(communities[offset:])):
        show_community(community, ind + offset)

def plot_all_games():
    with open('fixed_vanilla.json', 'r') as fp:
        parsed_vanilla = json.load(fp)
        for counter, game in tqdm(enumerate(parsed_vanilla[:4])):
            animate_game(game, counter, speed=8)
        
def build_adjacency_graph(games):
    G = nx.Graph()
    for game in games:
        gallery_shapes = [action[0] for action in game['actions'] if action[2] is not None]
        for edge in zip(gallery_shapes[:-1], gallery_shapes[1:]):
            if G.has_edge(*edge):
                weight = G.get_edge_data(*edge).get('weight')
                weight += 1
                attrs = {
                    edge: {'weight': weight}
                }
                nx.set_edge_attributes(G, attrs)
            else:
                G.add_edge(*edge, weight=1)

    return G

def prune_edges(G, min_attr_for_node, attr):
    attr_for_edge = nx.get_edge_attributes(G, attr)
    edges_for_pruning = []
    for edge in G.edges:
        if attr_for_edge[edge] < min_attr_for_node:
            edges_for_pruning.append(edge)

    G.remove_edges_from(edges_for_pruning)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G

def total_node_weight(G, attr, node):
    weights = nx.get_edge_attributes(G, attr)
    all_node_weights = [
        weights[edge] if edge in weights else weights[edge[::-1]] for edge in G.edges(node)
    ]
    return np.sum(all_node_weights)


def transform_graph_attribute(G, attr, func):
    edge_attribute = nx.get_edge_attributes(G, attr)
    new_attributes = {}
    for edge in G.edges:
        new_attributes[edge] = {attr: func(edge_attribute[edge])}

    nx.set_edge_attributes(G, new_attributes)

def build_clusters_and_communities(verbose=True):
    if verbose:
        print('Loading vanilla data')
    with open('fixed_vanilla.json', 'r') as fp:
        parsed_vanilla = json.load(fp)
        for game in parsed_vanilla:
            game['actions'] = np.array([action for action in game['actions']])

    # parsed_vanilla = utils.get_vanilla() # Imports the vanilla data (Currently uses a faulty vanilla with the tutorial in it)
    if verbose:
        print('Loading connected clusters data')
    all_clusters = get_all_clusters(parsed_vanilla)
    #test_community_finding_by_shape(parsed_vanilla)
    # connected_clusters_graph = load_cluster_graph(parsed_vanilla)
    connected_clusters_graph = build_adjacency_graph(parsed_vanilla)
    subfolder = 'adjacency_communities_girvin_translation_minus2_res_3_5'
    prune_edges(connected_clusters_graph, min_attr_for_node=2, attr='weight')
    transform_graph_attribute(connected_clusters_graph, 'weight', lambda x:x-2)

    #communities = nx.community.greedy_modularity_communities(cluster_graph, weight='weight')
    #connected_clusters = load_connected_clusters(verbose) # Each exploit cluster is treated like a vertex, a node

    # Build super-graph of connected clusters
    #connected_clusters_graph = nx.from_edgelist(connected_clusters)
    from ..behavioral import _utils as ut
    #ut.segment_explore_exploit(parsed_vanilla[0]['actions'], normalize_by_steps=True)
    all_connected_components = sorted([connected_component for connected_component in nx.connected_components(connected_clusters_graph)], key=len)
    largest_connected_component = all_connected_components[-1]
    largest_cc_subgraph = connected_clusters_graph.subgraph(largest_connected_component)
    
    # Find communities of connected components
    communities = [community for community in nx.community.greedy_modularity_communities(largest_cc_subgraph, resolution=3.5, weight='weight')]
    #communities = [community for community in nx.community.louvain_communities(largest_cc_subgraph, resolution=1, weight='weight')]
    if verbose:
        offset = 0
        print('Saving all communities')
        for ind, community in tqdm(enumerate(communities[offset:])):
            fig = utils.visualization.show_shape_from_size_dict({shape: total_node_weight(G=connected_clusters_graph, attr='weight', node=shape) for shape in community})
            file_name = 'community_{comm}.png'.format(comm=ind + offset)
            path = 'C:\\Users\\Yogevhen\\Desktop\\Project\\CFGpy\\communities'
            utils.visualization.save_plot(fig=fig, file_name=file_name, path=path, subfolder='tests')
    top_communities = communities[:N_TOP_COMMUNITIES]
    dimensions_count_per_shape = count_shapes_in_many_communities(top_communities) # Gallery meaning scores

    all_gallery_shapes = get_all_gallery_shapes(parsed_vanilla)
    core_shapes = {
        shape for shape in dimensions_count_per_shape if is_core_shape(dimensions_count_per_shape[shape])
    }
    if verbose:
        print('Amount of core shapes with more than 2 meaning: {length}'.format(length=len([dimensions_count_per_shape[shape] for shape in core_shapes if np.sum(dimensions_count_per_shape[shape]) > 2])))

    all_non_core_galleries = all_gallery_shapes.difference(core_shapes)

    all_shapes = get_all_shapes()
    all_non_gallery_shapes = all_shapes.difference(all_gallery_shapes)

    gallery_graph = build_gallery_graph(parsed_vanilla, False)
    gallery_graph_2 = build_gallery_graph(parsed_vanilla, True)

    gallery_graph_connected_components = sorted([connected_component for connected_component in nx.connected_components(gallery_graph)], key=len)
    largest_gallery_graph_connected_component = gallery_graph_connected_components[-1]
    gallery_graph_cc_subgraph = gallery_graph.subgraph(largest_gallery_graph_connected_component)

    induced_meaning = build_meaning(all_galleries_connected=largest_gallery_graph_connected_component, gallery_meaning_scores=dimensions_count_per_shape, gallery_graph_connected=gallery_graph_cc_subgraph)

    # Show shapes that appear >5 times

if __name__ == "__main__":
    build_clusters_and_communities()