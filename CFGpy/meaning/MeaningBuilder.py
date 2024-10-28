import sys
sys.path.append('D:\\ComDePri\\simCFG')
sys.path.append('D:\\ComDePri\\ComDePy')
import os
import simCFG
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools
import networkx as nx
import json

from tqdm import tqdm
from itertools import groupby
from collections import Counter
from .. import behavioral
from ..behavioral import _consts as consts
from .. import utils

CONNECTED_CLUSTERS = r'connected_clusters.json'
MIN_INTERSECT_TO_CONNECT_CLUSTERS = 4
MIN_SHAPES_FOR_CLUSTER = 3
MIN_COUNT_TO_SHOW = 5
N_TOP_COMMUNITIES = 8

# Move somewhere else when I finish with this
def show_community(community, community_number):
    unique_community_shapes = set([shape for community_set in community for shape in community_set])
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
    
    if not os.path.isdir('communities'):
        os.mkdir('communities')
    plt.savefig('communities/community_{community_number}.png'.format(community_number=community_number))
    plt.close()
    return

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

def visualize_common_shapes():
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

def show_communities(communities):
    print()

def build_clusters_and_communities(verbose=True):
    if verbose:
        print('Loading vanilla data')
    with open('fixed_vanilla.json', 'r') as fp:
        parsed_vanilla = json.load(fp)

    # parsed_vanilla = utils.get_vanilla() # Imports the vanilla data (Currently uses a faulty vanilla with the tutorial in it)
    if verbose:
        print('Loading connected clusters data')
    connected_clusters = load_connected_clusters(verbose) # Each exploit cluster is treated like a vertex, a node

    # Build super-graph of connected clusters
    connected_clusters_graph = nx.from_edgelist(connected_clusters)
    all_connected_components = sorted([connected_component for connected_component in nx.connected_components(connected_clusters_graph)], key=len)
    largest_connected_component = all_connected_components[-1]
    largest_cc_subgraph = connected_clusters_graph.subgraph(largest_connected_component)
    
    # Find communities of connected components
    communities = [community for community in nx.community.greedy_modularity_communities(largest_cc_subgraph)]
    if verbose:
        offset = 0
        print('Saving all communities')
        for ind, community in tqdm(enumerate(communities[offset:])):
            show_community(community, ind + offset)
    import ipdb;ipdb.set_trace()
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