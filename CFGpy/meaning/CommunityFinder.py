import json
import networkx as nx
import numpy as np
import itertools

from collections import Counter
from cdlib import algorithms
from cdlib import classes as cdlib_classes
from itertools import groupby, combinations
from tqdm import tqdm
from networkx.readwrite import json_graph

from .. import utils
from . import MeaningBuilder as mb

CLUSTER_GRAPH = r'cluster_graph.json'
DEFAULT_VANILLA_DATA_PATH = r'clean_vanilla_converted.json'

DEFAULT_COMMUNITY_PRUNING_NUMBER = 0
DEFAULT_PRUNING_NUMBER = 0

class CommunityFinder():
    max_community_size = 80
    min_community_size = 2
    adj_graph = None
    shared_clusters_graph = None
    superclusters_graph = None
    setlike_types = [set, frozenset]

    def __init__(self, graph_building_method, vanilla_data_path=DEFAULT_VANILLA_DATA_PATH, verbose=False):
        self.graph_building_method = graph_building_method
        self.load_vanilla_data(vanilla_data_path)
        self.verbose = verbose

        self.graph = self.init_graph()
    
    def load_vanilla_data(self, vanilla_data_path):
        '''
            Loads vanilla data from a json file. 
            We expect the shapes to be written as int ids.
        '''
        with open(vanilla_data_path, 'r') as fp:
            self.vanilla_data = json.load(fp)
            for game in self.vanilla_data:
                game['actions'] = np.array([action for action in game['actions']])

    def set_current_graph(self, desired_graph_building_method):
        self.graph_building_method = desired_graph_building_method
        if self.graph_building_method == 'adjacency':
            self.weight_attr = 'adj_weight'
            self.G = self.adj_graph
            if self.adj_graph is None:
                self.init_adjacency_graph()
        elif self.graph_building_method == 'shared_clusters':
            self.weight_attr = 'shared_clusters'
            self.G = self.shared_clusters_graph
            if self.shared_clusters_graph is None:
                self.init_shared_cluster_graph()
        elif self.graph_building_method == 'superclusters':
            self.weight_attr = 'supercluster_weight'
            self.G = self.superclusters_graph
            if self.superclusters_graph is None:
                self.init_supercluster_graph()
        else:
            raise NotImplementedError('Graph building method not supported')

    def init_graph(self):
        if self.graph_building_method == 'adjacency':
            self.init_adjacency_graph()
        elif self.graph_building_method == 'shared_clusters':
            self.init_shared_cluster_graph()
        elif self.graph_building_method == 'superclusters':
            self.init_supercluster_graph()
        else:
            raise NotImplementedError('Graph building method not supported')
        
    def init_adjacency_graph(self):
        G = nx.Graph()
        for game in self.vanilla_data:
            gallery_shapes = [action[0] for action in game['actions'] if action[2] is not None]
            for edge in zip(gallery_shapes[:-1], gallery_shapes[1:]):
                if G.has_edge(*edge):
                    weight = G.get_edge_data(*edge).get('adj_weight')
                    weight += 1

                    attrs = {
                        edge: {'adj_weight': weight}
                    }
                    nx.set_edge_attributes(G, attrs)
                else:
                    G.add_edge(*edge, adj_weight=1)

        self.weight_attr = 'adj_weight'
        self.adj_graph = G
        self.G = G

    def init_shared_cluster_graph(self):
        clusters = mb.get_all_clusters(self.vanilla_data)
        G = nx.Graph()
        counter = 0
        if self.verbose:
            clusters = tqdm(clusters)

        for cluster in clusters:
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

        self.weight_attr = 'shared_clusters'
        self.shared_clusters_graph = G
        self.G = G

    def init_supercluster_graph(self, graph_loading_path=CLUSTER_GRAPH):
        '''
        Builds a graph from exploit clusters/bouts, an edge exists between clusters that share shapes
        The weight is the amount of shared shapes
        '''
        try:
            with open(graph_loading_path, 'r') as f:
                G = json_graph.node_link_graph(json.load(f))
                self.weight_attr = 'supercluster_weight'
                self.superclusters_graph = G
                self.G = G

                return

        except FileNotFoundError:
            if self.verbose:
                print('Clusters graph not found, recalculating')
            for game in self.vanilla_data:
                game['actions'] = np.array([action for action in game['actions']])

            clusters = mb.get_all_clusters(self.vanilla_data)
            G = nx.Graph()
            all_pairs = itertools.product(clusters, clusters)
            if self.verbose:
                all_pairs = tqdm(all_pairs)
            for pair in all_pairs:
                supercluster_weight = np.intersect1d(pair[0], pair[1]).shape[0]
                if pair[0] != pair[1] and supercluster_weight > 0:
                    G.add_edge(tuple(pair[0]), tuple(pair[1]), supercluster_weight=supercluster_weight)

            self.weight_attr = 'supercluster_weight'
            self.superclusters_graph = G
            self.G = G
            if self.verbose:
                print('Saving connected clusters calculation')
            with open(graph_loading_path, 'w') as fp:
                json.dump(json_graph.node_link.node_link_data(G), fp)

    def prune_edges(self, edge_pruning_number):
        attr_for_edge = nx.get_edge_attributes(self.G, self.weight_attr)
        edges_for_pruning = []
        for edge in self.G.edges:
            if attr_for_edge[edge] < edge_pruning_number:
                edges_for_pruning.append(edge)

        self.G.remove_edges_from(edges_for_pruning)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))

    def transform_weights(self, func):
        self.transform_graph_attribute(self.G, self.weight_attr, func)

    def find_communities(self, method, filter_largest_cc=True):
        graph_for_community_search = self.G
        if filter_largest_cc:
            all_connected_components = sorted([connected_component for connected_component in nx.connected_components(self.G)], key=len)
            largest_connected_component = all_connected_components[-1]
            graph_for_community_search = self.G.subgraph(largest_connected_component)

        return method(graph_for_community_search)

    def get_communities(self, method, min_comm_size=min_community_size, max_comm_size=max_community_size, pruning_number=DEFAULT_COMMUNITY_PRUNING_NUMBER, filter_largest_cc=True):
        communities = self.find_communities(method=method, filter_largest_cc=filter_largest_cc)
        if type(communities) is cdlib_classes.node_clustering.NodeClustering:
            communities = communities.communities

        self.communities = [comm for comm in communities if len(comm) in range(min_comm_size, max_comm_size)]
        self.update_community_size_dicts(pruning_number)

        return self.communities

    def update_community_size_dicts(self, pruning_number):
        community_size_dicts = self.count_community_sizes(self.communities)
        pruned_community_size_dicts = []
        for community_size_dict in community_size_dicts:
            pruned_community_size_dict = {
                node: community_size_dict[node] for node in community_size_dict
                if community_size_dict[node] > pruning_number
            }
            if len(pruned_community_size_dict) >= self.min_community_size:
                pruned_community_size_dicts.append(pruned_community_size_dict)

        sorted_indices = np.argsort([len(sd) for sd in pruned_community_size_dicts])[::-1]
        pruned_community_size_dicts = [pruned_community_size_dicts[i] for i in sorted_indices]
        self.communities = [self.communities[i] for i in sorted_indices]
        self.community_size_dicts = pruned_community_size_dicts
    
    def prune_communities(self, community_size_range):
        self.communities = [community for community in self.communities if len(community) in range(*community_size_range)]

    def plot_and_save_community_graphs(self, path, subfolder_name):
        if getattr(self, 'community_size_dicts', None) is None:
            raise ValueError('No communities found yet, run get_communities first')

        for ind, community_size_dict in enumerate(self.community_size_dicts):
            fig = utils.visualization.show_shape_from_size_dict(shapes_dict=community_size_dict)
            file_name = 'community_{comm}.png'.format(comm=ind)
            utils.visualization.save_plot(fig=fig, file_name=file_name, path=path, subfolder=subfolder_name)

    def count_community_sizes(self, communities):
        if self.graph_building_method == 'superclusters':
            return self._count_sizes_from_superclusters(communities)
        else:
            return self._count_sizes_by_weights(communities)

    def _count_sizes_by_weights(self, communities):
        '''
            Takes a list of communities, where each community is a list of nodes
            Returns a list of dictionaries counting the total weight of each node in the community
        '''
        edge_weights = nx.get_edge_attributes(self.G, self.weight_attr)
        
        community_size_dicts = []
        for community in communities:
            if len(community) == 1: # Skip single node communities
                continue

            size_dict = {}
            for node in community:
                total_weight = np.sum([edge_weights.get(edge, edge_weights.get(edge[::-1])) for edge in self.G.edges(node) if edge[0] in community and edge[1] in community])
                size_dict[node] = total_weight

            community_size_dicts.append(size_dict)

        return community_size_dicts

    def _count_sizes_from_superclusters(self, communities):
        '''
            Takes a list of communities, where each community is a list of clusters
            Flattens the clusters and counts the amount of each node in total.
            So if a node is in 3 clusters, it will have a count of 3.
        '''
        community_size_dict = []
        for community in communities:
            flattened_list = [node for cluster in community for node in cluster]
            shapes_with_count = Counter(flattened_list)
            community_size_dict.append(shapes_with_count)

        return community_size_dict

    def prune_communities(self, communities, community_size_range):
        '''
            Prunes communities based on the amount of nodes in the community
        '''
        self.communities = [community for community in communities if len(community) in range(*community_size_range)]

    def merge_and_add_communities(self, community_a, community_b):
        '''
            Merges two communities by adding all nodes from community_b to community_a
        '''
        if community_a in self.communities:
            self.communities.remove(community_a)

        if community_b in self.communities:
            self.communities.remove(community_b)

        if type(community_a) != type(community_b): 
            error_message = 'Community types mismatch: a is {a}, b is {b}'.format(a=type(community_a), b=type(community_b))
            raise ValueError(error_message)
        elif type(community_a) in self.setlike_types and type(community_b) in self.setlike_types:
            merged_community = community_a.union(community_b)
        elif type(community_a) is list and type(community_b) is list:
            merged_community = community_a + community_b
        else:
            raise ValueError('Community type {c_type} not supported'.format(c_type=type(community_a)))

        self.communities.append(merged_community)

    def get_score_similarity_vector(self, community, score_func):
        '''
            Returns a vector of score similarities between a community and all other communities
        '''
        communities = self.communities
        if self.graph_building_method == 'superclusters':
            community = set([node for cluster in community for node in cluster])
            communities = [
                set([node for cluster in other_community for node in cluster]) for other_community in communities
            ]
        score_vector = np.zeros(len(communities))
        for i, other_community in enumerate(communities):
            if community == other_community:
                score_vector[i] = 1
                continue

            score_vector[i] = score_func(community, other_community)

        return score_vector

    def merge_communities_based_on_score(self, threshold, score_func, pruning_number=DEFAULT_PRUNING_NUMBER):
        '''
            Merges communities based on score similarity given a threshold.
        '''
        if len(self.communities) == 1:
            if self.verbose:
                print('There is only one community')
            return

        merged_communities = []
        score_matrix = np.zeros((len(self.communities), len(self.communities)))
        # Populate the score matrix
        for i, community in enumerate(self.communities):
            score_vector = self.get_score_similarity_vector(community, score_func)
            score_matrix[i] = score_vector
        
        merge_matrix = score_matrix > threshold
        for i, row in enumerate(merge_matrix):
            if not merge_all and np.sum(row) == 1:
                continue

            for j, merge in enumerate(row):
                if merge:
                    merged_communities.append(self.communities[j])
                    self.merge_and_add_communities(self.communities[i], self.communities[j])
        for community in sorted(self.communities, key=lambda x: len(x)):
            score_vector = self.get_score_similarity_vector(community, score_func)
            most_similar_community = np.argsort(score_vector)[-2] # Get the most similar community that is not itself
            if score_vector[most_similar_community] < threshold:
                continue
            
            to_merge = [self.communities[most_similar_community]]
            if merge_all:
                for i in np.argsort(score_vector)[::-1][2:]: # Skip the most similar community
                    if score_vector[i] < threshold:
                        break

                    to_merge.append(self.communities[i])

            for community_to_merge in to_merge:
                self.merge_and_add_communities(community, community_to_merge)

            self.merge_communities_based_on_score(threshold, score_func, merge_all)
            break

        self.update_community_size_dicts(pruning_number)

    @staticmethod
    def jaccard_score(a, b):
        if type(a) in CommunityFinder.setlike_types and type(b) in CommunityFinder.setlike_types:
            return len(a.intersection(b)) / len(a.union(b))

        return len([i for i in a if i in b]) / len(set(a + b))

    @staticmethod
    def overlap_score(a, b):
        return len([i for i in a if i in b]) / min(len(a), len(b))

    @staticmethod
    def get_affine_trans(i):
        return lambda x: x - i + 1
    
    @staticmethod
    def transform_graph_attribute(G, attr, func):
        edge_attribute = nx.get_edge_attributes(G, attr)
        new_attributes = {}
        for edge in G.edges:
            new_attributes[edge] = {attr: func(edge_attribute[edge])}

        nx.set_edge_attributes(G, new_attributes)

# CommunityFinder workflow: 
    # load a graph and Add weights and attributes
    # Prune if needed
    # Find communities using a community detection algorithm
    # Prune/Merge communities as needed
    # Visualize and save the communities