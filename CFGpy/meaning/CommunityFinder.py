import lzma
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
    max_community_size = 300
    min_community_size = 2
    min_nodes_in_community = 3
    distance_coeff = 10
    adj_graph = None
    shared_clusters_graph = None
    superclusters_graph = None
    setlike_types = [set, frozenset]
    adj_attr = 'adj_weight'
    shared_clusters_attr = 'shared_clusters_weight'
    supercluster_attr = 'supercluster_weight'

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
            self.weight_attr = self.adj_attr
            self.G = self.adj_graph
            if self.adj_graph is None:
                self.init_adjacency_graph()
        elif self.graph_building_method == 'shared_clusters':
            self.weight_attr = self.shared_clusters_attr
            self.G = self.shared_clusters_graph
            if self.shared_clusters_graph is None:
                self.init_shared_cluster_graph()
        elif self.graph_building_method == 'superclusters':
            self.weight_attr = self.supercluster_attr
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
        
    def init_adjacency_graph(self, set_G=True):
        G = nx.Graph()
        for game in self.vanilla_data:
            gallery_shapes = [action[0] for action in game['actions'] if action[2] is not None]
            for edge in zip(gallery_shapes[:-1], gallery_shapes[1:]):
                if G.has_edge(*edge):
                    weight = G.get_edge_data(*edge).get(self.adj_attr)
                    weight += 1

                    attrs = {
                        edge: {self.adj_attr: weight}
                    }
                    nx.set_edge_attributes(G, attrs)
                else:
                    G.add_edge(*edge, adj_weight=1)

        self.weight_attr = self.adj_attr
        self.adj_graph = G
        self.adj_weights = nx.get_edge_attributes(G, self.adj_attr)
        if set_G:
            self.G = G

    def init_shared_cluster_graph(self, set_G=True):
        clusters = mb.get_all_clusters(self.vanilla_data)
        G = nx.Graph()
        counter = 0
        if self.verbose:
            clusters = tqdm(clusters)

        for cluster in clusters:
            for edge in combinations(cluster, 2):
                counter += 1
                if G.has_edge(*edge):
                    number_of_shared_clusters = G.get_edge_data(*edge).get(self.shared_clusters_attr)
                    number_of_shared_clusters += 1
                    attrs = {
                        edge: {self.shared_clusters_attr: number_of_shared_clusters}
                    }
                    nx.set_edge_attributes(G, attrs)
                else:
                    G.add_edge(*edge, shared_clusters=1)

        self.weight_attr = self.shared_clusters_attr
        self.shared_clusters_graph = G
        if set_G:
            self.G = G

    def init_supercluster_graph(self, graph_loading_path=CLUSTER_GRAPH, set_G=True):
        '''
        Builds a graph from exploit clusters/bouts, an edge exists between clusters that share shapes
        The weight is the amount of shared shapes
        '''
        try:
            with open(graph_loading_path, 'r') as f:
                G = json_graph.node_link_graph(json.load(f))
                self.weight_attr = self.supercluster_attr
                self.superclusters_graph = G
                if set_G:
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

            self.weight_attr = self.supercluster_attr
            self.superclusters_graph = G
            if set_G:
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

        self.unflattened_communities = False
        if self.graph_building_method == 'superclusters':
            self.unflattened_communities = True

        self.communities = communities
        self.prune_communities((min_comm_size, max_comm_size))
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
            if len(pruned_community_size_dict) >= self.min_nodes_in_community:
                pruned_community_size_dicts.append(pruned_community_size_dict)

        sorted_indices = np.argsort([len(sd) for sd in pruned_community_size_dicts])[::-1]
        self.community_size_dicts = [pruned_community_size_dicts[i] for i in sorted_indices]
        self.communities = [self.communities[i] for i in sorted_indices]

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

    def prune_communities(self, community_size_range):
        self.communities = [community for community in self.communities if len(community) in range(*community_size_range)]

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

        score_matrix = np.zeros((len(self.communities), len(self.communities)))
        for i, community in enumerate(self.communities):
            score_vector = self.get_score_similarity_vector(community, score_func)
            score_matrix[i] = score_vector
        
        merged_communities = []
        merge_matrix = np.triu(score_matrix > threshold)
        for i, row in enumerate(merge_matrix):
            merged_community = self.get_merged_communities([self.communities[j] for j in np.nonzero(row)[0]])
            merged_communities.append(merged_community)

        self.communities = merged_communities
        self.update_community_size_dicts(pruning_number)

    @staticmethod
    def get_merged_communities(communities):
        '''
            Merges multiple communities into one community
        '''
        merged_community = set()
        for community in communities:
            merged_community.update(set(community))

        return merged_community

    @staticmethod
    def jaccard_score(a, b):
        if type(a) in CommunityFinder.setlike_types and type(b) in CommunityFinder.setlike_types:
            return len(a.intersection(b)) / len(a.union(b))

        return len([i for i in a if i in b]) / len(set(a + b))

    @staticmethod
    def overlap_score(a, b):
        return len([i for i in a if i in b]) / min(len(a), len(b))
    
    def compute_distances_from_graph(self, G, weights):
        self.distances = {}
        shortest_paths = nx.all_pairs_shortest_path(G, cutoff=10)
        for node_and_paths in tqdm(shortest_paths, total=G.number_of_nodes()):
            node = node_and_paths[0]
            paths = node_and_paths[1]
            for other_node, path in paths.items():
                self.distances[(node, other_node)] = self.distances.get((other_node, node))
                if self.distances[(node, other_node)] is None:
                    self.distances[(node, other_node)] = self.get_path_distance(path, weights)
        
    def save_distance_graph(self, path):
        distances_list = []
        for key, value in self.distances.items():
            if value == np.inf:
                value = None

            distances_list.append([list(key), value])

        with lzma.open(path + '.xz', 'w') as f:
            f.write(json.dumps(distances_list).encode())

    def load_distance_graph(self, path):
        with lzma.open(path + '.xz', 'r') as f:
            distances_list = json.loads(f.read())
            self.distances = {}
            for key, value in distances_list:
                if value is None:
                    value = np.inf

                self.distances[tuple(key)] = value

    def silhouette_score(self, cluster):
        other_clusters = [c for c in self.communities if c != cluster]
        if self.unflattened_communities:
            cluster = [node for sub_cluster in cluster for node in sub_cluster]
            other_clusters = [[node for sub_cluster in other_cluster for node in sub_cluster] for other_cluster in other_clusters]

        silhouette_scores = []
        for node in cluster:
            silhouette_score = self.silhouette_score_data_point(node, cluster, other_clusters)
            silhouette_scores.append(silhouette_score)
        
        return np.mean(silhouette_scores)
        
    def silhouette_score_data_point(self, point, own_cluster, other_clusters):
        if len(own_cluster) == 1:
            return 0

        a = self.silhouette_score_a(point, own_cluster)
        b = self.silhouette_score_b(point, other_clusters)
                                               
        return (b - a) / max(a, b)
    
    def silhouette_score_a(self, point, own_cluster):
        if len(own_cluster) == 1:
            raise ValueError('Cannot calculate silhouette score for single node cluster')
        
        return 1/(len(own_cluster) - 1) * np.sum([self.distances[(point, other_point)] for other_point in own_cluster if other_point != point])

    def silhouette_score_b(self, point, other_clusters):
        mean_distances = []

        for other_cluster in other_clusters:
            mean_distance = 1/len(other_cluster) * np.sum([self.distances[(point, other_point)] for other_point in other_cluster])
            mean_distances.append(mean_distance)
            
        return np.min(mean_distances)
    
    def compute_distances_between_nodes(self, G, weight_attr, node, other_node):
        if node == other_node:
            return 0

        edge = (node, other_node)
        weights = nx.get_edge_attributes(G, weight_attr)
        edge_weight = weights.get(edge, weights.get(edge[::-1]))
        if edge_weight is not None:
            return 1/edge_weight

        for path in nx.shortest_simple_paths(G, node, other_node): 
            # Note: Shortest simple paths can get weights, I can switch it but then I need
            # to change the multiplication by the distance coefficient since shortest paths no longer take length of path into account
            if self.get_path_distance(G, path, weight_attr) is not None:
                return self.get_path_distance(G, path, weight_attr)
        
        return np.inf

    def get_path_distance(self, path, weights):
        distance = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            weight = weights.get(edge, weights.get(edge[::-1]))
            if weight is None:
                return None

            distance += (self.distance_coeff**i)/weight

        return distance

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