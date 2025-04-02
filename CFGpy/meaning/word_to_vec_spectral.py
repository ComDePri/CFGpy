import os

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import json
import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
#from meaning_embedding import get_shape_binary_matrix, show_binary_matrix, getImage
from sklearn.decomposition import PCA, NMF
import seaborn as sns

from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import SpectralEmbedding

import scipy.special

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from ..utils import ID2COORD, get_shape_binary_matrix
from . import embedder

VANILLA_PATH = r'C:\Users\User\Desktop\Projects\Meaning_Landscape\code\CFGpy\CFGpy\datafiles\clean_vanilla_converted.json'

class WordToVecSpectral(embedder.Embedder):
    N_ALL_SHAPES = len(ID2COORD) - 1  # subtract 1 because index 0 in ID2COORD is a placeholder, not a shape

    SHAPE_COLOR = "#32CD32"  # CSS "limegreen", as used in the game
    SHAPE_COLOR_RGB = [50, 205, 50]  # RGB equivalent of SHAPE_COLOR
    SHAPE_BG_COLOR = "k"
    GALLERY_BG_COLOR = "r"

    def load_data(self):
        self.ID2COORD = ID2COORD
        self.n_all_shapes = len(ID2COORD) - 1

        with open(VANILLA_PATH) as vanilla_fp:  # path to post-parsed vanilla.json file
            self.vanilla = json.load(vanilla_fp)

        self.all_trajectories = [
            [str(action[0]) for action in v_game["actions"] if action[2] is not None]
            for v_game in self.vanilla
        ]

    def embed_word_to_vec(self, epochs=1000, window=2):
        e_logger = EpochLogger() # TODO: change so this gets the data of the model here
        self.model = Word2Vec(sentences=self.all_trajectories, vector_size=20, window=window, min_count=1, workers=4,
                                callbacks=[e_logger], epochs=epochs, hs=0, negative=100, compute_loss=True, alpha=1e-4, min_alpha=1e-10, shrink_windows=False)
        # get all embeddings
        self.all_shapes = set([shape for traj in self.all_trajectories for shape in traj])
        self.shapes_in_model = [shape for shape in self.all_shapes if shape in self.model.wv]
        self.all_shapes_embeddings = {shape: self.model.wv[shape] for shape in self.shapes_in_model}

    def embed_spectral(self, pruning_number=4):
        counter = Counter([shape for traj in self.all_trajectories for shape in traj])
        self.shapes2cluster = [s for s in counter if counter[s] > pruning_number]
        shape_embeddings2cluster = np.array([self.all_shapes_embeddings[shape] for shape in self.shapes2cluster])

        self.shape_embeddings2cluster = SpectralEmbedding(n_components=10, n_neighbors=8).fit_transform(shape_embeddings2cluster)
    
    def find_clusters_dbscan(self, eps=0.036, min_samples=6):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        dbscan.fit(self.shape_embeddings2cluster)

        self.dbscan = dbscan

    def plot_dbscan_results(self):
        epsilons = np.linspace(0.001,0.5,300)
        min_samples_vals = np.arange(3, 10)
        fig, (ax_nclust, ax_max_size) = plt.subplots(1,2, figsize=(14, 6))
        for min_samples in min_samples_vals:
            n_clusters = []
            max_size = []
            for eps in tqdm.tqdm(epsilons):
                self.find_clusters_dbscan(eps, min_samples)
                n_clusters.append(np.max(self.dbscan.labels_))
                max_size.append(pd.Series(self.dbscan.labels_[self.dbscan.labels_ > -1]).value_counts().max())

            plt.sca(ax_nclust)
            plt.plot(epsilons, n_clusters, '-o', label=f"Min Samples: {min_samples}")
            plt.xlabel("Epsilon")
            plt.ylabel("Number of Clusters")
            plt.sca(ax_max_size)
            plt.plot(epsilons, max_size, '-o', label=f"Min Samples: {min_samples}")
            plt.xlabel("Epsilon")
            plt.ylabel("Max Cluster Size")
        plt.sca(ax_nclust)
        plt.legend()
        plt.sca(ax_max_size)
        plt.legend()
        plt.show()

    def find_communities(self):
        self.load_data()
        self.embed_word_to_vec()
        self.embed_spectral()
        self.find_clusters_dbscan(0.036, 6)
        self.plot_clusters()
    
    def plot_clusters(self):
        for cluster in range(np.max(self.dbscan.labels_)):
            cluster_shapes = np.array(self.shapes2cluster)[self.dbscan.labels_ == cluster]
            # print(f"Cluster {cluster}")
            # print(cluster_shapes)
            n = len(cluster_shapes)
            nh = int(np.sqrt(n))
            nv = int(np.ceil(n / nh))
            fig, axes = plt.subplots(nh, nv, figsize=(2 * nh, 2 * nv))
            axes_flat = axes.flatten()
            for shape, ax in zip(cluster_shapes, axes_flat):
                ax.imshow(self.get_shape_image(int(shape)))
                ax.axis("off")
            plt.suptitle(f"Cluster {cluster}")
            plt.savefig(r"communities\word2vec\cluster_{}.png".format(cluster))
            plt.close(fig)

    def get_shape_image(self, shape):
        REP_FACTOR = 8
        binary_mat = get_shape_binary_matrix(shape)
        binary_mat = np.repeat(binary_mat, REP_FACTOR, axis=0)
        binary_mat = np.repeat(binary_mat, REP_FACTOR, axis=1)
        binary_mat[np.arange(0, binary_mat.shape[0], REP_FACTOR), :] = 0
        binary_mat[:, np.arange(0, binary_mat.shape[1], REP_FACTOR)] = 0
        binary_mat[np.arange(REP_FACTOR - 1, binary_mat.shape[0], REP_FACTOR), :] = 0
        binary_mat[:, np.arange(REP_FACTOR - 1, binary_mat.shape[1], REP_FACTOR)] = 0

        colors = np.array([[0, 0, 0], self.SHAPE_COLOR_RGB])
        img = colors[binary_mat.astype(int)]
        # get color for the frame based on cluster label
        return img
    
    def get_landscape_top_shapes(self, meaning, n):
        clusters_top_shapes = []
        for i in range(8):
            top_n_shapes = np.argsort(meaning[:, i])[::-1][:n]

            nh = int(np.sqrt(n))
            nv = int(np.ceil(n / nh))
            fig, axes = plt.subplots(nh, nv, figsize=(2 * nh, 2 * nv))
            axes_flat = axes.flatten()
            for shape, ax in zip(top_n_shapes, axes_flat):
                ax.imshow(self.get_shape_image(int(shape)))
                ax.axis("off")
            plt.suptitle(f"Cluster {i}")
            plt.savefig(r"communities\original_meaning\cluster_{}.png".format(i))
            plt.close(fig)

class EpochLogger(CallbackAny2Vec):
    def __init__(self, keep_score=False, score_data=None):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.losses = []
        self.keep_score = keep_score
        self.score_data = score_data
        self.scores = None
        self.sentences = []

        self.all_windows = []

    def on_epoch_begin(self, model):
        if self.epoch % 10 == 0:
            self.calculate_loss(model)
            print('Loss before epoch {}: {}'.format(self.epoch, self.loss_now))

        self.losses.append(self.loss_now)
        if self.keep_score:
            self.scores = model.score(self.score_data)
        self.epoch += 1

    def load_sentences(self, model):
        with open(VANILLA_PATH) as vanilla_fp:  # path to post-parsed vanilla.json file
            self.vanilla = json.load(vanilla_fp)

        self.all_trajectories = [
            [str(action[0]) for action in v_game["actions"] if action[2] is not None]
            for v_game in self.vanilla
        ]

        for trajectory in self.all_trajectories:
            self.sentences += [
                trajectory[i - model.window - 1:i + model.window]
                for i in range(model.window + 1, len(trajectory) - model.window + 1)
            ]

    def calculate_loss(self, model):
        """Calculate the loss for the current epoch.

        Returns
        -------
        float
            Loss for the current epoch.

        """
        # Calculate the loss for each word and context in gensim
        # Word2Vec model
        loss = 0

        if self.sentences == []:
            self.load_sentences(model)
            self.compute_sampling_distribution(model)

        for sentence in self.sentences:
            loss += self.calculate_loss_for_word(sentence[0:model.window] + sentence[model.window + 1:], sentence[model.window], model)

        self.loss_now = loss / len(self.sentences) # Check that this is correct

    def calculate_loss_for_word(self, context_words_list, word, model):
        """Get the Negative log likelihood for a given word and context.

        Parameters
        ----------
        context_words_list : list of (str and/or int)
            List of context words, which may be words themselves (str)
            or their index in `self.wv.vectors` (int).
        word : word (str)
            Word to calculate the negative log likelihood for.

        Returns
        -------
        float
            Negative log likelihood for the given word and context.

        """
        if word in model.wv:
            word_index = model.wv.get_index(word)
        else:
            return 0

        word2_indices = [model.wv.get_index(w) for w in context_words_list if w in model.wv]
        if len(word2_indices) != len(context_words_list):
            return 0

        wc = model.wv.vectors[word2_indices]
        l1 = np.sum(wc, axis=0)
        if word2_indices and model.cbow_mean:
            l1 /= len(word2_indices)

        log_p_unnormed = np.dot(l1, model.wv.vectors.T)
        log_z = scipy.special.logsumexp(log_p_unnormed)

        ns_term_optimized = np.inner(self.ns_dist, scipy.special.log_expit(-log_p_unnormed))
        ns_J = - (scipy.special.log_expit(log_p_unnormed[word_index]) + ns_term_optimized)

        return ns_J

        # return -log_p_unnormed[word_index] + log_z
    
    def compute_sampling_distribution(self, model):
        """Calculate the negative sampling distribution for the given model."""

        # Build a distribution function for the sentences
        unnormalized_dist = np.array([model.wv.get_vecattr(i, 'count')**model.ns_exponent for i in range(len(model.wv))])
        self.ns_dist = unnormalized_dist / np.sum(unnormalized_dist)