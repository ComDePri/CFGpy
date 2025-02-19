from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import json
import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
# set interactive backend for matplotlib
if __name__ == '__main__':
    mpl.use('macosx')

ID2COORD = np.load("grid_coords.npy") # should change to the right path after installation
N_ALL_SHAPES = len(ID2COORD) - 1  # subtract 1 because index 0 in ID2COORD is a placeholder, not a shape

SHAPE_COLOR = "#32CD32"  # CSS "limegreen", as used in the game
SHAOE_COLOR_RGB = [50, 205, 50]  # RGB equivalent of SHAPE_COLOR
SHAPE_BG_COLOR = "k"
GALLERY_BG_COLOR = "r"


def get_shape_binary_matrix(shape_id):
    """
    Converts a shape's ID to its binary matrix representation.
    :param shape_id: int
    :return: 2D ndarray with dtype float, all values are binary
    """
    coords = ID2COORD[shape_id]
    binary_mat = np.array([list(np.binary_repr(row, width=10)) for row in coords],
                          dtype=float)
    nrow, ncol = np.max(np.nonzero(binary_mat), axis=1) + 1
    binary_mat = binary_mat[:nrow, :ncol]  # truncate all zeros rows and cols from bottom and right respectively

    return binary_mat


def show_binary_matrix(binary_mat, is_gallery=False, save_filename=None):
    """
    Displays the binary matrix representation of a shape.
    :param binary_mat: a binary matrix representation of a shape.
    :param is_gallery: True iff this a gallery shape. affects background color.
    :param save_filename: a filename to save the image, or None (to avoid saving).
    """
    bg_color = GALLERY_BG_COLOR if is_gallery else SHAPE_BG_COLOR
    nrow, ncol = binary_mat.shape

    plt.figure(facecolor=bg_color)
    plt.gca().tick_params(color=bg_color, labelcolor=bg_color)  # remove tick marks from both axes
    plt.xticks(np.arange(ncol))  # set vertical gridlines
    plt.yticks(np.arange(nrow))  # set horizontal gridlines
    plt.grid(color=bg_color, lw=3)  # create gridlines
    for spine in plt.gca().spines.values():  # remove the frame from around plot
        spine.set_edgecolor(bg_color)

    plt.imshow(binary_mat, extent=(0, ncol, nrow, 0), cmap=ListedColormap([bg_color, SHAPE_COLOR]))
    if save_filename:
        plt.savefig(save_filename)
    plt.show()


def getImage(img, zoom=1):
    return OffsetImage(img, zoom=zoom)


if __name__ == '__main__':
    np.random.seed(1)
    DIRECTIONAL = False
    with open("vanilla.json") as vanilla_fp: # path to the correct post-parsed vanilla.json file
        vanilla = json.load(vanilla_fp)
    all_gallery_transitions = []
    all_gallery_shapes = set()
    for game in tqdm.tqdm(vanilla):
        pre_shape = None
        for action in game["actions"]:
            if action[2] is not None:
                all_gallery_shapes.add(action[0])
                if pre_shape is not None:
                    all_gallery_transitions.append((pre_shape, action[0]))
                pre_shape = action[0]

    all_gallery_shapes = list(all_gallery_shapes)
    shape2idx = {shape: idx for idx, shape in enumerate(all_gallery_shapes)}
    idx2shape = {idx: shape for shape, idx in shape2idx.items()}
    n_shapes = len(all_gallery_shapes)
    transition_matrix = np.zeros((n_shapes, n_shapes))
    for pre_shape, post_shape in tqdm.tqdm(all_gallery_transitions):
        transition_matrix[shape2idx[pre_shape], shape2idx[post_shape]] += 1
        if not DIRECTIONAL:
            transition_matrix[shape2idx[post_shape], shape2idx[pre_shape]] += 1
    shape_counts = transition_matrix.sum(axis=1)
    # normalize so that each row sums to 1
    # transition_matrix = transition_matrix / shape_counts[:, None]

    # plot the explained variance of SVD components to decide on # of embeddings
    from sklearn.decomposition import PCA, NMF
    N_COMPS = 100
    pca = PCA(n_components=N_COMPS)
    pca.fit(transition_matrix)
    #
    plt.plot(np.arange(1,N_COMPS+1),np.cumsum(pca.explained_variance_ratio_), '-ok')
    plt.xlabel("Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()



    # Get the projection of the shapes to the first 5 components
    n_components = 2
    pca_emb = pca.transform(transition_matrix)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=100, verbose=1)
    shape_embeddings = tsne.fit_transform(pca_emb)
    # Use agglo clustering to get dense regions
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(shape_embeddings)
    cluster = AgglomerativeClustering(n_clusters=10, metric='euclidean', linkage='average')
    cluster.fit(shape_embeddings)


    # shape_embeddings = u[:, :n_components]
    # shape_embeddings = pca.transform(transition_matrix)[:,:n_components]
    # plot a scatter of the first 2 components, use the binary matrix representation of the shapes as markers

    fig = plt.figure()
    # set figure background to black
    fig.patch.set_facecolor('black')
    plt.axis('off')
    for idx, shape in idx2shape.items():
        if shape_counts[idx] > 16:
            print(idx)
            REP_FACTOR = 8
            binary_mat = get_shape_binary_matrix(shape)
            binary_mat = np.repeat(binary_mat, REP_FACTOR, axis=0)
            binary_mat = np.repeat(binary_mat, REP_FACTOR, axis=1)
            binary_mat[np.arange(0, binary_mat.shape[0], REP_FACTOR), :] = 0
            binary_mat[:, np.arange(0, binary_mat.shape[1], REP_FACTOR)] = 0
            binary_mat[np.arange(REP_FACTOR-1, binary_mat.shape[0], REP_FACTOR), :] = 0
            binary_mat[:, np.arange(REP_FACTOR - 1, binary_mat.shape[1], REP_FACTOR)] = 0

            colors = np.array([[0,0,0], SHAOE_COLOR_RGB])
            img = colors[binary_mat.astype(int)]
            # get color for the frame based on cluster label
            rect_color = (np.array(plt.get_cmap("tab10")(cluster.labels_[idx])[:3]) * 255).astype(int)
            img[:,0,:] = rect_color
            img[0,:,:] = rect_color
            img[-1,:,:] = rect_color
            img[:,-1,:] = rect_color
            print(shape_embeddings[idx])
            # use img as the marker on the scatter
            ab = AnnotationBbox(getImage(img), (shape_embeddings[idx][0], shape_embeddings[idx][1]), frameon=False)
            plt.gca().add_artist(ab)
            # plt.imshow(img, extent=(shape_embeddings[idx,0], shape_embeddings[idx,0]+0.005, shape_embeddings[idx,1], shape_embeddings[idx,1]+0.01))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()





