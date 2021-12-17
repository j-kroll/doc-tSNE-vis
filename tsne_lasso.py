from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import mplcursors
import nltk
from nltk import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import textwrap

class SelectFromCollection:
    def __init__(self, ax, collection, is_first_cluster):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.is_first_cluster = is_first_cluster

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.fc = collection.get_facecolors()
        self.colors = np.array([[0.6, 0.6, 0, 1], [1, 0.4, 1, 1], [0, 0.8, 0.8, 1]])

        self.fc[:] = self.colors[0]
        self.collection.set_facecolors(self.fc)

        self.lasso = None
        self.ind = []

    def activate(self, ax):
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        if self.is_first_cluster:
            self.fc[:] = self.colors[0]
            self.fc[self.ind] = self.colors[1]
        else:
            for i, pt in enumerate(self.fc):
                if (pt==self.colors[2]).all():
                    self.fc[i] = self.colors[0]
            self.fc[self.ind] = self.colors[2]
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self, next_selector=None):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()
        if next_selector:
            next_selector.activate(self.ax)
            next_selector.fc = self.fc

def main():

    lassoed_sentences_A = []
    lassoed_dfs_A = []
    lassoed_df_A = []

    lassoed_sentences_B = []
    lassoed_dfs_B = []
    lassoed_df_B = []

    def get_adj_matrix(lassoed_df, all_df, color, cluster_ax, line_ax):
        num_data = len(lassoed_df)
        adj_matrix = np.empty(shape=(num_data,num_data))
        max_dist = -1
        max_uv = None
        min_dist = 1000
        min_uv = None
        for u in range(num_data):
            for v in range(num_data):
                dist = distance.euclidean(lassoed_df.iloc[u].high_dim, lassoed_df.iloc[v].high_dim)
                adj_matrix[u][v] = dist
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_uv = (u,v)
                if dist > max_dist:
                    max_dist = dist
                    max_uv = (u,v)
        print("Max dist {}: {}".format(max_uv, max_dist))
        print(lassoed_df.iloc[max_uv[0]].sent)
        print(lassoed_df.iloc[max_uv[1]].sent)
        print("Min dist {}: {}".format(min_uv, min_dist))
        print(lassoed_df.iloc[min_uv[0]].sent)
        print(lassoed_df.iloc[min_uv[1]].sent)
        color_arr = np.tile(color,(lassoed_df.shape[0],1,1))
        cluster_sc = cluster_ax.scatter(*zip(*lassoed_df["tsne"]), c=color_arr, s=8)
        cluster_ax.scatter(lassoed_df.iloc[min_uv[0]]["tsne"][0], lassoed_df.iloc[min_uv[0]]["tsne"][1], edgecolors=np.array([.1,.5,.15,1]), facecolors="None", s=40)
        cluster_ax.scatter(lassoed_df.iloc[min_uv[1]]["tsne"][0], lassoed_df.iloc[min_uv[1]]["tsne"][1], edgecolors=np.array([.1,.5,.15,1]), facecolors="None", s=40)
        cluster_ax.scatter(lassoed_df.iloc[max_uv[0]]["tsne"][0], lassoed_df.iloc[max_uv[0]]["tsne"][1], edgecolors=np.array([1,0,0,1]), facecolors="None", s=80)
        cluster_ax.scatter(lassoed_df.iloc[max_uv[1]]["tsne"][0], lassoed_df.iloc[max_uv[1]]["tsne"][1], edgecolors=np.array([1,0,0,1]), facecolors="None", s=80)
        centroid_high_dim = np.mean(lassoed_df["high_dim"], axis=0)
        height = 0
        centroid_dists = []
        for i in range(len(lassoed_df)):
            p = lassoed_df.iloc[i]
            c_pt = np.tile(color,(1,1,1))
            dist_from_centroid = [distance.euclidean(p.high_dim, centroid_high_dim), height]
            centroid_dists.append(dist_from_centroid)
        lassoed_df["dist_from_centroid"] = centroid_dists
        line_ax.scatter(*zip(*lassoed_df["dist_from_centroid"]), c=color_arr, s=8)
        cursor_sc = mplcursors.cursor(cluster_ax, hover=2)
        cursor_sc.connect("add", lambda sel: sel.annotation.set_text(
            textwrap.fill(
                lassoed_df["sent"][sel.index], 20
            )
        ))
        cursor_lineplot = mplcursors.cursor(line_ax, hover=2)
        cursor_lineplot.connect("add", lambda sel: sel.annotation.set_text(
            textwrap.fill(
                lassoed_df["sent"][sel.index], 20
            )
        ))
        plt.show()
        return adj_matrix

    def accept(event):

        if event.key == "a":
            print("Selected A cluster:")
            print(selector1.xys[selector1.ind])
            selector1.disconnect(selector2)
            fig.suptitle("Select B cluster...")
            fig.canvas.draw()
            for tsne_coords in selector1.xys[selector1.ind]:
                item_df = df.loc[df.tsne.apply(lambda tc: (tc==tsne_coords).all())]
                lassoed_sentence = item_df.iloc[0].sent
                lassoed_sentences_A.append(lassoed_sentence)
                lassoed_dfs_A.append(item_df)
                print(item_df.sent)
            lassoed_df_A = pd.concat(lassoed_dfs_A, ignore_index=True)
            print("\n".join(lassoed_sentences_A))
            print(get_adj_matrix(lassoed_df_A, df, selector1.colors[1], ax2, ax4))

        elif event.key == "b":
            print("Selected B cluster:")
            print(selector2.xys[selector2.ind])
            selector2.disconnect()
            fig.suptitle("Selected 2 clusters!")
            fig.canvas.draw()
            for tsne_coords in selector2.xys[selector2.ind]:
                item_df = df.loc[df.tsne.apply(lambda tc: (tc==tsne_coords).all())]
                lassoed_sentence = item_df.iloc[0].sent
                lassoed_sentences_B.append(lassoed_sentence)
                lassoed_dfs_B.append(item_df)
            lassoed_df_B = pd.concat(lassoed_dfs_B, ignore_index=True)
            print("\n".join(lassoed_sentences_B))
            print(get_adj_matrix(lassoed_df_B, df, selector2.colors[2], ax3, ax5))

        elif event.key == "f":
            print("Finished.")
            fig.suptitle("Finished cluster selection (X to exit)")
            fig.canvas.draw()

        elif event.key == "x":
            plt.close("all")

    nltk.download("punkt")

    f = open("sherlock.txt", "r")
    # f = open("candide.txt", "r")
    # f = open("alice.txt", "r")
    # f = open("jane_eyre.txt", "r")
    # f = open("grimm.txt", "r")
    contents = f.read().replace("\n", " ")
    sent_text = sent_tokenize(contents)
    print("Read and tokenized sentences...")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sent_text)
    print("Got high-dimensional sentence embeddings...")
    
    tsne_emb = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(sentence_embeddings)
    print("Calculated t-SNE embeddings...")
    print("Number of sentences:", len(sent_text))
    print("Sentence embeddings shape:", sentence_embeddings.shape)
    print("t-SNE embeddings shape:", tsne_emb.shape)

    data = []
    for index, s in enumerate(sent_text):
        data.append((tsne_emb[index], sentence_embeddings[index], sent_text[index], [0.6, 0.6, 0, 1]))
    print(len(data), "sentence datapoints\n")

    df = pd.DataFrame(data, columns=["tsne", "high_dim", "sent", "color"])
    print(df.info())

    fig = plt.figure()
    grid = plt.GridSpec(5, 4)
    ax1 = plt.subplot(grid[:, 0:2])
    ax2 = plt.subplot(grid[0:3, 2])
    ax3 = plt.subplot(grid[0:3, 3])
    ax4 = plt.subplot(grid[3, 2:])
    ax5 = plt.subplot(grid[4, 2:], sharex=ax4)

    fig.set_dpi(150)
    fig.canvas.mpl_connect("key_press_event", accept)

    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)

    sc = ax1.scatter(*zip(*df["tsne"]), c=df.color, s=8)

    selector1 = SelectFromCollection(ax1, sc, True)
    selector2 = SelectFromCollection(ax1, sc, False)

    selector1.activate(ax1)

    cursor = mplcursors.cursor(ax1, hover=2)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        textwrap.fill(
            df["sent"][sel.index], 20
        )
    ))

    fig.suptitle("Select A cluster...")

    plt.savefig("tsne_chapters_lasso.png")
    plt.show()

if __name__ == "__main__":
    main()
