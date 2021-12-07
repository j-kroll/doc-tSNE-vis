#pip3 install -qU sentence-transformers bioinfokit mplcursors

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import mplcursors
import nltk
from nltk import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import re
from scipy.spatial import distance
import seaborn as sns
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
        self.collection.set_facecolors(self.fc) # TODO try out set_sizes() to not interfere with chapter color encoding

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

    def get_adj_matrix(lassoed_df, all_df, color, cluster_ax):
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
        cluster_sc = cluster_ax.scatter(*zip(*lassoed_df.iloc[[max_uv[0], max_uv[1], min_uv[0], min_uv[1]], :]["tsne"]), c="black", s=10)
        line_plt = ax4.scatter([0,1,4,5,7,9,10], [2,2,2,2,2,2,2], c="black", s=12)
        line_plt = ax4.scatter([2,4,6,8,12,15], [4,4,4,4,4,4], c="green", s=12)
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
            lassoed_df_A = pd.concat(lassoed_dfs_A, ignore_index=True)
            print("\n".join(lassoed_sentences_A))
            print(get_adj_matrix(lassoed_df_A, df, selector1.colors[1], ax2))

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
            print(get_adj_matrix(lassoed_df_B, df, selector2.colors[2], ax3))

        elif event.key == "f":
            print("Finished.")
            fig.suptitle("Finished cluster selection (X to exit)")
            fig.canvas.draw()

        elif event.key == "x":
            plt.close("all")

    nltk.download("punkt")

    f = open("sherlock.txt", "r")
    contents = f.read().replace("\n", " ")
    sent_text = sent_tokenize(contents)
    print("Read and tokenized sentences...")
    print(sent_text[:3])
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sent_text)
    print("Got high-dimensional sentence embeddings...")
    
    tsne_emb = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(sentence_embeddings)
    print("Calculated t-SNE embeddings...")
    print(len(sent_text))
    print(sentence_embeddings.shape)
    print(tsne_emb.shape)
    
    colors = sns.color_palette("Paired", 16)
    print(len(colors), "colors")

    sents_by_chapter = []
    ch = 0
    new_ch = False
    ch_sent = []

    data = []
    for index, s in enumerate(sent_text):
        color = colors[ch]
        label = "Ch. {} (n={})".format(str(ch).zfill(2), str(index).zfill(3))
        data.append((tsne_emb[index], sentence_embeddings[index], ch, sent_text[index], color, label))
        new_ch = False
        if re.match("(.*( ){2,}[IVX]{1,}\.)|(^[IVX]{1,}\.)", s):
            new_ch = True
            ch += 1
    print(ch, "Chapter index")
    print(len(data), "sentence datapoints\n")

    ch_chunks = []
    idx = 0
    for c in sents_by_chapter:
        ch_chunks.append(idx)
        idx += len(c)

    df = pd.DataFrame(data, columns=["tsne", "high_dim", "ch", "sent", "color", "label"])
    print(df.info())

    fig = plt.figure()
    grid = plt.GridSpec(3, 3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[0:2, 1])
    ax3 = plt.subplot(grid[0:2, 2])
    ax4 = plt.subplot(grid[2, :])
    fig.set_dpi(150)
    fig.canvas.mpl_connect("key_press_event", accept)

    ax4.yaxis.set_major_locator(ticker.NullLocator())
    ax4.xaxis.set_ticks_position("bottom")
    ax4.tick_params(which="major", width=1.00)
    ax4.tick_params(which="major", length=5)
    ax4.tick_params(which="minor", width=0.75)
    ax4.tick_params(which="minor", length=2.5)
    ax4.set_ylim(0, 5)
    ax4.patch.set_alpha(0.0)

    sc = ax1.scatter(*zip(*df["tsne"]), c=df.color, s=8) #, label=df.label) # TODO add label/legend

    selector1 = SelectFromCollection(ax1, sc, True)
    selector2 = SelectFromCollection(ax1, sc, False)

    selector1.activate(ax1)

    cursor = mplcursors.cursor(hover=True)
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
