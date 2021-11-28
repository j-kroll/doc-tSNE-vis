#pip3 install -qU sentence-transformers bioinfokit mplcursors

from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import mplcursors
import nltk
from nltk import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import textwrap
import warnings

class SelectFromCollection:

    def __init__(self, ax, collection, is_first_cluster):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.is_first_cluster = is_first_cluster

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.fc = collection.get_facecolors()
        self.colors = np.array([[0.6, 0.6, 0, 1], [0.7, 0.4, 1, 1], [0, 0.8, 0.8, 1]])
        self.fc = np.tile(self.fc, (self.Npts, 1))

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

    def accept(event):
        if event.key == "a":
            print("Selected A cluster:")
            print(selector1.xys[selector1.ind])
            selector1.disconnect(selector2)
            ax.set_title("Select B cluster...")
            fig.canvas.draw()
        elif event.key == "b":
            print("Selected B cluster:")
            print(selector2.xys[selector2.ind])
            selector2.disconnect()
            ax.set_title("Selected 2 clusters!")
            fig.canvas.draw()
        elif event.key == "f":
            print("Finished.")
            ax.set_title("Finished cluster selection (X to exit)")
            fig.canvas.draw()
        elif event.key == "x":
            plt.close("all")

    warnings.filterwarnings("ignore")

    nltk.download("punkt")

    f = open("sherlock.txt", "r")
    contents = f.read().replace("\n", " ")
    sent_text = sent_tokenize(contents)[:400]
    print("Read and tokenized sentences...")
    print(sent_text[:3])
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sent_text)
    print("Got high-dimensional sentence embeddings...")
    
    tsne_emb = TSNE(n_components=2, perplexity=30.0, n_iter=250, verbose=1).fit_transform(sentence_embeddings) # TODO n_iter=1000
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
        data.append((tsne_emb[index], ch, sent_text[index], color, label))
        new_ch = False
        if re.match("(.*( ){2,}[IVX]{1,}\.)|(^[IVX]{1,}\.)", s):
            new_ch = True
            ch += 1

    print("\n----\n")
    print(data[13])
    print("\n")
    print(data[44])
    print("\n----\n")
    print(ch, "Chapter index")
    print(len(data), "sentence datapoints\n")

    ch_chunks = []
    idx = 0
    for c in sents_by_chapter:
        ch_chunks.append(idx)
        idx += len(c)

    df = pd.DataFrame(data, columns=["tsne", "ch", "sent", "color", "label"])

    fig, ax = plt.subplots()
    fig.set_dpi(150)
    fig.canvas.mpl_connect("key_press_event", accept)

    sc = ax.scatter(*zip(*df["tsne"]), c=df.color, s=8) #, label=df.label) # TODO add label/legend

    selector1 = SelectFromCollection(ax, sc, True)
    selector2 = SelectFromCollection(ax, sc, False)

    selector1.activate(ax)

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        textwrap.fill(
            df["sent"][sel.index], 20
        )
    ))

    ax.set_title("Select A cluster...")

    plt.savefig("tsne_chapters_lasso.png")
    plt.show()

if __name__ == "__main__":
    main()
