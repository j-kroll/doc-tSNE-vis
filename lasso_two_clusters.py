import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromCollection:

    def __init__(self, ax, collection, is_first_cluster):
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

    def activate(self):
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        if self.is_first_cluster:
            print('Is first cluster? {}'.format(self.is_first_cluster))
            self.fc[:] = self.colors[0]
            self.fc[self.ind] = self.colors[1]
        else:
            print('Is first cluster? {}'.format(self.is_first_cluster))
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
            next_selector.activate()
            next_selector.fc = self.fc
            print(next_selector.fc.size)
            print(next_selector.fc[0])

if __name__ == '__main__':

    data = np.random.rand(100, 2)

    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector1 = SelectFromCollection(ax, pts, True)
    selector2 = SelectFromCollection(ax, pts, False)

    selector1.activate()

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
        elif event.key == "enter":
            print("Finished.")
            ax.set_title("")
            fig.canvas.draw()
        elif event.key == "x":
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Select A cluster...")

    plt.show()
