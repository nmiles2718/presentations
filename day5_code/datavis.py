#!/usr/bin/env python

from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval, LogStretch, ManualInterval
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patch
import matplotlib.gridspec as gridspec
import numpy as np



class DataVisualizer(object):

    def __init__(self):
        self._segm_cmap = None

    @property
    def segm_cmap(self):
        return self._segm_cmap

    @segm_cmap.setter
    def segm_cmap(self, value):
        self._segm_cmap = value

    def rand_cmap(self, N):
        """ Generate a random colormap for the segmentation map

        Parameters
        ----------
        N

        Returns
        -------

        """
        ncolors = N + 1
        prng = np.random.RandomState(1234)
        h = prng.uniform(low=0.0, high=1.0, size=ncolors)
        s = prng.uniform(low=0.2, high=0.7, size=ncolors)
        v = prng.uniform(low=0.5, high=1.0, size=ncolors)
        hsv = np.dstack((h, s, v))

        rgb = np.squeeze(colors.hsv_to_rgb(hsv))
        rgb[0] = (0, 0, 0)
        cmap = colors.ListedColormap(rgb)
        return cmap

    def plot_segmap(
            self,
            img_data,
            segmap,
            centroids=None,
            ax1=None,
            ax2=None,
            vmin=None,
            vmax=None,
            units=None,
            fs=10
    ):
        """ Plot segmentation map returned by SExtractor

        Parameters
        ----------
        img_data
        segmap
        centroids
        ax1
        ax2
        vmin
        vmax
        units
        fs

        Returns
        -------

        """
        if vmin is not None and vmax is not None:
            interval = ManualInterval(vmin=vmin, vmax=vmax)
        else:
            interval = ZScaleInterval()

        norm = ImageNormalize(
            img_data,
            stretch=LinearStretch(),
            interval=interval
        )

        segm_cmap = self.rand_cmap(np.max(segmap))
        self._segm_cmap = segm_cmap
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(nrows=2,
                                           ncols=1,
                                           sharex=True,
                                           sharey=True)
        else:
            fig = ax1.get_figure()

        im1 = ax1.imshow(img_data, norm=norm, cmap='gray', origin='lower')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label(units, fontsize=fs)
        im2 = ax2.imshow(segmap, cmap=self.segm_cmap, origin='lower')

        if centroids is not None:
            for xy in centroids:
                aperture = self.mk_aperture(xy, r=3, c='red')
                ax1.add_patch(aperture)

        return fig, ax1, ax2

    def mk_aperture(self, xy, r, c):
        """

        Parameters
        ----------
        xy
        r
        c

        Returns
        -------

        """
        aperture = patch.Circle(xy=xy,
                                radius=r,
                                color=c,
                                fill=False,
                                lw=1.5)
        return aperture

    def plot_sources(
            self,
            data,
            centroids,
            vmin=0,
            vmax=1000,
            radius=3,
            color='red',
            ax=None
    ):
        """Draw apertures at each position (x_pos[i], y_pos[i])

        Parameters
        ----------
        data
        centroid
        radius
        color
        ax

        Returns
        -------

        """
        m_interval = ManualInterval(vmin=vmin, vmax=vmax)
        norm = ImageNormalize(data,
                              stretch=LogStretch(),
                              interval=m_interval)
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.grid(False)
        ax.imshow(data,
                  norm=norm,
                  cmap='gray',
                  origin='lower')


        for xy in centroids:
            aperture = self.mk_aperture(xy, radius, color)
            ax.add_patch(aperture)
        return ax

