#!/usr/bin/env python
import argparse
import logging
import os

from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from datavis import DataVisualizer
import numpy as np
import scipy.ndimage as ndimage
import sep


parser = argparse.ArgumentParser()
parser.add_argument('-fname',
                    help='Path to image')

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger('FindSources')

class Photometry(DataVisualizer):
    def __init__(self, fname):
        super(DataVisualizer, self).__init__()
        self._fname = fname
        self._data = {}
        self._source_catalog = {}
        self._phot_catalog = {}

    @property
    def source_catalog(self):
        """Source catalog"""
        return self._source_catalog

    @source_catalog.setter
    def source_catalog(self, value):
        self._source_catalog = value

    @property
    def phot_catalog(self):
        return self._phot_catalog

    @phot_catalog.setter
    def phot_catalog(self, value):
        self._phot_catalog = value    

    @property
    def data(self):
        """Data for a given chip"""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def fname(self):
        return self._fname

    @fname.setter
    def fname(self, value):
        self._fname = value

    def read_file(self, fname=None, extnums=[1,2], extname='sci', drz=False):
        if fname is not None:
            f = fname
        else:
            f = self._fname

        if drz:
            with fits.open(f) as hdu:
                ext = 0
                self.data[f"{extname}{1}"] = \
                    hdu[ext].data.byteswap().newbyteorder()
                self.data['prhdr'] = hdu[0].header
                self.data[f"{extname}{1}hdr"] = hdu[ext].header
                self.get_wcs(extnum=0, drz=drz)
        else:
            with fits.open(f) as hdu:
                self.data['prhdr'] = hdu[0].header
                exptime = self.data['prhdr']['EXPTIME']
                for extnum in extnums:
                    ext_tuple = (extname, extnum)
                    try:
                        ext = hdu.index_of(ext_tuple)
                    except (KeyError, ValueError) as e:
                        LOG.info(e)
                    else:
                        self.data[f"{extname}{extnum}_hdr"] = hdu[ext].header
                        # TODO: Add functionality for chekcing the units
                        # if 'sci' in extname:
                            # units = self.data[f"{extname}{extnum}_hdr"]['BUNIT']
                            # if units != 'ELECTRONS':
                        self.data[f"{extname}{extnum}"] = \
                            hdu[ext].data.byteswap().newbyteorder()
                        if extname == 'sci':
                            self.get_wcs(extnum=extnum, drz=drz)

                    if extname == 'sci' and \
                        self.data[f"{extname}{extnum}_wcs"] is None:
                        self.get_wcs(extnum=extnum, drz=drz)

    def get_wcs(self, extnum, drz=False):
        """ Extract the WCS information for each chip"""
        if drz:
            keyname = f'sci{extnum+1}_wcs'
        else:
           keyname =  f'sci{extnum}_wcs'

        with fits.open(self.fname) as hdu:
            # chip 2
            self.data[keyname] = WCS(
               fobj=hdu,
               header=hdu[extnum].header
            )

            #chip 1
            # self.data['sci2_wcs'] = WCS(fobj=hdu, header=hdu[4].header)

        msg = (
            'Retreieving WCS information...\n{}\n'
            'WCS Information for {}\n'
            '{}\n'
            '{}'.format('-'*79, os.path.basename(self.fname),
                        self.data[keyname],
                        '-'*79)
        )
        LOG.info(msg)

    def convert_to_sky(self, catalog, wcs):
        """Convert the pixel coordinates from (x, y) to (RA, DEC)

        Generate a list of astropy.SkyCoord using the image WCS and a list of
        positions taken from the photometry catalog.

        Parameters
        ----------
        phot_table : astropy.table.Table
            Photometry catalog returned by photutils.aperture.aperture_photometry()
        wcs : astropy.wcs.WCS
            WCS object containing the full WCS solution parsed from the FITS
            image's header

        Returns
        -------
        sky_coords : astropy.coordinates.SkyCoord
            RA/DEC of the sources analyzed

        """
        sky_coords = SkyCoord.from_pixel(xp = catalog['x'],
                                   yp = catalog['y'],
                                   wcs = wcs, origin=0)
        return sky_coords

    def find_sources(
            self,
            thresh=3.5,
            extname='sci',
            extnum=1,
            dq_mask=None,
            use_kernel=True,
            deblend_cont=0.01,
            kernel_size=5,
            save=True
    ):
        """ Find sources that are thresh*sigma above background

        Parameters
        ----------
        bkg
        hdr
        thresh
        kernel

        Returns
        -------

        """
        if self.data[f"{extname}{extnum}"] is None:
            self.read_file()

        if use_kernel:
            LOG.info('Creating a 2-D Gaussian kernel')
            kernel = Gaussian2DKernel(
                1, x_size=kernel_size, y_size=kernel_size
            )
            kernel.normalize()
            kernel = kernel.array
        else:
            LOG.info('Using default kernel')
            kernel = np.ones((kernel_size, kernel_size))

        LOG.info('Generating BKG mesh using SExtractor')
        self.data[f"{extname}{extnum}_bkg"] = \
            sep.Background(self.data[f"{extname}{extnum}"])

        # LOG.info(dq_mask)
        gains = self.data['prhdr']['ATODGN*']
        avg_gain = np.sum(gains.values()) / len(gains)
        source_extract_config = {
            'thresh': thresh,
            'err': self.data[f"{extname}{extnum}_bkg"].globalrms,
            'gain': avg_gain,
            'minarea': 11,
            'filter_kernel': kernel,
            'mask': dq_mask,
            'deblend_nthresh': 32,
            'deblend_cont': deblend_cont,
            'clean': False,
            'segmentation_map': True
        }
        LOG.info(
            'Performing global background subtraction to find sources..'
            )
        bkg_subtracted = self.data[f"{extname}{extnum}"] - \
                         self.data[f"{extname}{extnum}_bkg"]
        source_catalog, segmap = sep.extract(
            data=bkg_subtracted,
            **source_extract_config
        )
        self.data[f"{extname}{extnum}_segmap"] = segmap        
        self.source_catalog[f"{extname}{extnum}"] = Table(source_catalog)
        self.source_catalog[f"{extname}{extnum}"]['id'] = \
            np.arange(0, len(source_catalog['x']))

    def compute_photometry(self, extname, extnum, r=3, r_in=14, r_out=16):
        """

        Parameters
        ----------
        extname
        extnum
        r
        r_in
        r_out

        Returns
        -------

        """

        self.phot_catalog[f"{extname}{extnum}"] = \
            self.source_catalog[f"{extname}{extnum}"]['id', 'x', 'y']

        LOG.info(
            'Computing local background within a circular annulus...'
        )
        area_annulus = np.pi*(r_out**2 - r_in**2)
        bkg_sum, bkg_sum_err, flag = sep.sum_circann(
            self.data[f"{extname}{extnum}"],
            x=self.phot_catalog[f"{extname}{extnum}"]['x'],
            y=self.phot_catalog[f"{extname}{extnum}"]['y'],
            rin=14,
            rout=16
        )

        self.phot_catalog[f"{extname}{extnum}"]['bkg_per_pix'] = bkg_sum/area_annulus

        LOG.info(
            'Computing aperture sum within a 3 pixel radius...'
        )
        # Note we use the non-background subtracted one
        ap_sum, apsumerr, flag = sep.sum_circle(
            self.data[f"{extname}{extnum}"],

            self.phot_catalog[f"{extname}{extnum}"]['x'],
            self.phot_catalog[f"{extname}{extnum}"]['y'],
            r=3
        )
        area_ap = np.pi*r**2

        ap_sum_bkg_sub = \
            ap_sum - \
            area_ap * self.phot_catalog[f"{extname}{extnum}"]['bkg_per_pix']
        
        # Very simplistic error computation
        ap_sum_err_final = np.sqrt(
            ap_sum + 
            area_ap * self.phot_catalog[f"{extname}{extnum}"]['bkg_per_pix']
        )

        # Record the aperture sum
        self.phot_catalog[f"{extname}{extnum}"][f'sum_{r:.0f}pix'] = \
            ap_sum_bkg_sub
        # Record the error in the aperture sum
        self.phot_catalog[f"{extname}{extnum}"][f'sum_{r:.0f}pix_err'] = \
            ap_sum_err_final
        # Record the magnitude
        self.phot_catalog[f"{extname}{extnum}"][f'mag_{r:.0f}pix'] = \
            -2.5 * np.log10(ap_sum_bkg_sub)

        nan_filter = np.isnan(
            self.phot_catalog[f"{extname}{extnum}"][f"mag_{r:.0f}pix"]
        )

        # Record the error in the magnitude
        self.phot_catalog[f"{extname}{extnum}"][f'mag_{r:.0f}pix_err'] = \
            1.0857 * ap_sum_err_final / ap_sum_bkg_sub

        # Conver the X/Y coords to RA/DEC
        sky_coords = self.convert_to_sky(
            self.phot_catalog[f"{extname}{extnum}"],
            self.data[f"{extname}{extnum}_wcs"]
        )

        self.phot_catalog[f"{extname}{extnum}"]['ra'] = sky_coords.ra
        self.phot_catalog[f"{extname}{extnum}"]['dec'] = sky_coords.dec

        # Filter out all rows where any value is NaN
        self.phot_catalog[f"{extname}{extnum}"] = \
            self.phot_catalog[f"{extname}{extnum}"][~nan_filter]

    def run_analysis(self, ap_radius=3, r_in=14, r_out=16):
        """
        
        Parameters
        ----------
        ap_radius
        r_in
        r_out

        Returns
        -------

        """
        self.read_file(extname='sci', extnums=[1, 2], drz=False)
        self.read_file(extname='dq', extnums=[1, 2], drz=False)

        for extnum in [1, 2]:
            self.find_sources(
                thresh=3, 
                extname='sci', 
                extnum=extnum, 
                dq_mask=self.data[f"dq{extnum}"].astype(float)
            )
            self.compute_photometry(
                extname='sci', 
                extnum=extnum,
                r=ap_radius,
                r_in=r_in,
                r_out=r_out
            )

        output_columns = [
            'ra',
            'dec',
            'bkg_per_pix',
            f'sum_{ap_radius:.0f}pix',
            f'sum_{ap_radius:.0f}pix_err',
            f'mag_{ap_radius:.0f}pix',
            f'mag_{ap_radius:.0f}pix_err',
        ]
        # Stack the two tables row-wise
        self.phot_catalog['combined'] = vstack(
            [
                self.phot_catalog['sci1'][output_columns],
                self.phot_catalog['sci2'][output_columns]
            ],
            join_type='inner'
        )


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    phot = Photometry(**args)
    phot.run_analysis()