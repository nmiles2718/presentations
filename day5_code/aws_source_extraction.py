#!/usr/bin/env python
import logging
import os
import shutil

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.visualization import ImageNormalize, LinearStretch, \
    ZScaleInterval, ManualInterval
import boto3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('ggplot')
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

# Local python class that contains all the source finding functionality
from extract_photometry import Photometry

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    )
LOG = logging.getLogger('ACSLambdaSky')
LOG.setLevel(logging.INFO)


def download_file(event):
    """Download the FITS file passed in the Lambda Event

    Parameters
    ----------
    event : dict
        Python `dict` with data needed by the Lambda function

    Returns
    -------

    """
    fname = event['fits_s3_key']
    bucket_name = event['fits_s3_bucket']
    s3 = boto3.resource('s3')
    bkt = s3.Bucket(bucket_name)
    bkt.download_file(
        fname,
        f"/tmp/{os.path.basename(fname)}",
        ExtraArgs={"RequestPayer": "requester"}
    )
    return f"/tmp/{os.path.basename(fname)}"

def get_image_metadata(phot):
    """Store some key metadata

    Parameters
    ----------
    phot : `extract_photometry.Photometry`
    data_dict :

    Returns
    -------

    """
    data_dict = {}
    keys = [
        'targname',
        'exptime',
        'filter1',
        'filter2',
        'expstart',
        'aperture',
        'expstart'
    ]
    for key in keys:
        if key == 'filter1' and \
                'clear' not in phot.data['prhdr'][key].lower():
            data_dict['filter'] = phot.data['prhdr'][key]
        elif key == 'filter2' and \
                'clear' not in phot.data['prhdr'][key].lower():
            data_dict['filter'] = phot.data['prhdr'][key]
        elif 'filt' not in key:
            data_dict[key] = phot.data['prhdr'][key]

    return data_dict


def clean_up(dirname="/tmp"):
    """ Delete all files created and stored in the lambda tmp directory

    Lambda can only store so much data in tmp and when many function
    invocations occur in a short period of time, memory will sometimes persist
    innovcations and so to avoid filling up the maximum allowable memory for
    a function, we need to delete the files we stored

    Parameters
    ----------
    dirname : str
        path to /tmp on lambda function

    Returns
    -------

    """
    # try:
    shutil.rmtree(dirname, ignore_errors=True)


def plot_inset(
        data,
        ax,
        centroids=None,
        cmap=None,
        vmin=None,
        vmax=None,
        w=100,
        h=100
):
    """Plot an inset in the image plot

    Parameters
    ----------
    data
    norm
    ax
    w
    h

    Returns
    -------

    """
    ax_inset = inset_axes(ax, width="30%", height="30%")
    ax_inset.grid(False)
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec='r', lw=1.25)

    for label in ['bottom','top','left','right']:
        ax_inset.spines[label].set_color('r')

    ax_inset.tick_params(
        axis='both',
        which='major',
        bottom=False,
        top=False,
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False
    )

    ymax, xmax = data.shape

    xlim = (xmax//5 - w, xmax//5 + w)
    ylim= (ymax//3 - h, ymax//3 + h)
    if cmap is not None:
        ax_inset.imshow(
            data,
            cmap=cmap,
            origin='lower',
        )
    else:
        if vmin is not None and vmax is not None:
            interval = ManualInterval(vmin=vmin, vmax=vmax)
        else:
            interval = ZScaleInterval()
        norm = ImageNormalize(
            data,
            stretch=LinearStretch(),
            interval=interval
        )
        ax_inset.imshow(
            data,
            norm=norm,
            origin='lower',
            cmap='gray'
        )

    if centroids is not None:
        for xy in centroids:
            aperture = Circle(
                xy=xy,
                radius=3,
                color='red',
                fill=False,
                lw=0.75
            )
            ax_inset.add_patch(aperture)
    ax_inset.set_xlim(xlim)
    ax_inset.set_ylim(ylim)
    return ax_inset


def plot_sources(phot, metadata, units=None):
    """

    Parameters
    ----------
    phot
    metadata
    units

    Returns
    -------

    """
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6,8),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace':0, 'hspace':0.05}
    )

    for a in ax:
        a.grid(False)
    # Stack chip1 on top of chip 2
    stacked_segmap = np.concatenate(
        [phot.data['sci1_segmap'], phot.data['sci2_segmap']], axis=0
    )

    stacked_data = np.concatenate(
        [phot.data['sci1'], phot.data['sci2']], axis=0
    )

    centroids_chip1 = list(
        zip(
            phot.phot_catalog['sci1']['x'],
            phot.phot_catalog['sci1']['y']
        )
    )

    # Add 2048 to the y coords for chip 1 to transform into the stacked frame
    # Remember, the **second** SCI extension corresponds to chip 1
    centroids_chip2 = list(
        zip(
            phot.phot_catalog['sci2']['x'],
            phot.phot_catalog['sci2']['y'] + 2048
        )
    )
    centroids = centroids_chip1 + centroids_chip2

    fig, ax1, ax2 = phot.plot_segmap(
        img_data=stacked_data,
        segmap=stacked_segmap,
        centroids=centroids,
        ax1=ax[0],
        ax2=ax[1],
        units=units
    )

    ax_inset1 = plot_inset(
        stacked_data,
        centroids=centroids,
        ax=ax1
    )

    ax_inset2 = plot_inset(stacked_segmap, cmap=phot.segm_cmap, ax=ax2)

    targname = metadata['targname']
    exptime = metadata['exptime']
    filt = metadata['filter']

    ax1.set_title(f"{targname}\n{exptime}, {filt}")

    return fig


def upload_results(event, files_to_upload=[], upload_dir=None):
    s3 = boto3.resource('s3')
    for f in files_to_upload:
        s3.meta.client.upload_file(
            f,
            event['s3_output_bucket'],
            f"results/{upload_dir}/{os.path.basename(f)}"
        )


def process_event(event):
    fname = event['fits_s3_key']
    fname = download_file(event)
    basename = os.path.basename(fname).split('_')[0]
    phot = Photometry(fname=fname)

    # Run the photometry pipeline
    # this will find sources and perform aperture photometry
    phot.run_analysis(
        ap_radius=event['radius'],
        r_in=event['r_inner'],
        r_out=event['r_outer']
    )
    try:
        units = phot.data['sci1_hdr']['BUNIT']
    except KeyError as e:
        try:
            units = phot.data['sci2_hdr']['BUNIT']
        except KeyError as e:
            units = 'unknown'

    metadata = get_image_metadata(phot=phot)
    metadata['aperture_radius'] = event['radius']
    metadata['sky_annulus_inner_radius'] = event['r_inner']
    metadata['sky_annulus_outer_radius'] = event['r_outer']
    # Add the metadata for this image to the combined photometry catalog
    phot.phot_catalog['combined'].meta['comments'] = [
        f"{key}: {val}" for key, val in metadata.items()
    ]
    # Plot the identified sources
    fig = plot_sources(phot=phot, metadata=metadata, units=units)
    # return tb_sky, tb_corners
    fig_fname = f"/tmp/{basename.replace('.fits','.png')}"
    catalog_fname = f"/tmp/{basename.split('.')[0]}_{event['radius']:.0f}pix_aper_phot.cat"

    fig.savefig(
        fig_fname,
        format='png',
        dpi=250,
        bbox_inches='tight'
    )

    phot.phot_catalog['combined'].write(
        catalog_fname,
        format='ascii',
        overwrite=True
    )
    files_to_upload = [fig_fname, catalog_fname]
    upload_results(
        event=event,
        files_to_upload=files_to_upload,
        upload_dir=basename
    )
    # Now we delete the file we downloaded to ensure if memory is persisted
    # between Lambdas, we won't fill up our disk space.
    # https://stackoverflow.com/questions/48347350/aws-lambda-no-space-left-on-device-error
    clean_up(dirname='/tmp')


def handler(event, context):
    process_event(event)



if __name__ == "__main__":
    event={
        'fits_s3_key':'/Users/nmiles/presentations/day5_code/test_data/association1_crj.fits',
        'fits_s3_bucket':'bucket',
        'radius': 3,
        'r_inner': 14,
        'r_outer': 16,
        's3_output_bucket': '',
    }
    handler(event, None)
