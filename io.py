"""
analogy.io
==========
Readers for PIVlab-style output: region-of-interest geometry and the A/B
frame numbers encoded in PIV result filenames.
"""
import re
import pandas as pd

__all__ = [
    'roi_geom',
    'PIV_framenumbers',
]


def roi_geom(filename):
    '''
    Reads a PIvab output file and extracts information about the  region
        of interest, or the selected observation window, outside of which
        data is ignored

    Parameters
    ----------
    filename : str
        Name/location of PIV resurs file containing x,y,u,v data.

    Returns
    -------
    pd.DataFrame
        Compilation of measurements of the selected PIV region of interest.

    '''
    # read data
    data = pd.read_csv(filename,skiprows=3,usecols = [0,1,2,3],\
                               names=['x','y','u','v'])
    width = data.x.max() - data.x.min()
    height  = data.y.max() - data.y.min()
    #return width,height,xmin,xmax,ymin,ymax
    return width,height,data.x.min(),data.x.max(),data.y.min(),data.y.max()


def PIV_framenumbers(filename):
    '''
    Read A/B frame numbers from PIvab output.
    Assumes that the frame number is the last number in the filename before
        extension.
    Also reads filename for input, assuming filenumber is last number before
        extension.
    '''
    frame_line = []
    f=open(filename)
    lines=f.readlines()
    frame_line = lines[1]
    filenumber = re.findall(r'\d+', filename)[-1]
    A_frame = re.findall(r'\d+',frame_line.split()[4])[-1]
    B_frame = re.findall(r'\d+',frame_line.split()[7])[-1]
    f.close()
    return int(filenumber), int(A_frame), int(B_frame)
