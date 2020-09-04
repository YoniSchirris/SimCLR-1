### Macenko method for normalization

import os

import numpy as np
import PIL
import staintools
from PIL import Image

from modules.custom_staintools.stain_normalizer import StainNormalizer


class MyHETransform:
    """Normalize HE colour distribution"""
    def __init__(self, henorm='', path_to_target_im='', lut_root_dir=''):

        if henorm == 'macenko':
            self.transform = CustomMacenkoNormalizer(path_to_target_im=path_to_target_im).transform
        elif henorm == 'babak':
            raise NotImplementedError
            self.transform = CustomBabakNormalizer(lut_root_dir=lut_root_dir)
        else:
            self.transform = lambda x: x

    def __call__(self, x):
        return self.transform(x)

class CustomMacenkoNormalizer():
    """Macenko method for H&E normalization. Takes a target tile, and transforms the color space of other tiles
    to the target tile H&E colour distribution
    """
    def __init__(self, path_to_target_im=""):
        self.normalizer = StainNormalizer(method='macenko')
        target_im = Image.open(path_to_target_im)
        target_im = np.array(target_im)
        self.normalizer.fit(target_im)
        
    def transform(self, im: PIL.Image) -> PIL.Image:
        """Transforms a given image to the initialized target image

        Args:
            im (PIL.Image): Takes a PIL image, as it's developed to be used within a pytorch transforms composition

        Returns:
            PIL.Image: Returns a transforemd PIL Image, so that it can be used further on in the transforms pipeline
        """
        filename = im.filename['tile']
        im = np.array(im)
        transformed_im = self.normalizer.transform(im, filename)
        return Image.fromarray(transformed_im)


class CustomBabakNormalizer():
    """
    Babak normalizer, as 
    https://github.com/computationalpathologygroup/WSICS
    https://github.com/francescociompi/stain-normalization-isbi-2017/blob/master/apply_stain_normalization.py
    """

    def __init__(self, lut_root_dir):
        self.lut_root_dir = lut_root_dir

    def apply_lut(tile, lut):
        """ 
        Apply look-up-table to tile to normalize H&E staining.
        As used in https://github.com/francescociompi/stain-normalization-isbi-2017/blob/master/apply_stain_normalization.py 
        """
        ps = tile.shape # tile size is (rows, cols, channels)
        reshaped_tile = tile.reshape((ps[0]*ps[1], 3))
        normalized_tile = np.zeros((ps[0]*ps[1], 3))
        idxs = range(ps[0]*ps[1])  
        Index = 256 * 256 * reshaped_tile[idxs,0] + 256 * reshaped_tile[idxs,1] + reshaped_tile[idxs,2]
        normalized_tile[idxs] = lut[Index.astype(int)]
        return normalized_tile.reshape(ps[0], ps[1], 3).astype(np.uint8)


    def transform(self, im: PIL.Image) -> PIL.Image:
        wsi_filename = im.filename['wsi'] # In the TCGA dataloader, we save TCGA_ID . DOT_ID . svs in the PIL object
        im = np.asarray(im)

        #TODO Find a way to map from tile -> WSI -> LUT.

        lut_filename = os.path.splitext(filename)[0] + '_lut.tif'
        absolute_lut_filepath = os.path.join(self.lut_root_dir, lut_filename)
        lut = np.asarray(Image.open(absolute_lut_filepath)).squeeze()
        normalized_tile = self.apply_lut(im, lut)
        normalized_tile = Image.fromarray(normalized_tile)
        return normalized_tile
