### Macenko method for normalization

import staintools
import PIL
from PIL import Image
import numpy as np
import os

class MyHETransform:
    """Normalize HE colour distribution"""
    def __init__(self, henorm='', path_to_target_im=''):
        if henorm == 'macenko':
            self.transform = CustomMacenkoNormalizer(path_to_target_im=path_to_target_im).transform
        elif henorm == 'babak':
            raise NotImplementedError
            self.transform = CustomBabakNormalizer()
        else:
            self.transform = lambda x: x

    def __call__(self, x):
        return self.transform(x)

class CustomMacenkoNormalizer():
    """Macenko method for H&E normalization. Takes a target tile, and transforms the color space of other tiles
    to the target tile H&E colour distribution
    """
    def __init__(self, path_to_target_im=""):
        self.normalizer = staintools.stain_normalizer.StainNormalizer(method='macenko')
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
        im = np.array(im)
        transformed_im = self.normalizer.transform(im)
        return Image.fromarray(transformed_im)


class CustomBabakNormalizer():
    """
    Babak normalizer, as 
    https://github.com/computationalpathologygroup/WSICS
    https://github.com/francescociompi/stain-normalization-isbi-2017/blob/master/apply_stain_normalization.py
    """

    def __init__(self, root_dir_for_luts):
        self.root_dir_for_luts = root_dir_for_luts

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
        filename = im.filename
        im = np.asarray(im)

        #TODO Find a way to map from tile -> WSI -> LUT.

        lut_filename = os.path.splitext(filename)[0] + '_lut.tif'
        lut = np.asarray(Image.open(lut_filename)).squeeze()
        normalized_tile = self.apply_lut(im, lut)
        normalized_tile = Image.fromarray(normalized_tile)
        return normalized_tile












