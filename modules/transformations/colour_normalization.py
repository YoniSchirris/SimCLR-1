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
            self.transform = CustomMacenkoNormalizerHistomicsTK(path_to_target_im=path_to_target_im).transform
        elif henorm == 'babak':
            raise NotImplementedError
            self.transform = CustomBabakNormalizer(lut_root_dir=lut_root_dir)
        else:
            self.transform = lambda x: x

    def __call__(self, x):
        return self.transform(x)

class CustomMacenkoTransformerHistomicsTK():
    """
    As adjusted from
    https://digitalslidearchive.github.io/HistomicsTK/_modules/histomicstk/preprocessing/color_normalization/deconvolution_based_normalization.html#deconvolution_based_normalization

    Initializes with a target image (str, absolute path to target image) and computes the target stain

    .transform() fits the stain of the given image (np array, WxHxC) to the target stain

    """
    def __init__(self, path_to_target_im: str=None, stains=None, stain_unmixing_routine_params=None):
        stains = ['hematoxylin', 'eosin'] if stains is None else stains
        stain_unmixing_routine_params = (
            {} if stain_unmixing_routine_params is None else
            stain_unmixing_routine_params)
        stain_unmixing_routine_params['stains'] = stains
        
        self.stain_unmixing_routine_params = stain_unmixing_routine_params

        im_target = Image.open(path_to_target_im)
        im_target = np.asarray(im_target)
        
        if im_target is None:
            # Normalize to 'ideal' stain matrix if none is provided
            W_target = np.array(
                [stain_color_map[stains[0]], stain_color_map[stains[1]]]).T
            self.W_target = complement_stain_matrix(W_target)

        elif im_target is not None:
            # Get W_target from target image
            self.W_target = stain_unmixing_routine(
                im_target, **stain_unmixing_routine_params)

    def transform(self, im_src: PIL.Image) -> PIL.Image:
        im_src = np.asarray(im_src) # Convert to np array

        _, StainsFloat, _ = color_deconvolution_routine(
            im_src, **self.stain_unmixing_routine_params)

        im_src_normalized = color_convolution(StainsFloat, self.W_target)

        im_src_normalized = Image.fromarray(im_src_normalized) # convert back to PIL Image
        return im_src_normalized
        

class CustomMacenkoNormalizer():
    """Macenko method for H&E normalization. Takes a target tile, and transforms the color space of other tiles
    to the target tile H&E colour distribution

    This uses the staintools implementation

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
