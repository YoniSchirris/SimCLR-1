## This is stain_normalizer as copied from the staintools repo
## Instead, this imports a slight adjustment to get_concentrations
## It does not use multiprocessing within get_concentrations itself, as this clashes with pytorch' multiprocessing of the dataloader

import numpy as np


from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.miscellaneous.optical_density_conversion import convert_OD_to_RGB

from modules.custom_staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from modules.custom_staintools.miscellaneous.get_concentrations import get_concentrations


class StainNormalizer(object):

    def __init__(self, method):
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        elif method.lower() == 'vahadane':
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception('Method not recognized.')

    def fit(self, target):
        """
        Fit to a target image.

        :param target: Image RGB uint8.
        :return:
        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target, multiThread=True)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        self.stain_matrix_target_RGB = convert_OD_to_RGB(self.stain_matrix_target)  # useful to visualize.

    def transform(self, I, filename=''):
        """
        Transform an image.

        :param I: Image RGB uint8.
        :return:
        """
        stain_matrix_source = self.extractor.get_stain_matrix(I)

        # Yoni's edit: get_strain_matrix returns false if the linalg doesn't work out. If that doesn't work out, we return an untransformed image
        if isinstance(stain_matrix_source, np.ndarray):
            source_concentrations = get_concentrations(I, stain_matrix_source, multiThread=False)
            maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
            source_concentrations *= (self.maxC_target / maxC_source)
            tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
            return tmp.reshape(I.shape).astype(np.uint8)
        else:
            print(f"=== This image was seen as background by Macenko method, and is therefore not transformed: {filename}")
            return I.astype(np.uint8)
