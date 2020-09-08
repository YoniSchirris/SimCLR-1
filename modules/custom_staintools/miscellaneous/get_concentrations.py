import spams

from staintools.miscellaneous.optical_density_conversion import convert_RGB_to_OD


def get_concentrations(I, stain_matrix, regularizer=0.01, multiThread=False):
    """
    Estimate concentration matrix given an image and stain matrix.

    :param I:
    :param stain_matrix:
    :param regularizer:
    :param multiThread: Set numThreads of spams.lasso to -1 if multiThread, else set to a single thread
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    numThreads = 1 if not multiThread else -1
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True, numThreads=numThreads).toarray().T
