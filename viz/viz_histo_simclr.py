import torch
import sys
sys.path.append('..')
from modules.transformations.simclr import TransformsSimCLR

paths = ['/Users/yoni/dropbox/uva/ai/nki/histogenomics-msc-2019/yoni-code/MsiPrediction/data/msidata/crc_dx/train/MSS/blk-TGINRMPEMKGT-TCGA-CM-6164-01Z-00-DX1.png',
'/Users/yoni/dropbox/uva/ai/nki/histogenomics-msc-2019/yoni-code/MsiPrediction/data/msidata/crc_dx/train/MSS/blk-TGIQPQRNNDKP-TCGA-D5-6541-01Z-00-DX1.png']

from skimage import io, transform

transformer = TransformsSimCLR(size=224)
for path in paths:
    a,b = transformer(io.imread(path))