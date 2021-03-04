import pandas as pd
import os
from PIL import Image

# basis_file='/project/schirris/basisscripts/step_3/data_basis_brca_with_labels_and_splits.csv'
# basis_root='/project/schirris/tiled_data_large'

tcga_file='/home/yonis/histogenomics-msc-2019/yoni-code/MsiPrediction/metadata/tcga/brca_dx/data_tcga_brca_dxwith_ddr_labels_and_splits_dropna_subsample_500_with_hrd_tertile.csv'
tcga_root='/project/yonis/tcga_brca_dx/tiled_data_large'
print('hi')
df = pd.read_csv(tcga_file).sample(frac=1).reset_index()
print('hi2')

for label in [0,1]:
    for index, row in df[df['tertile_HRD_Score']==label].reset_index().iterrows():
        print(index)
        im_name = os.path.join(tcga_root, f'case-{row["case"]}',
                                    row['dot_id'],
                                    'jpeg',
                                    f"tile{row['num']}.jpg"
                              )
        print("Top tertile HRD" if label == 1 else "Bottom tertile HRD")
        display(Image.open(os.path.join(im_name)))
        if index == 2:
            break





