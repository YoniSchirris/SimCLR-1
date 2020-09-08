cd ..
array=('PARPi7_bin' 'median_mutLoad_silent' 'median_mutLoad_nonsilent' 'median_tp53_score' 'median_rppa_ddr_score' 'median_HRD_Score' 'median_purity' 'median_ploidy' 'median_CNA_frac_altered' 'BER' 'NER' 'MMR' 'FA' 'HDR' 'NHEJ' 'DR' 'TLS' 'NP' 'Others' 'core-BER' 'core-NER' 'core-MMR' 'core-FA' 'core-HR' 'core-NHEJ' 'core-DR' 'core-TLS' 'core-DS' 'any-mut' 'any-base' 'any-mmr' 'any-strand' 'core-base' 'core-mrr' 'core-strand' 'core-mmr-and-hr')
for i in "${array[@]}"
do
    python -m testing.logistic_regression with workers=3 config_file=./config/test-tcga-brca.yaml precompute_features=False use_precomputed_features=True use_precomputed_features_id=437 logistic_epochs=10 ddr_label="$i"
done
