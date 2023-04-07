# TISR_APTX4869

Source code of team APTX4869 (3rd x2 MR2HR) in Thermal Image Super-Resolution Challenge Results - PBVS 2023

This codebase is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for the effort and great work of the authors.

## Preprocess

change the paths in `process_scripts/alignment_tisr_mr2hr.py` and run the script to align the MR and HR thermal images first.

## Run train and test

change the paths in the `options_train/exp_name.yml` and `options_test/exp_name.yml` to your own dataset location, and then run the bash scripts to train the model.

```
sh run_train_track1.sh
```

after the model is trained, use the following command to conduct inference

```
python test_track1.py -opt options_test/exp_name.yml
```

you can also use the trained result in [GoogleDrive](https://drive.google.com/drive/folders/1gKn-WgoT9DsuawNKldvMCIEbOJVetVre) to inference the testset directly.


