#/bin/bash
export

# if DO_PREPROCESS is set, run the preprocessing script
if [ -n "$DSVG_DO_PREPROCESS" ]; then
    echo "Running preprocessing script..."
	uv run dataset/preprocess.py --data_folder $DSVG_PRP_DATA_FOLDER --output_folder $DSVG_PRP_OUTPUT_FOLDER  --output_meta_file  $DSVG_PRP_OUTPUT_FOLDER/meta.csv
    echo "Preprocessing completed. Output saved to $DSVG_PRP_OUTPUT_FOLDER/meta.csv."
    file $DSVG_PRP_OUTPUT_FOLDER/meta.csv
else
    echo "Skipping preprocessing step."
fi


# uv run deepsvg/train.py --config-module configs.unsymbols.ours_hier_ord --log-dir /tmp/deepsvglogdir
#
# env_ours .. contains hyperparameters from DVSG_TRAIN_[BS/NUM_GPUS/DATA_DIR]
uv run deepsvg/train.py --config-module configs.unsymbols.env_ours_hier_ord --log-dir $DSVG_TRAIN_LOGDIR


