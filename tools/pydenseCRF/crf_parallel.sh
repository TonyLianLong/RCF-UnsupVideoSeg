SEQS='blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog drift-chicane drift-straight goat horsejump-high kite-surf libby motocross-jump paragliding-launch parkour scooter-black soapbox'
ANNOTATION_DIR=./saved/saved_rcf_stage2.2/saved_eval_export
# 20 epochs on davis
STEP=4320
# Note that the parallel command should come from moreutils
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 parallel -j 32 python tools/pydenseCRF/crf.py --input 'data/data_davis/JPEGImages/480p' --output output --annotation-dir $ANNOTATION_DIR --step $STEP --seq -- $SEQS
