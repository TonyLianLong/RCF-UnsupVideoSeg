# DAVIS 2016 evaluation
This tool is based on https://github.com/hkchengrex/davis2016-evaluation, with modifications to make DAVIS 2016 work on our exported segmentation masks (by default it assumes 2017). The ownership of the code in this directory, except the modifications, belongs to the author.

This is **not** an official script. However, the results should be close to the original one, from the original README.

Example (in the project main directory):
```shell
python tools/davis2016-evaluation/evaluation_method.py --task unsupervised --davis_path data/data_davis --year 2016 --step 4320 --results_path saved/saved_rcf_stage2.2/saved_eval_export_crf
```

## Notes from the Original README
Using the [precomputed results](https://davischallenge.org/davis2016/soa_compare.html), the numbers are the same as those on the leaderboard so I think this script is correct. Note that it accepts results in the 0~255 (thresholded at 128) format, not the 0/1 pixel format. 

See also:

https://github.com/davisvideochallenge/davis2017-evaluation

https://github.com/davisvideochallenge/davis2017-evaluation/issues/4