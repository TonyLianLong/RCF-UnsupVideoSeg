# torchCRF
This is a GPU implementation of CRF that supplements the code for our work Bootstrapping Objectness from Videos by Relaxed Common Fate and Visual Grouping.

It is typically faster when compared to the CPU implementations in `denseCRF` packages.

`torchCRF` is a wrapper built upon [DenseCRF](https://github.com/heiwang1997/DenseCRF). The actual implementation follows its original license. The torch wrapper follows MIT License. Please carry the `LICENSE` file in the main repo and this description file if you plan to detach this implementation for other uses.

# Installation
You need to ensure your CUDA version matches with torch, and you have cuda toolkit 11.x installed. Then you can install with:
```
python setup.py install
```

If you want to develop, run:
```
python setup.py develop
```

Changes to the code will be reflected instantly for Python changes and will be reflected when you compile for CUDA changes.

# Citation
Please cite our work if you find our work inspiring or use our code in your work:
```
@inproceedings{lian2023bootstrapping,
  title={Bootstrapping Objectness from Videos by Relaxed Common Fate and Visual Grouping},
  author={Lian, Long and Wu, Zhirong and Yu, Stella X},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

@article{lian2022improving,
  title={Improving Unsupervised Video Object Segmentation with Motion-Appearance Synergy},
  author={Lian, Long and Wu, Zhirong and Yu, Stella X},
  journal={arXiv preprint arXiv:2212.08816},
  year={2022}
}
```
