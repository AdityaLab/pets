# Performative Time-Series Forecasting

## Publication

Implementation of the paper "Performative Time-Series Forecasting."

Authors: Leo Zhiyuan Zhao, Alexander Rodríguez, B.Aditya Prakash

Paper + Appendix: [https://arxiv.org/abs/2310.06077](https://arxiv.org/abs/2310.06077)

## Training

METR-LA Dataset download link: [https://zenodo.org/record/5146275/files/METR-LA.csv?download=1](https://zenodo.org/record/5146275/files/METR-LA.csv?download=1)

Example to run the covid dataset:

```
python3 run_covid.py --seed 0 --dev cuda:0 --model rnn --fps 0
```

Example to run the metrla dataset:

```
python3 run_metrla.py --seed 0 --dev cuda:0 --model rnn --fps 0
```

Example to run the metrla dataset out-of-distribution test:

```
python3 run_metrla_ood.py --seed 0 --dev cuda:0 --model rnn --fps 0
```

Implemented Models: ```{rnn, lstnet, transformer, informer}```

```fps=0``` is running conventional forecasting models, ```fps=1``` is running fps.


## Contact

If you have any questions about the code, please contact Leo Zhiyuan Zhao at  ```leozhao1997[at]gatech[dot]edu```.

## Citation

If you find our work useful, please cite our work:

```
@article{zhao2023performative,
  title={Performative Time-Series Forecasting},
  author={Zhao, Zhiyuan and Rodriguez, Alexander and Prakash, B Aditya},
  journal={arXiv preprint arXiv:2310.06077},
  year={2023}
}
```
