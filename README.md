# MagiNet

It's a pytorch implementation of paper "MagiNet: Mask-Aware Graph Imputation Networkfor Missing Spatio-temporal Data".

## Requirements

```shell
pip install -r requirements.txt
```

## Datasets

1. Download the raw dataset (link will be open source soon)

   - PEMS-BAY
   - METR-LA
   - Seattle
   - Chengdu
   - Shenzhen

2. Preprocess the dataset:

   ```shell
   python prepare_split_data.py --dataset='METR-LA'
   
   python prepare_miss_data.py --dataset='METR-LA' --miss_mechanism='MCAR' --miss_ratio=0.5 --seqlen=12
   ```

## How to run

```
python main.py --dataset='METR-LA'
```

