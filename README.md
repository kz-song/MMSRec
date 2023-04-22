# MMSRec

Due to a lack of GPU machines, this work has been delayed for several months. As a result, only a limited number of ablation experiments are included in the paper. We plan to supplement with more experiments when resources become available in the future.

## Environment

Create virtual environment from configuration file

```python
conda env create --file configs/mmsrec.yml
```

Activate virtual environment

```python
conda activate mmsrec
```

Install and download CLIP (used as feature extractor)

```python
sh weights/clip/download.sh
```



## Download Datasets

### Pretraining Datasets

#### 1.Webvid

Go to `/dataset/webvid/preprocess/` directory

Install git-lfs

```python
sudo apt-get install git-lfs
```

Download dataset

```python
git lfs clone https://huggingface.co/datasets/iejMac/CLIP-WebVid.git
```

Extract dataset

```python
sh download.sh
```

Generate training files

```python
python process_item.py
python process_seq.py
```

#### 2.MSR-VTT

Go to `dataset/msrvtt/preprocess` directory

Download dataset

```python
sh download.sh
```

Generate training files

```python
python process_item.py
python process_seq.py
```

### Recommendation Datasets

#### 1.Amazon

Go to `dataset/amazon/preprocess` directory

Download dataset (This will download Beauty, Sports, Clothing, and Home datasets. You can modify and adjust according to your needs)

```python
sh download.sh
```

Scrape image links from the dataset and generate training files

```python
python process_item.py
```

Feature extraction

```python
python extract_features.py
```

#### 2.Movielens-1M

Go to `dataset/movielens-1m/preprocess` directory

Download dataset

```python
sh download.sh
```

Scrape video data

```python
python download_videos.py
```

Generate training files

```python
python process_item.py
```

Feature extraction

```python
python extract_features.py
```



## Quick Start

### 1.Pretraining

Execute the following command to start pretraining

```python
sh pretrain_webvid.sh
```

**Note**: The script assumes each node has 8 GPUs by default. You can modify the following parameter for custom configuration

```
--nproc_per_node=8 \
```

The configuration file for pretraining is `configs/pretraining/pretrain_webvid.yaml`

### 2.Finetune on Amazon

Finetune the pre-trained model on Amazon

```python
sh finetune_amazon.sh
```

The configuration file is `configs/pretraining/finetune_amazon.yaml`

### 3.Finetune on Movielens-1M

Finetune the pre-trained model on Movielens-1M

```python
sh finetune_movielens.sh
```

The configuration file is `configs/pretraining/finetune_movielens.yaml`



























