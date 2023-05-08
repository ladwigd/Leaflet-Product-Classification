# Leaflet-Product-Classification
This git repository contains the code to the paper "Fine-Grained Product Classification on Leaflet Advertisements".

## Abstract
We describe a first publicly available fine-grained product recognition dataset based on leaflet images. Using advertisement leaflets, collected over several years from different European retailers, we provide a total of 41.6k manually annotated product images in 832 classes. Further, we investigate three different approaches for this fine-grained product classification task, Classification by Image, by Text, as well as by Image and Text. The approach "Classification by Text" uses the text extracted directly from the leaflet product images. We show, that the combination of image and text as input improves the classification of visual difficult to distinguish products. The final model leads to an accuracy of 96.4% with a Top-3 score of 99.2%.<br>
![Visual Abstract Product Leaflet Classification](/reports/visual_abstract.png)
<br>
The figure depicts an example of the promotions of the same product in the leaflets of two different retailers. Price monitoring based on printed leaflets is a key data analysis task in retail, which technically can be defined as a fine-grained, multi-modal classification problem.

## Data
The Dataset can be found here: [Products Leaflets Dataset](https://zenodo.org/record/7869954#.ZFTN8M7P3tV)

## Paper
Has bee accepted at the [CVPR 23 Workshop on Fine-Grained Visual Categorization](https://sites.google.com/view/fgvc10)

Preprint is available here: [Fine-Grained Product Classification on Leaflet Advertisements](https://arxiv.org/abs/2305.03706)

## Code
The code is written in Python, the models are build with Pytorch.
It includes the image classification, text extraction, text classification and model combination.

### Installation
Linux:
```
pip install split-folders
apt install tesseract-ocr
apt-get install tesseract-ocr-deu
pip install pytesseract
```
