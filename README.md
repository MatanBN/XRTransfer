# XRTransfer: Transfer Learning Between Related Tasks Using Expected Label Proportions
Code and dataset for EMNLP2019 [paper](https://arxiv.org/abs/1909.00430) "Transfer Learning Between Related Tasks Using Expected Label Proportions".
To run this code you need to download python 2.7, numpy package and the [DyNet framework](https://github.com/clab/dynet).
# Data
1. To use this code you need to download the glove embeddings vectors, you can download
them [from this link](https://drive.google.com/file/d/1fa2xOnlHJ5A8-Y480Vs5_uGp8X2-PhuM/view)  
After downloading, the Glove.txt file should be located in the project's data folder.
2. You need the unlabeled parsed reviews, download them [from this link](https://drive.google.com/open?id=1xEmLU6UxOlCjfeVqLrMPFfFpuzz4Hjqn) (if you skip to step 7 and download the necessary [files](https://drive.google.com/open?id=1CPNnt5V2wAHI2h01a4sSYmTMV0rkYKJR) to do so then you don't have download this)

# 
# Running
To train an aspect/fragment based model you need to run each of the following steps. Note that if you are only interested in the XR training, you can skip to step 7 by downloading the required files and unzipping them in the main folder, download them [from this link](https://drive.google.com/open?id=1CPNnt5V2wAHI2h01a4sSYmTMV0rkYKJR).
1. Train a sentence-level classifier-to do so, run "python run_sentence_based_model.py, a model called "BiLSTMClassifier" will be created in the models folder (the "models" folder will be created as well if it does not exist yet). 
2. Decompose the unlabeled parsed reviews, this will create an unlabeled dataset of sentences and fragments-to do so, run "python decompose_sents.py", a file called "unlabeled" will be created in the data folder. 
3. Decompose the semeval dataset, this will create a dataset of sentence, their aspects, and an associated fragments of the aspects-to do so and the data/semeval16 and data/semeval15 folders, run "python decompose_sents_aspects.py", this will create train,dev and test files in the semeval15 and semeval16 folders.
4. Label the unlabeled decomposed reviews using the sentence-level classifier-to do so run "python label_sents.py data/unlabeled data/xr/train", this will create a file called train in data/xr, this will be the XR training file.
5. Label the sentences in the semeval train+dev set, this will be used to estimate the distribution of aspects sentiment given sentence sentiment-to do so, run "python label_sents.py data/semeval15/train,data/semeval15/dev data/xr/sents_aspects_labels" this will create the file "sents_aspects_labels" which will contain noisy labels for the sentence-level sentiment of the ABSC datasets.
6. Estimate the distribution of aspect-level sentiment given sentence-level sentiment, this provides the signal to train a model using the xr loss-to do so run "python estimate_dists.py", this will create a .npy file which will contain the conditional distributions.
7. Train a fragment based model using the xr loss-to do so run "python run_fragment_based_model.py --train data/xr/train --test data/semeval16/test", this will train a fragment based model and test it on on the semeval-2016 test set, the model will be saved so you can also test it later on the semval15 test so if you run "python run_fragment_based_model.py --test data/semeval15/test". Note that due to the very large corpus, the training procedure might take a while (in our server it took 4-5 days).
8. If you want to further finetune the fragment-based model to an aspect-based model run "python finetune_aspect_based_model.py", it will finetune it using the semeval16 dataset, if you want to use the semeval15 dataset, run "python finetune_aspect_based_model.py --train data/semeval15/train --dev data/semeval15/dev --test data/semeval15/test --model_path BiLSTMAttFinetuning2015"

# Cite
If you use the code, please cite the following paper:
```
@inproceedings{ben-noach-goldberg-2019-transfer,
    title = "Transfer Learning Between Related Tasks Using Expected Label Proportions",
    author = "Ben Noach, Matan  and
      Goldberg, Yoav",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1004",
    doi = "10.18653/v1/D19-1004",
    pages = "31--42",
    abstract = "Deep learning systems thrive on abundance of labeled training data but such data is not always available, calling for alternative methods of supervision. One such method is expectation regularization (XR) (Mann and McCallum, 2007), where models are trained based on expected label proportions. We propose a novel application of the XR framework for transfer learning between related tasks, where knowing the labels of task A provides an estimation of the label proportion of task B. We then use a model trained for A to label a large corpus, and use this corpus with an XR loss to train a model for task B. To make the XR framework applicable to large-scale deep-learning setups, we propose a stochastic batched approximation procedure. We demonstrate the approach on the task of Aspect-based Sentiment classification, where we effectively use a sentence-level sentiment predictor to train accurate aspect-based predictor. The method improves upon fully supervised neural system trained on aspect-level data, and is also cumulative with LM-based pretraining, as we demonstrate by improving a BERT-based Aspect-based Sentiment model.",
}
```
