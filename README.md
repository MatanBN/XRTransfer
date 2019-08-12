# XRTransfer
Code and dataset for "Transfer Learning Between Related Tasks Using Expected Label Proportions" paper.
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