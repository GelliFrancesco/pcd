# PCD

Dataset and Source Code for the ACM MM18 paper: "Beyond the Product: Discovering Image Posts for Brands in Social Media"

In this package we provide:
* Instagram dataset, consisting in list of posts including metadata and Instagram code (which can be used to retrieve the original post).
* List of negative samples that were used for training.
* Python source code for replicating paper experiments and extracting visual features.


## REQUIREMENTS
Required Python 2, with packages: PyTorch, pandas, sklearn, tqdm
Note: computation on cpu can be slow.


## DATASET
Dataset can be downloaded [here](https://drive.google.com/file/d/1LtmuhaqKfHX-ExEt5FLPNYxThptS4obl/view?usp=sharing). It consists in:
* brand_list.csv: list of brands with additional information
* training/posts.csv: metadata for training posts (brand_username,post_code,num_likes,num_comments,date_posted,date_crawled)
* training/captions.csv: text captions for training posts 
* training/neg_samples.csv: negative samples used for training
* testing/posts.csv: metadata for testing posts (brand_username,post_code,num_likes,num_comments,date_posted,date_crawled)
* testing/captions.csv: text captions for testing posts 
* features/features.npy: visual features, previously extracted using the provided extract_visual_feature.py script. Not released given large size.
* features/map_list.pickle: indicates in which way visual features are ordered. Also generated with the provided extract_visual_feature.py script.


## EXTRACT VISUAL FEATURES
1. Install the required Python packages.
2. Download and unzip the dataset in the folder pcd/.
3. Download the image posts for both training and testing datasets in a single folder. Images can be downloaded wit the Instagram API or scraped from Instagram Webiste using the shortcode ID provided in posts.csv. Each single image needs to be saved with the name: [shortcode].jpg. For video and multiple-image posts (sidecars) a single image need to be selected (for the paper we selected the video thumbnail and the first image in the sidecar).
4. In code/extract_visual_features.py change the variable "image_folder_path" to match the image directory.
5. In code/extract_visual_features.py change settings such as gpu and batch size
6. Move to code/ folder and execute python extract_visual_features.py


## RUNNING TRAINING CODE
1. Install the required Python packages.
2. Download and unzip the dataset in the folder pcd/.
3. Extract the visual features as explained in the previous paragraph.
4. In code/train.py change settings such as gpu and batch size
5. Move to code/ folder and execute python train.py


## RUNNING TESTING CODE
1. Install the required Python packages.
2. Download and unzip the dataset in the folder pcd/.
3. Extract the visual features as explained in the "EXTRACT VISUAL FEATURES" paragraph.
4. In code/test.py change settings such as gpu
5. Move to code/ folder and execute python test.py
