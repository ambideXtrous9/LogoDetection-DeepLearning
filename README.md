# LOGO-DETECTION-USING-CNN

LOGO DETECTION STEPS

1. Download the dataset from the given link:
http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
and unzip it inside a folder with 4 text files . Then  again unzip the flickr_logo.

2. After that open the terminal or cmd(administrator) and install following packages:
Pip install tensorflow
Pip install opencv-python
Pip install scipy
Pip install sklearn
Pip install selectivesearch
Pip install numpy

3. Now unzip all the codes in the same folder where you kept the flickr logo folder

4. Now run the command:   python3 GEN_TEXT.py  
This will generate the  train_annot_with_bg_class.txt file

5. Once it is completed then run : python3 CROP_N_AUG.py 
As we do not have large no. of images so this code will crop the images in size of 64x32 from the Flickr image folder and apply augmentation method to make the dataset large for 
Training .This code also classifies the images brand wise which will be found inside 
This automatically created folder: flickr_logos_27_dataset_cropped_augmented_images
Pickle file also be created here which contains features of the images such as width, height and no. of channel.

6. Then run : python3 TRAIN_TEST_SPLIT.py
This will create train(90%) test(10%) partition for each brand inside flickr_logos_27_dataset_cropped_augmented_images this folder and create the pickle files


7. Then run the : pyhton3 MODEL.py
This is the code for train the model and the trained models will be saved inside train_models folder. These are the parameter for training I have used. You can change these values in MODEL.py
LEARING_RATE = 0.0001
MAX_STEPS = 20001
BATCH_SIZE = 64
PATCH_SIZE = 5

8. Once model is trained and saved then run :  python3 DETECT.py <test_image_filename.jpg>
Place the test image where you kept all the codes.
This is the threshold probability used for detection:
PRED_PROB_THRESH = 0.999
You can lower this value at the top portion of the code DETECT.py
