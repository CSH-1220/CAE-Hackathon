# CAE-Hackathon
Eyes on the Water - CAE Hackathon

The introduction of our proposed backend solution to Eyes on the Water CAE hackathon.

## Install and Usage

The required packages are listed in the [requirements.txt](folder/file_to_link_to.txt)

## System demonstration

To demonstrate our system, open the [tutorial notebook](folder/file_to_link_to.txt):

If you want to test your own image dataset, simply add all the images to a directory. In the notebook, change the **img_dir** to the path of your directory (we recommend using an absolute path to avoid any issues with the notebook execution).

```
img_dir = 'path/to/your/image/directory/'
```

Next, execute the code below to see the system's predictions for single and multiple images from your dataset.

In the **Help us label the data** section, you can view the output results after a user takes a photo using River-Pedia in our system. It includes the system's predicted categories and probabilities, and provides detected pollution locations as references for the user. This allows users to assist us in labeling and categorizing, fully leveraging the advantages of River-Pedia and reducing the hassle of labeling.

```
# Replace image_path with the path of any image you want to predict.

label_helper(model, image_path = '/Assets/GBR201600000000012.jpg' , device = device)
```


## Acknowledgement

The code is largely based on [C-Tran](https://github.com/QData/C-Tran) and [YOLOv8](https://github.com/ultralytics/ultralytics).