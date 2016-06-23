## Food Image Recognition Capstone Project Update


### Project Context

- Company X is a leading food publisher with a large online and offline presence
- They are responsible for publishing a large number of recipe books in different languages across the world, and they also have a popular online presence with a website containing thousands of recipes across multiple cuisines
- Company X is working on producing a next generation recipe management app for smartphones.
- A key feature that they wish to ship is the ability to for users to be able to identify dishes and the associated recipes simply by using the camera on their device, or by uploading a photo of the dish in question.
- However, Company X does not have computer vision expertise in-house and they need assistance in identifying and training a machine learning algorithm capable of recognising and identifying dishes with a sufficient level of accuracy for this feature to be useful to their users.
- This feature is currently considered “exploratory” by Company X, and as such they are not able to invest significant amounts of money in a large team or dedicated hardware


### Key Question / Problem Statement

Is it possible to train a machine learner capable of distinguishing between and identifying individual dishes from a photo, while minimising the initial cost of external resources such as computing power?


### Data Sources

1. Food 101
    - 101 Categories, 1000 images per category (750 training + 250 test)
    - Source: EETHZ Vision Lab (http://www.vision.ee.ethz.ch/datasets_extra/food-101/)
2. Menu-Match Dataset
    - 646 images, 1,386 tagged food items across 41 categories
    - Source: Microsoft Research
(http://research.microsoft.com/en-us/um/redmond/projects/menumatch/data/)
3. UNICT FD889
    - 889 Distinct plates of food
    - Source: University of Catania (http://iplab.dmi.unict.it/UNICT-FD889/)
4. UEC FOOD 100
    - 100 categories of food images (mainly Japanese); Each category contains approx. 100 images
    - Source: http://foodcam.mobi/dataset100.html


### Initial Data Exploration 1

I started with the Menu-Match Dataset from Microsoft Research.

This dataset includes 646 images across 42 different categories, however most images are tagged with multiple categories, including up to four categories for some images.

Additionally, there are recurring themes across categories, for instance the top 5 categories for labels 1, 2,3 and 4 are:

Label 1 | Label 2 | Label 3 | Label 4
--------| --------| --------| --------
Brown Rice | Panang Curry Chicken | Spicy String Beans | Vegetable Spring Roll
Jasmine Rice | Orange Chicken | Panang Curry Chicken | Yellow Curry with Chicken
Cheese Pizza | Pepperoni Pizza | Yellow Curry with Chicken | Stir-fry Beef
Bread Sticks | Potato Bread | Stir-fry Beef | Spinach Red Curry with Tofu
Pineapple Pizza | Jasmine Rice | Vegetable Spring Roll | 

My first approach was to group labels into Super Categories, such as:

- Pizza
- Chicken
- Rice
- Soup
- Lasagne
- Salad
- Bread

However ultimately I decided not to continue to use this particular dataset for a number of reasons:

- Different categories applying to the same image (out of 646 images, only 106 have just one label)
- Unequal numbers of images per category
- Overall quite a small dataset
- The images are not standardized (there are 41 different image sizes)

For more details see: [image_analysis_1.ipynb](../notebooks/image_analysis_1.ipynb) and [image_analysis_2.ipynb](../notebooks/image_analysis_2.ipynb)

### Initial Data Exploration 2

I moved on to using the Food 101 Dataset from the EETHZ Vision Lab.

This is a ~5GB dataset with 101,000 images in total across 101 separate classes (each class having exactly 1000 images).
All images are in jpg format, and have been rescaled to have a maximum side length of 512 pixels.

Given the size of the dataset and the associated challenges with processing such large quanitities of information, I decided to reduce the size of the dataset and decided to prioritize image classes based upon the most popular recipe search terms on Google.

I downloaded from Google AdWords Tools the most popular recipe-related search terms in the USA in 2015, and then compared the search terms to the classes in the dataset; of the top 100 search terms, there was a direct match with 13 classes in the Food 101 dataset.

Going forward, I am focusing on the top 12 matching classes which are:

- Pork chop
- Lasagna
- French toast
- Guacamole
- Apple pie
- Cheesecake
- Hamburger
- Fried rice
- Carrot cake
- Chocolate cake
- Steak
- Pizza


### Exploration of the Top 12 Image Classes

One of the first things I did was to standardize all of the images for the top 12 classes by rotating them to the same orientation and scaling to a common size.

Of the 12,000 images, 60% have size (512, 512, 3) and this is the standard size that I will use going forward.

#### Histograms

The next step was to compare RGB histograms for these classes by:

1. Calculate an average image for each class across pixel values
2. Calculate and plot the histograms for Red, Green and Blue respectively

For the twelve classes these look like:

**Red Histograms**

![Red Histograms](images/interim/red_histograms.png)

**Green Histograms**

![Green Histograms](images/interim/green_histograms.png)

**Blue Histograms**

![Blue Histograms](images/interim/blue_histograms.png)

We can see that there is clear variation between color distributions across the average images for each class.

#### 2 Dimensions

Next I tried plotting the different classes in 2-Dimensions according to the following pseudocode:

- Find the 40 'closest' images to the average for each class
- Reduce images to 50 dimensions using PCA
- Further reduce to 2 dimensions using TSNE

**2-D Representation**

![2-D Plot](images/interim/2D.png)

From the plot we cannot see any clear patterns or groupings for the different classes.


#### Pre-trained Model

Finally I ran a pre-trained model on 10 images from each class. The model comes from Andrej Karpathy (https://github.com/karpathy/neuraltalk).

The results were mixed. In most cases the model was able to identify that the image was of some type of food and the general classification was of the form:

    A plate of *item of food* with a fork / other food item

The classes with the most successful predictions were:

- Cheesecake
- Carrot Cake
- Chocolate Cake
- Pizza

For more details see: [interim_report.ipynb](../notebooks/interim_report.ipynb)

### Histogram classifier

After the initial analysis and exploration, I moved on to classification techniques.

I started with a very basic model based upon the histogram for the average image in each class, using three different distance metrics for comparison:

1. Histogram Intersection
2. L1 Norm
3. Euclidean Distance

Furthermore, I constructed the model histogram three different ways:

- Greyscale only
- RGB
- RGB + Greyscale

The accuracy results were:

Histogram Type| Intersection | L1 Norm | Euclidean 
--------------|:------------:|:-------:|:---------:
Greyscale Only | 10.27% | 10.27% | 9.37%
RGB | 10.50% | 10.50% | 9.93%
RGB + Greyscale | 9.90% | 9.90% | 9.13%

Observations:

1. In general, given 12 classes, we would expect around 8% classification rate simply by picking at random, and so in this case we do only slightly better using the simple histogram classifier.
2. There is no difference between using Intersection and L1 Norm as a distance metric
3. The RGB only histogram gives the best result in this case, and adding more features by including Greyscale data actually decreases accuracy, however the difference is very small, and it is important to note that this is only based on one run of the model
4. Independent of the histogram type, the classifiers typically had low precision across all classes, meaning a high tendency towards false positives
5. The only class for which the classifiers had good recall was Pork Chop, with a score of 0.76 in the RGB + Greyscale model; otherwise, recall was also low across classes

For more details see: [histogram_classifier.ipynb](../notebooks/histogram_classifier.ipynb)

### K Nearest Neigbours

I continued looking at histograms as representations of images, but next moved on to K-nearest neighbors to see if I could improve the classification accuracy.

In this case I used the scikit-learn implementation, with Num Neighbors = 10, trying both uniform and weighted voting, and once again using:

- Greyscale only histogram
- RGB histogram
- RGB + Greyscale histogram

The results in this case were:

Histogram Type| Uniform | Weighted 
--------------|:------------:|:-------:
Greyscale Only | 15.9% | 15.0%
RGB | 19.1% | 18.8%
RGB + Greyscale | 18.0% | 18.6%

Observations:

1. For this model, adding more 'features' in the histogram representations also results in a slight improvement in accuracy.
2. There is very little difference between using Uniform and Weighted voting in the model
3. All models saw an improvement in precision and recall across classes, and there were no particularly strong results for a given class, indicating a more 'even' classifier than the simple histogram model

For more details see: [K_nearest_neighbour.ipynb](../notebooks/K_nearest_neighbour.ipynb)

#### Comparison of Results

**Accuracy rates:**

![classification_rates_1](images/interim/classification_rates_1.png)

**Random Classifier:**

_Overall Metrics & Per-Class Metrics_

Classifier accuracy = 8.3%

![random_metrics](images/interim/random_metrics.png)

_Confusion Matrix_

![random_confusion](images/interim/random_confusion.png)

**RGB Histogram Classifier:**

_Overall Metrics & Per-Class Metrics_

![rgb_hist_metrics](images/interim/rgb_hist_metrics.png)

_Confusion Matrix_

![rgb_hist_confusion](images/interim/rgb_hist_confusion.png)

**K Nearest Neighbors: RGB Histograms as Features**

_Overall Metrics & Per-Class Metrics_

![knn_metrics](images/interim/knn_metrics.png)

_Confusion Matrix_

![knn_confusion](images/interim/knn_confusion.png)

Our **Random Classifier** has low precision and recall on all classes (0.11 or below) and we can see from the confusion matrix that its predictions are fairly evenly spread out across classes (as expected).

The overall accuracy of the **Average RGB Histogram** model is only slightly better than random (10.5% vs. 8.3%), but on the pork-chop class in particular it seems to do quite well on initial inspection with a recall score of 0.57.

However, the precision is low (0.11 for pork_chop), and from the confusion matrix we see that in fact it classifies lots of images from all classes as pork_chop, assigning 1,205 out of 3,000 test images to that class (40%).

Steak and Guacamole are the other two classes where it performs better for recall (0.39 and 0.16 respectively), however once again the precision score is low (0.11 and 0.10).

The **K-Nearest Neighbours** classifier improves on accuracy and is also a more balanced classifier across the board, with improved recall and precision for nearly all classes.

Again the confusion matrix seems better balanced, and the classifier does not seem to be getting stuck on one class in particular.

#### Summary

- From initial analysis it appeared that the RGB histograms were quite different for each of the top classes
- However the best classification rate I was able to achieve simply using the average histogram was just over 10%, and this seems to be more due to an underlying "naive" strategy of predicting a few classes in particular (pork_chop, steak and guacamole).
- Using histograms and K-nearest neighbours I was able to nearly double the classification accuracy to 19.50%, and furthermore this resulted in a more balanced classifier
- However at less than 20% accuracy, it is still not high enough to be of real practical use, and so the next step is to explore more complex models to try and achieve better results

