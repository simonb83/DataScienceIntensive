# Food Image Recognition Capstone Project


#### Project Context

- Company X is a leading food publisher with a large online and offline presence
- They are responsible for publishing a large number of recipe books in different languages across the world, and they also have a popular online presence with a website containing thousands of recipes across multiple cuisines
- Company X is working on producing a next generation recipe management app for smartphones.
- A key feature that they wish to ship is the ability to for users to be able to identify dishes and the associated recipes simply by using the camera on their device, or by uploading a photo of the dish in question.
- However, Company X does not have computer vision expertise in-house and they need assistance in identifying and training a machine learning algorithm capable of recognising and identifying dishes with a sufficient level of accuracy for this feature to be useful to their users.
- This feature is currently considered “exploratory” by Company X, and as such they are not able to invest significant amounts of money in a large team or dedicated hardware

#### Key Question / Problem Statement

Is it possible to train a machine learner capable of distinguishing between and identifying individual dishes from a photo, while minimising the initial cost of external resources such as computing power?

#### How will the Outcome Impact the Client?

Depending on the outcome of the project and the level of accuracy achieved, Company X will take one of three decisions:
1. Include the feature in the beta version of the app if a sufficient level of accuracy is achieved
2. Not include the feature initially, but increase the amount of investment in future work if the results offer promise for the future
3. Not include the feature and eliminate future funding if results or potential are not satisfactory

Ultimately Company X will heavily base their decision on the recommendations provided

#### Data Sources

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

Key references:
- http://webia.lip6.fr/~cord/Publications_files/CordCooking2015icme.pdf
- http://webia.lip6.fr/~wangxin/

#### Overall Approach

- Perform exploratory analysis on the datasets, e.g.,
    - Number of images, image labels, single vs. multi-class labels etc.
    - Image texture, bands and statistical features etc.
- Image pre-processing work
- Create a baseline classifier using traditional machine learning approaches
    - E.g., Naive Bayes, Random Forest, Support Vector Machine etc.
- Compare to a deep learning approach using transfer learning on an existing network
- Attempt to maximise classifier accuracy by fine-tuning model(s)

#### Deliverables

1. Complete code for project
2. Project Report
    - Detailed description of approach
    - Summary of algorithms used
    - Results of specific algorithms / models
    - Comparison of models
    - Recommendations for next steps

