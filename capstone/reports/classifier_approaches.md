### Different Classification Attempts

#### Basic Attempts Using Comparison to 'average' images for each class

1. **Greyscale histogram**

- **Features:** Greyscale histogram for complete image
- **Algorithm:** Nearest-1 Neighbor where training data is the average image for each class (by pixel values); tested using three different distance metrics (Intersection, L-1 Norm and Euclidean)
- **Classifier accuracy:** ~10%

2. **RGB Histogram**

- **Features:** RGB histogram for complete image
- **Algorithm:** Nearest-1 Neighbor where training data is the average image for each class (by pixel values); tested using three different distance metrics (Intersection, L-1 Norm and Euclidean)
- **Classifier accuracy:** ~10%

3. **Chained RGB + Greyscale Histogram**

- **Features:** Chained RGB and Greyscale histogram for complete image
- **Algorithm:** Nearest-1 Neighbor where training data is the average image for each class (by pixel values); tested using three different distance metrics (Intersection, L-1 Norm and Euclidean)
- **Classifier accuracy:** ~10%

#### Looking at 10-Nearest Neighbours for similarly basic features

After initial exploration using average images for each class, I expanded on the same overall approach by using individual images in the training data and k-Nearest Neighbors.
In the best case this led to an almost doubling of classifier accuracy.

4. **Greyscale histogram**

- **Features:** Greyscale histogram for complete image
- **Algorithm:** k-Nearest Neighbours, K=10, Uniform and Weighted voting
- **Classifier accuracy:** ~15%

5. **RGB histogram**

- **Features:** RGB histogram for complete image
- **Algorithm:** k-Nearest Neighbours, K=10, Uniform and Weighted voting
- **Classifier accuracy:** ~19%

6. **RGB and Greyscale histogram for complete image**

- **Features:** RGB and Greyscale histogram for complete image
- **Algorithm:** k-Nearest Neighbours, K=10, Uniform and Weighted voting
- **Classifier accuracy:** ~19%

#### Sticking with one set of features and comparing different algorithms

Sticking with the RGB Histogram as features, I tried comparing a number of different algorithms on the same training / testing data.

Overall the best result came from a Random Forest with 25% accuracy and F1 score of 0.241, followed by Polynomial SVC with accuracy 22% and F1 0.222.
The Random Forest was notably faster than the SVC with 7s training time compared to 2mins 4s for the SVC.

7. **Features:** RGB Histogram

- **Algorithm:** kNN, K=10, Uniform and Weighted Voting
- **Classifier accuracy:** ~19%
- **Training time:** 500ms
- **Test time:** 31s
- **F1:** 0.177


- **Algorithm:** Linear SVM
- **Classifier accuracy:** ~14%
- **Training time:** 3 mins 40s
- **Test time:** 12ms
- **F1:** 0.1


- **Algorithm:** SVC, polynomial kernel
- **Classifier accuracy:** ~22%
- **Training time:** 2 mins 4s
- **Test time:** 21s
- **F1:** 0.222


- **Algorithm:** Decision Tree
- **Classifier accuracy:** ~19%
- **Training time:** 3s
- **Test time:** 6ms
- **F1:** 0.166


- **Algorithm:** Random Forest
- **Classifier accuracy:** ~25%
- **Training time:** 7s
- **Test time:** 60ms
- **F1:** 0.241


- **Algorithm:** ADA Boost
- **Classifier accuracy:** ~21%
- **Training time:** 42s
- **Test time:** 144ms
- **F1:** 0.205


- **Algorithm:** Gaussian NB
- **Classifier accuracy:** ~19%
- **Training time:** 120ms
- **Test time:** 340ms
- **F1:** 0.177


- **Algorithm:** Multinomial Naive Bayes 
- **Classifier accuracy:** ~19%
- **Training time:** 50ms
- **Test time:** 15ms
- **F1:** 0.175


- **Algorithm:** Benoulli Naive Bayes 
- **Classifier accuracy:** ~10%
- **Training time:** 100ms
- **Test time:** 34ms
- **F1:** 0.067


- **Algorithm:** Linear Discriminant Analysis 
- **Classifier accuracy:** ~20%
- **Training time:** 1s
- **Test time:** 14ms
- **F1:** 0.196


- **Algorithm:** Quadratic Discriminant Analysis 
- **Classifier accuracy:** ~9%
- **Training time:** 2s
- **Test time:** 840ms
- **F1:** 0.056

#### Feature extraction using different approaches

I decided to focus on Random Forest as a classifier due to its better accuracy in my comparison, and also due to its significantly faster training and testing time, and instead focus on generating different types of features.

For benchmarking I also included results of K-Nearest Neighbors for different feature sets. By this stage, a combination of RGB Histogram values chained with individual pixels gave the best result: accuracy 27%, F1 0.26.

8. **K means clustering I**

- **Features:** Extracted using k-means clustering on RGB images individual pixels, k = 100
- **Algorithm:** kNN, Uniform weighting
- **Classifier accuracy:** ~9%
- **F1:** 0.084

9. **K means clustering II**

- **Features:** Extracted using k-means clustering on RGB images individual pixels, k = 100
- **Algorithm:** RandomForest, max_depth = 5, n_estimators = 15
- **Classifier accuracy:** ~11%
- **F1:** 0.093

10. **PCA I**

- **Features:** Extracted using PCA on RGB images individual pixels, n_features = 100
- **Algorithm:** KNN, Uniform weighting, k=9
- **Classifier accuracy:** ~6%
- **F1:** 0.051

11. **PCA II**

- **Features:** Extracted using PCA on RGB images individual pixels, n_features = 100
- **Algorithm:** Random Forest
- **Classifier accuracy:** ~7%
- **F1:** 0.067

12. **Pixels I**

- **Features:** Images rescaled to 32x32, with individual pizels as features.
- **Algorithm:** KNN, Uniform weighting, k=9
- **Classifier accuracy:** ~14%
- **F1:** 0.133

13. **Pixels II**

- **Features:** Images rescaled to 32x32, with individual pizels as features.
- **Algorithm:** RandomForest, max_depth=7, n_estimators=14
- **Classifier accuracy:** ~21%
- **F1:** 0.192

14. **Histogram + Pixels I**

- **Features:** RGB Histogram for complete image chained with individual pixels of 32x32 image.
- **Algorithm:** KNN, Uniform weighting, k=9
- **Classifier accuracy:** ~18%
- **F1:** 0.171

15. **Histogram + Pixels II**

- **Features:** RGB Histogram for complete image chained with individual pixels of 32x32 image.
- **Algorithm:** RandomForest, max_depth=10, n_estimators=70
- **Classifier accuracy:** ~27%
- **F1:** 0.26

16. **Histogram, Edges and Corners**

The next step was to try and include some more 'sophisticated' featues such as edges and corners, resulting in a nearly 2% improvement in classification accuracy.

- **Features:** RGB Histogram + Canny Edges mask + Corner Fast coordinates; Zero-variance features removed using VarianceThreshold; PCA to reduce each feature set to 100 features, leaving 300 in total
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=8, n_estimators=500
- **Classifier accuracy:** ~29%
- **F1:** 0.266

#### Sliding Window & Segment-based approaches

My next attempt to further improve accuracy was based on splitting each image into smaller parts and using these for generating features and classification.

This has resulted in the best result so far of accuracy of 33% and F1 0.319.

17. **Non-overlapping sliding window 32x32**

- **Features:** Image split into non-overlapping squares of side 32; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners.
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=14, n_estimators=500
- **Classifier accuracy:** ~33%
- **F1:** 0.319

18. **Non-overlapping sliding window 16x16**

Based upon the previous attempt, it seemed that corners were the least important feature, and so I tried eliminating these as features, and reducing the box size to get more granularity, however this resulted in a worse result than before.

- **Features:** Image split into non-overlapping squares of side 16; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners.
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=10, n_estimators=500
- **Classifier accuracy:** ~25%
- **F1:** 0.23

19. **Non-overlapping sliding window 32x32 + PCA**

- **Features:** Image split into non-overlapping squares of side 32; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners. Finally PCA to reduce each feature set to 30 features
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=14, n_estimators=500
- **Classifier accuracy:** ~28%
- **F1:** 0.268

20. **Non-overlapping sliding window 32x32 + PCA**

Similar to the approach in 90 but using PCA to reduce to 50 instead of 30 features.

- **Features:** Image split into non-overlapping squares of side 32; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners. Initial Random Forest classifier, and then top 50 features selected for each type based on importance in classifier.
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=14, n_estimators=500
- **Classifier accuracy:** ~18%
- **F1:** 0.131

21. **Non-overlapping sliding window 32x32, top features from initial Random Forest**

- **Features:** Image split into non-overlapping squares of side 32; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners. Initial Random Forest classifier, and then top 50 features selected for each type based on importance in classifier.
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=14, n_estimators=500
- **Classifier accuracy:** ~28%
- **F1:** 0.268

22. **Non-overlapping sliding window 32x32 + 16x16**

- **Features:** Image split into non-overlapping squares of side 32 and also of side 16; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners and then all features chained together for 32x32 and 16x16 squares.
- **Algorithm:** GridSearch for parameters; RandomForest, max_depth=14, n_estimators=500
- **Classifier accuracy:** ~33%
- **F1:** 0.313

23. **Classifier Cascade I**

- **Features:** Image split into non-overlapping squares of side 32; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners
- **Algorithm:** Random Forests trained as classifiers on each type of feature; Bayesian Net trained on predicted probabilities from each separate classifier.
- **Classifier accuracy:** ~29%
- **F1:** 0.286

- **Algorithm:** Random Forests trained as classifiers on each type of feature; Linear SVM trained on predicted probabilities from each separate classifier.
- **Classifier accuracy:** ~28%
- **F1:** 0.272

24. **Classifier Cascade II**

- **Features:** Image split into non-overlapping squares of side 8; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners

-> Left overnight but was still obtaining features in the morning

25. **Classifier Cascade III**

- **Features:** Image split into non-overlapping squares of side 16; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners
- **Algorithm:** Random Forests trained as classifiers on each type of feature; Bayesian Net trained on predicted probabilities from each separate classifier.
- **Classifier accuracy:** ~29%
- **F1:** 0.285

- **Algorithm:** Random Forests trained as classifiers on each type of feature; Linear SVM trained on predicted probabilities from each separate classifier.
- **Classifier accuracy:** ~28%
- **F1:** 0.258

25. **Segments as Features**

This approach is similar to using the sliding window as above, however instead of working with boxes of a fixed size, using segments identified for each image with the skimage SLIC algorithm (uses k-means clustering in Color-(x, y, z) space).

- **Features:** Splitting training images into segments and for each segment using Average, Max, Min, Range of Color values for RGB, + normed histograms for RGB
- **Algorithm:** Random forest trained on features for each segment of each training image; for test images, predict probabilities for each segment and choose class with highest average probability
- **Classifier accuracy:** ~22%
- **F1:** 0.211

26. **Non-overlapping sliding window 32x32 + LinearSVM**

- **Features:** Image split into non-overlapping squares of side 32; For each square calculate: Average Red pixel value, average Green pixel value, Average Blue pixel value, Number of edges, Number of corners
 
- **Algorithm:** GridSearch for parameters; LinearSVM, C = 0.001
- **Classifier accuracy:** ~30%
- **F1:** 0.284

27. **Linear Classifier + Stochastic Gradient Descent + SVM Loss**

- **Features:** Images of size 64x64, individual pixels as features, normalized by subtracting mean
 
- **Algorithm:** Optimization of SGD with SVM Loss; Learning Rate = 0.0001, Weight Regularization 0.1, 60,000 iterations
- **Classifier accuracy:** ~23%
- **F1:** 0.232
