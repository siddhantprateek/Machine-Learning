<h1 align="center">Classical Machine Learning Algorithms</h1>




## What is `make_blobs`?
The `make_blobs()` function can be used to generate blobs of points with a Gaussian distribution.
 - It is used to generate isotropic Gaussian blobs for clustering.


**i.e.**
```python
from sklearn.datasets import make_blobs 

#                                           The number of centers to 
#           number of samples       generate, or the fixed center locations.
                    👇                    👇
(X, y) = make_blobs(100, n_features=2, centers=4)
                            👆
                            # The number of features for each sampl
print(X.shape) # =>   (10, 2)
y # => array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])

```
### Some Problems 
- [Blob Classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
- [Moon Classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- [Circle Classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
- [Regression Test](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
