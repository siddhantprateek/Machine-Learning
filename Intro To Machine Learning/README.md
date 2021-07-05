# What is Machine Learning?
**Machine learning (ML)** is a modern software development technique and a type of artificial intelligence (AI) that enables computers to solve problems by using examples of real-world data. It allows computers to automatically learn and improve from experience without being explicitly programmed to do so.

* ML is how computers learn data to discover the patterns and make predictions.

@


**Machine learning** is part of the broader field of artificial intelligence. This field is concerned with the capability of machines to perform activities using human-like intelligence. Within machine learning there are several different kinds of tasks or techniques:
* In **supervised learning**, every training sample from the dataset has a corresponding **`label`** or **`output value`** associated with it. As a result, the algorithm learns to predict labels or output values.(i.e. Predicting saling price of the house)
* In **unsupervised learning**, there are **`no labels`** for the training data. A machine learning algorithm tries to learn the  **`underlying patterns`**  or  **`distributions`** that govern the data.
* In **reinforcement learning**, the algorithm figures out which **actions to take in a situation to maximize a reward** (in the form of a number) on the way to reaching a specific goal. This is a completely different approach than supervised and unsupervised learning.

##  How does machine learning differ from traditional programming-based approaches?
![Traditional programs vs. machine learning programs](https://video.udacity-data.com/topher/2021/April/608c4d18_tradml/tradml.png)
**In traditional problem-solving** with software, a person analyzes a problem and engineers a solution in code to solve that problem. For many real-world problems, this process can be laborious (or even impossible) because a correct solution would need to consider a vast number of edge cases.

Imagine, for example, the challenging task of writing a program that can detect if a cat is present in an image. Solving this in the traditional way would require careful attention to details like varying lighting conditions, different types of cats, and various poses a cat might be in.
> A **model** is generic program, made specific by data us to train it.
>  * A machine learning model is a block of code used to solve different problems.

**In machine learning**, the problem solver abstracts away part of their solution as a flexible component called a  **_model_**, and uses a special program called a  **_model training algorithm_**  to adjust that model to real-world data. The result is a trained model which can be used to predict outcomes that are not part of the data set used to train it.

In a way, machine learning automates some of the statistical reasoning and pattern-matching the problem solver would traditionally do.

The overall goal is to use a  **_model_**  created by a  **_model training algorithm_**  to **generate predictions** or find patterns in data that can be used to solve a problem.
## Understanding Terminology

![Perspectives of machine learning](https://video.udacity-data.com/topher/2021/May/60a294e0_ml/ml.png)

Fields that influence machine learning

Machine learning is a new field created at the intersection of statistics, applied math, and computer science. Because of the rapid and recent growth of machine learning, each of these fields might use slightly different formal definitions of the same terms.

## Terminology

**Machine learning**, or  _ML_, is a modern software development technique that enables computers to solve problems by using examples of real-world data.

In  **supervised learning**, every training sample from the dataset has a corresponding label or output value associated with it. As a result, the algorithm learns to predict labels or output values.

In  **reinforcement learning**, the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal.

In  **unsupervised learning**, there are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data.

# Components of Machine Learning
Nearly all tasks solved with machine learning involve three primary components:

-   A machine learning model
-   A model training algorithm
-   A model inference algorithm

![clay analogy for machine learning](https://video.udacity-data.com/topher/2021/April/608c4d95_clay99/clay99.png)

						Clay analogy for machine learning
## Clay Analogy for Machine Learning

You can understand the relationships between these components by imagining the stages of crafting a teapot from a lump of clay.

1.  First, you start with a block of raw clay. At this stage, the clay can be molded into many different forms and be used to serve many different purposes. You decide to use this lump of clay to make a teapot.
2.  So how do you create this teapot? You inspect and analyze the raw clay and decide how to change it to make it look more like the teapot you have in mind.
3.  Next, you mold the clay to make it look more like the teapot that is your goal.

Congratulations! You've completed your teapot. You've inspected the materials, evaluated how to change them to reach your goal, and made the changes, and the teapot is now ready for your enjoyment.


## What are machine learning models?

A machine learning model, like a piece of clay, can be molded into many different forms and serve many different purposes. A more technical definition would be that a machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided.

**Important**

> A model is an extremely generic program(or block of code), made specific by the data used to train it. It is used to solve different problems.


# Components of Machine Learning

![clay analogy for machine learning](https://video.udacity-data.com/topher/2021/April/608c4d95_clay99/clay99.png)

Clay analogy for machine learning

Nearly all tasks solved with machine learning involve three primary components:

-   A machine learning model
-   A model training algorithm
-   A model inference algorithm

## Clay Analogy for Machine Learning

You can understand the relationships between these components by imagining the stages of crafting a teapot from a lump of clay.

1.  First, you start with a block of raw clay. At this stage, the clay can be molded into many different forms and be used to serve many different purposes. You decide to use this lump of clay to make a teapot.
2.  So how do you create this teapot? You inspect and analyze the raw clay and decide how to change it to make it look more like the teapot you have in mind.
3.  Next, you mold the clay to make it look more like the teapot that is your goal.

Congratulations! You've completed your teapot. You've inspected the materials, evaluated how to change them to reach your goal, and made the changes, and the teapot is now ready for your enjoyment.

----------

## What are machine learning models?

A machine learning model, like a piece of clay, can be molded into many different forms and serve many different purposes. A more technical definition would be that a machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided.

**Important**

> A model is an extremely generic program(or block of code), made specific by the data used to train it. It is used to solve different problems.

  

#### Two simple examples

----------

**Example 1**

Imagine you own a snow cone cart, and you have some data about the average number of snow cones sold per day based on the high temperature. You want to better understand this relationship to make sure you have enough inventory on hand for those high sales days.

![Snow cones based on temperature](https://video.udacity-data.com/topher/2021/April/60871174_snowcone/snowcone.png)

						Snow cones sold regression chart


**Example 2**

Let's look at a different example that uses the same  _linear regression model_, but with different data and to answer completely different questions.

Imagine that you work in higher education and you want to better understand the relationship between the cost of enrollment and the number of students attending college. In this example, our model predicts that as the cost of tuition increases the number of people attending college is likely to decrease.

![average tuition cost regression chart](https://video.udacity-data.com/topher/2021/April/608711fc_tuition1/tuition1.png)

Average tuition regression chart

Using the same linear regression model (indicated by the solid line), you can see that the number of people attending college does go down as the cost increases.

----------

> Both examples showcase that a model is a generic program made specific by the data used to train it.

  

## Model Training

  

### How are model training algorithms used to train a model?

In the preceding section, we talked about two key pieces of information: a model and data. In this section, we show you how those two pieces of information are used to create a trained model. This process is called  _model training_.

  

### Model training algorithms work through an interactive process

Let's revisit our clay teapot analogy. We've gotten our piece of clay, and now we want to make a teapot. Let's look at the algorithm for molding clay and how it resembles a machine learning algorithm:

-   **Think about the changes that need to be made.**  The first thing you would do is inspect the raw clay and think about what changes can be made to make it look more like a teapot. Similarly, a model training algorithm uses the model to process data and then compares the results against some end goal, such as our clay teapot.
-   **Make those changes**. Now, you mold the clay to make it look more like a teapot. Similarly, a model training algorithm gently nudges specific parts of the model in a direction that brings the model closer to achieving the goal.
-   **Repeat.**  By iterating over these steps over and over, you get closer and closer to what you want until you determine that you’re close enough that you can stop.

----------

![clay analogy for machine learning](https://video.udacity-data.com/topher/2021/April/60787487_clay-clay/clay-clay.jpg)  
Think about the changes that need to be made

![Molding clay analoogy for machine learning](https://video.udacity-data.com/topher/2021/April/6078741e_clay-hands/clay-hands.jpg)  
Make those changes

## Model Inference: Using Your Trained Model

Now you have our completed teapot. You inspected the clay, evaluated the changes that needed to be made, and made them, and now the teapot is ready for you to use. Enjoy your tea!

_So what does this mean from a machine learning perspective?_  We are ready to use the model inference algorithm to generate predictions using the trained model. This process is often referred to as  **model inference.**

![completed clay teapot](https://video.udacity-data.com/topher/2021/April/60787242_teapot/teapot.jpg)

## Introduction to the five steps of Machine Learning
![Steps of machine learning](https://video.udacity-data.com/topher/2021/April/608c4397_steps/steps.png)

							Steps of machine learning

_These steps are iterative._ In practice, that means that at each step along the process, you review how the process is going. Are things operating as you expected? If not, go back and revisit your current step or previous steps to try and identify the breakdown.

### Step 1 : Define the Problem
## How do You Start a Machine Learning Task?

-   **_Define a very specific task._**
    -   Think back to the snow cone sales example. Now imagine that you own a frozen treats store and you sell snow cones along with many other products. You wonder, "‘How do I increase sales?" It's a valid question, but it's the  **opposite**  of a very specific task. The following examples demonstrate how a machine learning practitioner might attempt to answer that question.
        -   “Does adding a $1.00 charge for sprinkles on a hot fudge sundae increase the sales of hot fudge sundaes?”
        -   “Does adding a $0.50 charge for organic flavors in your snow cone increase the sales of snow cones?”
-   **_Identify the machine learning task we might use to solve this problem._**
    -   This helps you better understand the data you need for a project.
 
 ## What is a Machine Learning Task?

All model training algorithms, and the models themselves, take data as their input. Their outputs can be very different and are classified into a few different groups based on the  _task_ they are designed to solve. Often, we use the kind of data required to train a model as part of defining a machine learning task.

In this lesson, we will focus on two common machine learning tasks:

-   **Supervised** learning
-   **Unsupervised** learning

  

## Supervised and Unsupervised Learning

The presence or absence of labeling in your data is often used to identify a machine learning task.

![Supervised and unsupervised learning](https://video.udacity-data.com/topher/2021/April/608c4422_mltask/mltask.png)

Machine learning tasks

### Supervised tasks

A task is _supervised_  if you are using labeled data. We use the term  _labeled_  to refer to data that already contains the solutions, called  _labels_.
> For example: Predicting the number of snow cones sold based on the temperatures is an example of supervised learning.

![Labeled data](https://video.udacity-data.com/topher/2021/April/6087143c_snowcones2/snowcones2.png)

Labeled data

In the preceding graph, the data contains both a temperature and the number of snow cones sold. Both components are used to generate the linear regression shown on the graph. Our goal was to predict the number of snow cones sold, and we feed that value into the model. We are providing the model with labeled data and therefore, we are performing a  _supervised machine learning task_.

  

### Unsupervised tasks

A task is considered to be  _unsupervised_ if you are using _unlabeled data_. This means you don't need to provide the model with any kind of label or solution while the model is being trained.

Let's take a look at unlabeled data.

![picture of a tree](https://video.udacity-data.com/topher/2021/April/60750141_tree2/tree2.png)

![picture with tree highlighted](https://video.udacity-data.com/topher/2021/April/6075006a_tree/tree.png)

Image credit:  [Unsplash](https://unsplash.com/photos/VNblNq2sLQ0)

-   Take a look at the preceding picture. Did you notice the tree in the picture? What you just did, when you noticed the object in the picture and identified it as a tree, is called  _labeling the picture_. Unlike you, a computer just sees that image as a matrix of pixels of varying intensity.
-   Since this image does not have the labeling in its original data, it is considered  _unlabeled._

  

**How do we classify tasks when we don't have a label?**

Unsupervised learning involves using data that doesn't have a label. One common task is called  **clustering**. Clustering helps to determine if there are any naturally occurring groupings in the data.


i.e. **Identifying book micro-genres with unsupervised learning**

Imagine that you work for a company that recommends books to readers.

* _The assumption_: You are fairly confident that micro-genres exist, and that there is one called  _Teen Vampire Romance_. Because you don’t know which micro-genres exist, you can't use  **supervised learning**  techniques.

* This is where the  **unsupervised learning**  clustering technique might be able to detect some groupings in the data. The words and phrases used in the book description might provide some guidance on a book's micro-genre.

  

## Further Classifying by using Label Types

![Machine learning tasks](https://video.udacity-data.com/topher/2021/April/608c44b0_snsupersuper/snsupersuper.png)

										Machine learning tasks

Initially, we divided tasks based on the presence or absence of labeled data while training our model. Often, tasks are further defined by the type of label which is present.

In  **supervised** learning, there are two main identifiers you will see in machine learning:

-   A  **categorical** label _has a_ discrete _set of possible values. In a machine learning problem in which you want to identify the type of flower based on a picture, you would train your model using images that have been labeled with the categories of flower you would want to identify. Furthermore, when you work with categorical labels, you often carry out_ classification tasks*, which are part of the supervised learning family.
-   A  **continuous** (regression) label _does not have a discrete set of possible values, which often means you are working with numerical data. In the snow cone sales example, we are trying to predict the_ number* of snow cones sold. Here, our label is a number that could, in theory, be any value.

In unsupervised learning,  **clustering** is just one example. There are many other options, such as deep learning.
## Terminology

-   **Clustering**. Unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
-   A  **_categorical label_**  has a discrete set of possible values, such as "is a cat" and "is not a cat."
-   A  **continuous (regression) label**  does not have a discrete set of possible values, which means possibly an unlimited number of possibilities.
-   **Discrete**: A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week).
-   A  **label**  refers to data that already contains the solution.
-   Using  **unlabeled** data means you don't need to provide the model with any kind of label or solution while the model is being trained.

## Additional Reading

-   The  [AWS Machine Learning blog](https://aws.amazon.com/blogs/machine-learning/)  is a great resource for learning more about projects in machine learning.
-   You can use Amazon SageMaker  [to calculate new stats in Major League Baseball](https://aws.amazon.com/blogs/machine-learning/calculating-new-stats-in-major-league-baseball-with-amazon-sagemaker/).
-   You can also find an article on  [Flagging suspicious healthcare claims with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/flagging-suspicious-healthcare-claims-with-amazon-sagemaker/)  on the AWS Machine Learning blog.
-   What [kinds of questions and problems](https://docs.aws.amazon.com/machine-learning/latest/dg/machine-learning-problems-in-amazon-machine-learning.html)  are good for machine learning?

### Step Two: Build a Dataset
Working with data is perhaps the most overlooked—yet most important—step of the machine learning process. In 2017, an O’Reilly study showed that machine learning practitioners spend 80% of their time working with their data.

## The Four Aspects of Working with Data

![Steps of working with data](https://video.udacity-data.com/topher/2021/April/608c4dfa_datasteps/datasteps.png)

Steps of working with data

You can take an entire class just on working with, understanding, and processing data for machine learning applications. Good, high-quality data is essential for any kind of machine learning project. Let's explore some of the common aspects of working with data.

  

### Data collection

Data collection can be as straightforward as running the appropriate SQL queries or as complicated as building custom web scraper applications to collect data for your project. You might even have to run a model over your data to generate needed labels. Here is the fundamental question:

> Does the data you've collected match the machine learning task and problem you have defined?

  

### Data inspection

The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:

-   Outliers
-   Missing or incomplete values
-   Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model

  

### Summary statistics

Models can assume how your data is structured.

Now that you have some data in hand it is a good best practice to check that your data is in line with the underlying assumptions of your chosen machine learning model.

With many statistical tools, you can calculate things like the mean, inner-quartile range (IQR), and standard deviation. These tools can give you insight into the  _scope_,  _scale_, and  _shape_ of the dataset.

  

### Data visualization

You can use data visualization to see outliers and trends in your data and to help stakeholders understand your data.

Look at the following two graphs. In the first graph, some data seems to have clustered into different groups. In the second graph, some data points might be outliers.

![clusters of data image](https://video.udacity-data.com/topher/2021/April/6075c360_plot/plot.png)  
Some of the data seems to cluster in groups

![clusters of data image with outliers](https://video.udacity-data.com/topher/2021/April/6075c469_plot2/plot2.png)  
Some of the data points seem to be outliers

## Terminology

-   _Impute_ is a common term referring to different statistical tools which can be used to calculate missing values from your dataset.
-   _Outliers_ are data points that are significantly different from others in the same sample.

## Additional reading

-   In machine learning, you use several statistical-based tools to better understand your data. The  `sklearn`  library has many examples and tutorials, such as this example demonstrating  [outlier detection on a real dataset](https://sklearn.org/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py).

### Step Three: Model Training

## Splitting your Dataset

The first step in model training is to randomly split the dataset. This allows you to keep some data hidden during training, so that data can be used to evaluate your model before you put it into production. Specifically, you do this to test against the bias-variance trade-off. If you're interested in learning more, see the  **Further learning and reading**  section.

Splitting your dataset gives you two sets of data:

-   _Training dataset_: The data on which the model will be trained. Most of your data will be here. Many developers estimate about 80%.
-   _Test dataset_: The data withheld from the model during training, which is used to test how well your model will generalize to new data.

  

## Model Training Terminology

> The model training algorithm iteratively updates a model's parameters to minimize some loss function.

Let's define those two terms:

-   _Model parameters_: Model parameters are settings or configurations the training algorithm can update to change how the model behaves. Depending on the context, you’ll also hear other more specific terms used to describe model parameters such as  _weights_  and  _biases_. Weights, which are values that change as the model learns, are more specific to neural networks.
-   _Loss function:_ A loss function is used to codify the model’s distance from this goal. For example, if you were trying to predict a number of snow cone sales based on the day’s weather, you would care about making predictions that are as accurate as possible. So you might define a loss function to be “the average distance between your model’s predicted number of snow cone sales and the correct number.” You can see in the snow cone example this is the difference between the two purple dots.

  

## Putting it All Together

The end-to-end training process is

-   Feed the training data into the model.
-   Compute the loss function on the results.
-   Update the model parameters in a direction that reduces loss.

You continue to cycle through these steps until you reach a predefined stop condition. This might be based on a training time, the number of training cycles, or an even more intelligent or application-aware mechanism.

  

  

## Advice From the Experts

Remember the following advice when training your model.

1.  Practitioners often use machine learning frameworks that already have working implementations of models and model training algorithms. You could implement these from scratch, but you probably won't need to do so unless you’re developing new models or algorithms.
2.  Practitioners use a process called  _model selection_  to determine which model or models to use. The list of established models is constantly growing, and even seasoned machine learning practitioners may try many different types of models while solving a problem with machine learning.
3.  _Hyperparameters_  are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.
4.  Be prepared to iterate.

_Pragmatic problem solving with machine learning is rarely an exact science, and you might have assumptions about your data or problem which turn out to be false. Don’t get discouraged. Instead, foster a habit of trying new things, measuring success, and comparing results across iterations._

  

----------

# Extended Learning

This information hasn't been covered in the above video but is provided for the advanced reader.

### Linear models

One of the most common models covered in introductory coursework, linear models simply describe the relationship between a set of input numbers and a set of output numbers through a linear function (think of  _y = mx + b_  or a line on a  _x_ vs y  chart).

Classification tasks often use a strongly related logistic model, which adds an additional transformation mapping the output of the linear function to the range [0, 1], interpreted as “probability of being in the target class.” Linear models are fast to train and give you a great baseline against which to compare more complex models. A lot of media buzz is given to more complex models, but for most new problems, consider starting with a simple model.

### Tree-based models

Tree-based models are probably the second most common model type covered in introductory coursework. They learn to categorize or regress by building an extremely large structure of nested  _if/else blocks_, splitting the world into different regions at each if/else block. Training determines exactly where these splits happen and what value is assigned at each leaf region.

For example, if you’re trying to determine if a light sensor is in sunlight or shadow, you might train tree of depth 1 with the final learned configuration being something like  _if (sensor_value > 0.698), then return 1; else return 0_;. The tree-based model XGBoost is commonly used as an off-the-shelf implementation for this kind of model and includes enhancements beyond what is discussed here. Try tree-based models to quickly get a baseline before moving on to more complex models.

### Deep learning models

Extremely popular and powerful, deep learning is a modern approach based around a conceptual model of how the human brain functions. The model (also called a  _neural network_) is composed of collections of  _neurons_  (very simple computational units) connected together by  _weights_  (mathematical representations of how much information to allow to flow from one neuron to the next). The process of training involves finding values for each weight.

Various neural network structures have been determined for modeling different kinds of problems or processing different kinds of data.

A short (but not complete!) list of noteworthy examples includes:

-   **FFNN**: The most straightforward way of structuring a neural network, the Feed Forward Neural Network (FFNN) structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the previous layer.
-   **CNN**: Convolutional Neural Networks (CNN) represent nested filters over grid-organized data. They are by far the most commonly used type of model when processing images.
-   **RNN**/**LSTM**: Recurrent Neural Networks (RNN) and the related Long Short-Term Memory (LSTM) model types are structured to effectively represent  _for loops_  in traditional computing, collecting state while iterating over some object. They can be used for processing sequences of data.
-   **Transformer**: A more modern replacement for RNN/LSTMs, the transformer architecture enables training over larger datasets involving sequences of data.

## Machine Learning Using Python Libraries

-   For more classical models (linear, tree-based) as well as a set of common ML-related tools, take a look at  `scikit-learn`. The web documentation for this library is also organized for those getting familiar with space and can be a great place to get familiar with some extremely useful tools and techniques.
-   For deep learning,  `mxnet`,  `tensorflow`, and`pytorch`  are the three most common libraries. For the purposes of the majority of machine learning needs, each of these is feature-paired and equivalent.

## Terminology

**_Hyperparameters_** are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.

A **loss function**  is used to codify the model’s distance from this goal

**Training dataset**: The data on which the model will be trained. Most of your data will be here.

**Test dataset**: The data withheld from the model during training, which is used to test how well your model will generalize to new data.

**Model parameters**  are settings or configurations the training algorithm can update to change how the model behaves.

## Additional reading

-   The Wikipedia entry on the  [bias-variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) trade-off can help you understand more about this common machine learning concept.
-   In this  [AWS Machine Learning blog post](https://aws.amazon.com/blogs/machine-learning/build-a-model-to-predict-the-impact-of-weather-on-urban-air-quality-using-amazon-sagemaker/), you can see how to train a machine-learning algorithm to predict the impact of weather on air quality using Amazon SageMaker.

### Step Four: Model Evaluation

After you have collected your data and trained a model, you can start to evaluate how well your model is performing. The metrics used for evaluation are likely to be very specific to the problem you have defined.  _As you grow in your understanding of machine learning, you will be able to explore a wide variety of metrics that can enable you to evaluate effectively._

## Using Model Accuracy

Model accuracy is a fairly common evaluation metric.  _Accuracy_  is the fraction of predictions a model gets right.

Here's an example:

![flower pedals to determine flower type](https://video.udacity-data.com/topher/2021/April/607512ac_flowers/flowers.png)

Petal length to determine species

Imagine that you built a model to identify a flower as one of two common species based on measurable details like petal length. You want to know how often your model predicts the correct species. This would require you to look at your model's accuracy.

## Extended Learning

This information hasn't been covered in the above video but is provided for the advanced reader.

## Using Log Loss

_Log los_s seeks to calculate how  _uncertain_  your model is about the predictions it is generating. In this context, uncertainty refers to how likely a model thinks the predictions being generated are to be correct.

![log loss](https://video.udacity-data.com/topher/2021/April/60751378_jackets/jackets.png)

For example, let's say you're trying to predict how likely a customer is to buy either a jacket or t-shirt.

Log loss could be used to understand your model's uncertainty about a given prediction. In a single instance, your model could predict with 5% certainty that a customer is going to buy a t-shirt. In another instance, your model could predict with 80% certainty that a customer is going to buy a t-shirt. Log loss enables you to measure how strongly the model believes that its prediction is accurate.

In both cases, the model predicts that a customer will buy a t-shirt, but the model's certainty about that prediction can change.

----------

## Remember: This Process is Iterative

![Iterative steps of machine learning](https://video.udacity-data.com/topher/2021/April/608c4ecc_stepsiter/stepsiter.png)

Iterative steps of machine learning

Every step we have gone through is highly iterative and can be changed or re-scoped during the course of a project. At each step, you might find that you need to go back and reevaluate some assumptions you had in previous steps. Don't worry! This ambiguity is normal.

## Terminology

**Log loss**  seeks to calculate how  _uncertain_  your model is about the predictions it is generating.

**Model Accuracy**  is the fraction of predictions a model gets right.

## Additional reading

The tools used for model evaluation are often tailored to a specific use case, so it's difficult to generalize rules for choosing them. The following articles provide use cases and examples of specific metrics in use.

1.  [This healthcare-based example](https://aws.amazon.com/blogs/machine-learning/create-a-model-for-predicting-orthopedic-pathology-using-amazon-sagemaker/), which automates the prediction of spinal pathology conditions, demonstrates how important it is to avoid false positive and false negative predictions using the tree-based  `xgboost`  model.
2.  The popular  [open-source library  `sklearn`](https://scikit-learn.org/stable/modules/model_evaluation.html)  provides information about common metrics and how to use them.
3.  [This entry from the AWS Machine Learning blog](https://aws.amazon.com/blogs/machine-learning/making-accurate-energy-consumption-predictions-with-amazon-forecast/)  demonstrates the importance of choosing the correct model evaluation metrics for making accurate energy consumption estimates using Amazon Forecast.

### Step Five: Model Inference

Once you have trained your model, have evaluated its effectiveness, and are satisfied with the results, you're ready to generate predictions on real-world problems using unseen data in the field. In machine learning, this process is often called  **inference**.

## Iterative Process

![iteration of the process](https://video.udacity-data.com/topher/2021/April/608c4f0d_itersteps2/itersteps2.png)

Iteration of the entire machine learning process

Even after you deploy your model, you're always monitoring to make sure your model is producing the kinds of results that you expect. Tthere may be times where you reinvestigate the data, modify some of the parameters in your model training algorithm, or even change the model type used for training.