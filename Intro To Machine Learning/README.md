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

**Example 1**

Imagine you own a snow cone cart, and you have some data about the average number of snow cones sold per day based on the high temperature. You want to better understand this relationship to make sure you have enough inventory on hand for those high sales days.
1.  [  
    1. Lesson Outline](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/0821ddd0-6ff3-4e2f-8f83-1c14e7d0182f "1. Lesson Outline")
2.  [2. What is Machine Learning?](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/4eae70b0-6baf-4a50-bb97-dd5414168cac "2. What is Machine Learning?")
3.  [3. Components of Machine Learning](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/441ece0a-96c2-4a18-8f3e-3ff87f0e54ba "3. Components of Machine Learning")
4.  [4. Quiz: What is Machine Learning?](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/bc834d8f-ebc7-454e-a867-f95c5bdb82cc "4. Quiz: What is Machine Learning?")
5.  [5. Introduction to Machine Learning Steps](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/23a2f4cf-7388-4075-a14e-0d8321143fc2 "5. Introduction to Machine Learning Steps")
6.  [6. Define the Problem](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/5c88a61a-be26-46ed-a2ae-1103bf8300de "6. Define the Problem")
7.  [7. Quiz: Define the Problem](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/842e436d-7d99-400b-a6ab-574aa4d7541e "7. Quiz: Define the Problem")
8.  [8. Build a Dataset](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/8250d300-7f24-4a5e-bf7b-ee8810495191 "8. Build a Dataset")
9.  [9. Quiz: Build a Dataset](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/a9c78469-57ce-4b90-8e32-77bc88125754 "9. Quiz: Build a Dataset ")
10.  [10. Model Training](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/a1e39c48-02c1-45d9-8672-fc0a379b106f "10. Model Training")
11.  [11. Quiz Model Training](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/c18673bf-210a-4c85-8d08-3ad1920207d4 "11. Quiz Model Training ")
12.  [12. Model Evaluation](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/1d30b2aa-3b40-4fb3-b527-9ee40e7305c8 "12. Model Evaluation")
13.  [13. Quiz: Model Evaluation](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/124db6ba-d086-45e0-ab1f-1b53310af448 "13. Quiz: Model Evaluation ")
14.  [14. Model Inference](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/1603bbd9-6823-4797-b98d-e5d5b54c2ae4 "14. Model Inference")
15.  [15. Quiz: Model Inference](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/8beb9192-2239-4b50-842a-c9b32a0d3ec9 "15. Quiz: Model Inference  ")
16.  [16. Introduction to Examples](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/fa38fa4f-058e-49ba-ae5e-d1711f50347f "16. Introduction to Examples")
17.  [17. Example One: House Price Prediction](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/a9e074ab-2264-4bca-8c73-69668cec31dd "17. Example One: House Price Prediction")
18.  [18. Quiz: Example One](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/788304d6-4f55-4681-9078-b7cc874c4513 "18. Quiz: Example One")
19.  [19. Example Two: Book Genre Exploration](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/f068455f-4623-442e-ad52-4c793cf9bc7d "19. Example Two: Book Genre Exploration")
20.  [20. Quiz: Example Two](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/9355e1ee-bf7d-45cf-a623-3027d2790bbd "20. Quiz: Example Two")
21.  [21. Example Three: Spill Detection from Video](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/913a711c-8106-4209-8759-c78522bf4e83 "21. Example Three: Spill Detection from Video")
22.  [22. Quiz: Example Three](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/2eca3899-a422-47c5-bada-d481aea412c8 "22. Quiz: Example Three")
23.  [23. Final Quiz](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/fb2190d2-04a2-49f9-b11f-01a8af557e28 "23. Final Quiz")
24.  [24. Lesson Review](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/419c813d-4829-4fae-80a4-bca0ececdb6c "24. Lesson Review")
25.  [25. Glossary](https://classroom.udacity.com/nanodegrees/nd065/parts/a5a4c41f-9cc7-48bd-9f00-582f35a7da53/modules/885b116b-2ca3-453a-8df1-4ea4b436b5da/lessons/15cbb472-1fc6-44fa-a256-4ade21ee0c7f/concepts/bcc289d9-257e-41b3-88d2-71f82d8b5373 "25. Glossary")

-   Mentor Help
    
    Ask a mentor on our Q&A platform
-   Peer Chat
    
    Chat with peers and alumni
    

# Components of Machine Learning

SEND FEEDBACK

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



