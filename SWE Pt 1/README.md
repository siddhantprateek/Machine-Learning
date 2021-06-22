## Clean and Modular Code
-   _Production code_: Software running on production servers to handle live users and data of the intended audience. Note that this is different from  _production-quality code_, which describes code that meets expectations for production in reliability, efficiency, and other aspects. Ideally, all code in production meets these expectations, but this is not always the case.
-   _Clean code_: Code that is readable, simple, and concise. Clean production-quality code is crucial for collaboration and maintainability in software development.
-   _Modular_  code: Code that is `logically broken up into functions` and modules. Modular production-quality code that makes your code more `organized`, `efficient`, and `reusable`.
-   _Module_: A file. Modules allow `code to be reused` by `encapsulating` them into files that can be imported into other files.

##  Refactoring Code
-   _Refactoring_: `Restructuring` your code to `improve its internal structure` without `changing its external functionality`. This gives you a chance to `clean` and `modularize` your program after you've got it working.
-   Since it isn't easy to write your best code while you're still trying to just get it working, allocating time to do this is essential to `producing high-quality code`. Despite the initial time and effort required, this really pays off by speeding up your development time in the long run.
-   You become a much stronger programmer when you're constantly looking to improve your code. The more you refactor, the easier it will be to structure and write good code the first time.

### Write Clean Code 
```python
s = [45,69,88,97,87,85]
print(sum(s)/len(s))

s1 = [x ** 0.5 *10 for x in s]
print(sum(s1)/len(s1))
```

A clean production code looks like:
```python
import math
import numpy as np

test_scores = [45,69,88,97,87,85]
print(np.mean(test_scores))

curved_test_scores = [math.sqrt(score) * 10 for score in test_scores]
print(np.mean(curved_test_scores))
```

* _Be descriptive and imply type_
* _Be consistent but clearly differentiate_
* _Avoid abbreviations and single letters_
* _Long names aren't the same as descriptive names_
###  Writing clean code: Nice whitespace
-   Organize your code with consistent indentation: the standard is to use four spaces for each indent. You can make this a default in your text editor.
-   Separate sections with blank lines to keep your code well organized and readable.
-   Try to limit your lines to around 79 characters, which is the guideline given in the PEP 8 style guide. In many good text editors, there is a setting to display a subtle line that indicates where the 79 character limit is.

## Modular Code
### Tip: DRY `(Don't Repeat Yourself)`

Don't repeat yourself! Modularization allows you to reuse parts of your code. Generalize and consolidate repeated code in functions or loops.

### Tip: Abstract out logic to improve readability

`Abstracting` out code into a `function` not only `makes it less repetitive`, but also `improves readability` with `descriptive function names`, Although your code can become more readable when you abstract out logic into functions, it is possible to over-engineer this and have way too many modules, so use your judgement.

### Tip: Minimize the number of entities (functions, classes, modules, etc.)

There are trade-offs to having function calls instead of inline logic. If you have broken up your code into an unnecessary amount of functions and modules, you'll have to jump around everywhere if you want to view the implementation details for something that may be too small to be worth it. Creating more modules doesn't necessarily result in effective modularization.

### Tip: Functions should do one thing

Each function you write should be focused on doing one thing. If a function is doing multiple things, it becomes more difficult to generalize and reuse. Generally, if there's an "and" in your function name, consider refactoring.

### Tip: Arbitrary variable names can be more effective in certain functions

Arbitrary variable names in general functions can actually make the code more readable.

### Tip: Try to use fewer than three arguments per function

Try to use no more than three arguments when possible. This is not a hard rule and there are times when it is more appropriate to use many parameters. But in many cases, it's more effective to use fewer arguments. Remember we are modularizing to simplify our code and make it more efficient. If your function has a lot of parameters, you may want to rethink how you are splitting this up.

## Efficient Code
for efficent code we need to `Reduce the run time` and `Reduce space in memory`.

Knowing how to write code that runs efficiently is another essential skill in software development. Optimizing code to be more efficient can mean making it:

-   Execute faster
-   Take up less space in memory/storage

The project on which you're working determines which of these is more important to optimize for your company or product. When you're performing lots of different transformations on large amounts of data, this can make orders of magnitudes of difference in performance.

# Documentation

-   _Documentation_: Additional text or illustrated information that comes with or is embedded in the code of software.
-   Documentation is helpful for clarifying complex parts of code, making your code easier to navigate, and quickly conveying how and why different components of your program are used.
-   Several types of documentation can be added at different levels of your program:
    -   **Inline comments**  - line level
    -   **Docstrings**  - module and function level
    -   **Project documentation**  - project level


## Docstrings

Docstring, or documentation strings, are valuable pieces of documentation that explain the functionality of any function or module in your code. Ideally, each of your functions should always have a docstring.

Docstrings are surrounded by triple quotes. The first line of the docstring is a brief explanation of the function's purpose.

### One-line docstring

```python
def population_density(population, land_area):
    """Calculate the population density of an area."""
    return population / land_area

```

If you think that the function is complicated enough to warrant a longer description, you can add a more thorough paragraph after the one-line summary.

### Multi-line docstring

```python
def population_density(population, land_area):
    """Calculate the population density of an area.

    Args:
    population: int. The population of the area
    land_area: int or float. This function is unit-agnostic, if you pass in 
    values in terms of square km or square miles the function will 
    return a density in those units.

    Returns:
    population_density: population/land_area. The population density of a 
    particular area.
    """
    return population / land_area

```

The next element of a docstring is an explanation of the function's arguments. Here, you list the arguments, state their purpose, and state what types the arguments should be. Finally, it is common to provide some description of the output of the function. Every piece of the docstring is optional; however, doc strings are a part of good coding practice.

### Resources

-   [PEP 257 - Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
-   [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)