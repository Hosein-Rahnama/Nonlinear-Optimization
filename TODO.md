# TODO List

The following is a list for future work.


## Algorithms
- [ ] Consider interpolations for making an educated guess about the step length in the Nocedal line search algorithm.
- [ ] Improve safegaruds for the line search algorithms including devision by zeros, overflows or underflows.
- [ ] Think about relative tolerance for convergence.


## Code Structure
- [ ] Improve the procedure for setting a line search algorithm associated with a direction algorithm.
- [ ] Improve structure of `LineSearch` class by considering restriction of the objective function to a line as an input. This may require adding a new `FunctionLineRestriction` class. Also, the abstract `LineSearch` class should consider the **bracketing** and **zooming** phases as general templates for finding step lengths.
- [ ] Improve coding style, including `const` variables of constructors, `inline` and `const` methods, namings, etc.
- [ ] Improve `Result` class. Add `setTitle` method for the `Result` class.
- [ ] Think about abstraction of convergence criteria.


## Tests
- [ ] Create tests using CTest for the library. 
- [ ] Use `CUTE` library test functions.


## Visualization
- [ ] Add plotting options for the sequence of iterates in 2D and 3D examples.