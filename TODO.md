## TODO List

The following is a list for future work.

- [ ] Improve the procedure for setting a line search algorithm associated with a direction algorithm.
- [ ] Improve structure of `LineSearch` class by considering restriction of the objective function to a line as an input. This may require adding a new `FunctionLineRestriction` class. Also, the abstract `LineSearch` class should consider the **bracketing** and **zooming** phases as general templates for finding step lengths.
- [ ] Improve coding style, including `const` variables of constructors, `inline` functions, namings, etc.
- [ ] Create tests using CTest for the library.
