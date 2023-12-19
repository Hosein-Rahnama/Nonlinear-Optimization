#include <iostream>

#include <Optimization/LineSearchBackTrack.hpp>
#include <Optimization/BFGS.hpp>
#include <Optimization/SteepestDescent.hpp>


using namespace Optimization;


/* 
 *  See item (35) in Section 3 of the following paper for the definition of Chebyquad function.
 *
 *  Moré, J. J., Garbow, B. S., Hillstrom, K. E. (1981). Testing unconstrained optimization software.
 *  ACM Transactions on Mathematical Software (TOMS), 7(1), 17-41.
 */


const int n = 10;
const int m = n;


void evalChebyshev(const Eigen::VectorXd & parameters, Eigen::MatrixXd & chebyshevValue)
{
    for (int j = 0; j < n; j++)
    {
        chebyshevValue(0, j) = parameters(j);
    }
    for (int j = 0; j < n; j++)
    {
        chebyshevValue(1, j) = 2 * std::pow(parameters(j), 2) - 1;
    }
    for (int i = 2; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            chebyshevValue(i, j) = 2 * parameters(j) * chebyshevValue(i - 1, j) - chebyshevValue(i - 2, j);
        }
    }

    return;
}

void evalShiftedChebyshev(const Eigen::VectorXd & parameters, Eigen::MatrixXd & shiftedChebyshevValue)
{
    Eigen::VectorXd shiftedParameters = 2 * parameters - Eigen::VectorXd::Constant(n, 1);
    evalChebyshev(shiftedParameters, shiftedChebyshevValue);

    return;
}

void objFuncPart(const Eigen::VectorXd & parameters, Eigen::VectorXd & objFuncPartValue)
{
    Eigen::MatrixXd shiftedChebyshev(m, n);
    evalShiftedChebyshev(parameters, shiftedChebyshev);

    double averageChebyshev;

    for (int i = 0; i < m; i++)
    {
        averageChebyshev = 0;
        for (int j = 0; j < n; j++)
        {
            averageChebyshev += shiftedChebyshev(i, j);
        }
        averageChebyshev = (averageChebyshev / n);
        objFuncPartValue(i) = averageChebyshev;
        if ((i + 1) % 2 == 0)
        {
            objFuncPartValue(i) = objFuncPartValue(i) + (1.0 / (std::pow(i + 1, 2)- 1));
        }
    }
    
    return;
}

void evalChebyshevDerivative(const Eigen::VectorXd & parameters, Eigen::MatrixXd & chebyshevDerivative)
{
    Eigen::MatrixXd chebyshevValue(m, n);
    evalChebyshev(parameters, chebyshevValue);

    for (int j = 0; j < n; j++)
    {
        chebyshevDerivative(0, j) = 1;
    }
    for (int j = 0; j < n; j++)
    {
        chebyshevDerivative(1, j) = 4 * parameters(j);
    }
    for (int i = 2; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            chebyshevDerivative(i, j) = 2 * parameters(j) * chebyshevDerivative(i - 1, j) - chebyshevDerivative(i - 2, j) + 2 * chebyshevValue(i - 1, j);
        }
    }

    return;
}

void evalShiftedChebyshevDerivative(const Eigen::VectorXd & parameters, Eigen::MatrixXd & shiftedChebyshevDerivative)
{
    Eigen::VectorXd shiftedParameters = 2 * parameters - Eigen::VectorXd::Constant(n, 1);
    evalChebyshevDerivative(shiftedParameters, shiftedChebyshevDerivative);
    shiftedChebyshevDerivative *= 2;

    return;
}

void gradFuncPart(const Eigen::VectorXd & parameters, Eigen::MatrixXd & gradFuncPartValue)
{
    Eigen::MatrixXd shiftedChebyshevDerivative(m, n);
    evalShiftedChebyshevDerivative(parameters, shiftedChebyshevDerivative);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            gradFuncPartValue(i, j) = 1.0 * shiftedChebyshevDerivative(i, j) / n;
        }
    }

    return;
}

void objFunc(const Eigen::VectorXd & parameters, double & funcValue)
{
    Eigen::VectorXd objFuncPartValue(m);

    objFuncPart(parameters, objFuncPartValue);
    funcValue = 0;
    for (int i = 0; i < m; i++)
    {
        funcValue += std::pow(objFuncPartValue(i), 2);
    }

    return;
}

void gradFunc(const Eigen::VectorXd & parameters, Eigen::VectorXd & gradient)
{   
    Eigen::VectorXd objFuncPartValue(m);
    Eigen::MatrixXd gradFuncPartValue(m, n);

    objFuncPart(parameters, objFuncPartValue);
    gradFuncPart(parameters, gradFuncPartValue);

    gradient = 2 * gradFuncPartValue.transpose() * objFuncPartValue;

    return;
}

int main()
{
    std::shared_ptr<BaseAlgorithm> algorithm;
    Function objFuncInfoExactDerivative(objFunc, gradFunc);
    Function objFuncInfoApproxDerivative(objFunc);
    Eigen::VectorXd initialParameters(n);
    for (int j = 0; j < n; j++)
    {
        initialParameters(j) = (j + 1.0) / (n + 1.0);
    }
    Result result;

    // Steepest Descent, Nocedal Line Search, Exact Derivative
    SteepestDescent(objFuncInfoExactDerivative, initialParameters).solve(result);
    std::cout << "------------- Steepest Descent, Nocedal Line Search, Exact Derivative ------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // BFGS, Nocedal Line Search, Exact Derivative
    BFGS(objFuncInfoExactDerivative, initialParameters).solve(result);
    std::cout << "------------------- BFGS, Nocedal Line Search, Exact Derivative ------------------------" << std::endl;
    std::cout << result << std::endl << std::endl;
    
    // Steepest Descent, Nocedal Line Search, Approximate Derivative
    SteepestDescent(objFuncInfoApproxDerivative, initialParameters).solve(result);
    std::cout << "----------- Steepest Descent, Nocedal Line Search, Approximate Derivative --------------" << std::endl;
    std::cout << result << std::endl << std::endl;
    
    // BFGS, Nocedal Line Search, Approximate Derivative
    BFGS(objFuncInfoApproxDerivative, initialParameters).solve(result);
    std::cout << "----------------- BFGS, Nocedal Line Search, Approximate Derivative --------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // Steepest Descent, Backtracking Line Search, Exact Derivative
    algorithm = std::make_shared<SteepestDescent>(objFuncInfoExactDerivative, initialParameters);
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(objFuncInfoExactDerivative));
    algorithm->solve(result);
    std::cout << "------------ Steepest Descent, Backtracking Line Search, Exact Derivative --------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // BFGS, Backtracking Line Search, Exact Derivative
    algorithm = std::make_shared<BFGS>(objFuncInfoExactDerivative, initialParameters);
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(objFuncInfoExactDerivative));
    algorithm->solve(result);
    std::cout << "------------------ BFGS, Backtracking Line Search, Exact Derivative --------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // Steepest Descent, Backtracking Line Search, Approximate Derivative
    algorithm = std::make_shared<SteepestDescent>(objFuncInfoApproxDerivative, initialParameters);
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(objFuncInfoApproxDerivative));
    algorithm->solve(result);
    std::cout << "--------- Steepest Descent, Backtracking Line Search, Approximate Derivative -----------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // BFGS, Backtracking Line Search, Approximate Derivative
    algorithm = std::make_shared<BFGS>(objFuncInfoApproxDerivative, initialParameters);
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(objFuncInfoApproxDerivative));
    algorithm->solve(result);
    std::cout << "--------------- BFGS, Backtracking Line Search, Approximate Derivative -----------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    return 0;
}
