#include <iostream>

#include <Optimization/LineSearchBackTrack.hpp>
#include <Optimization/BFGS.hpp>
#include <Optimization/SteepestDescent.hpp>


using namespace Optimization;


/* 
 *  See item (33) in Section 3 of the following paper for the definition of Linear function - rank 1.
 *
 *  Moré, J. J., Garbow, B. S., Hillstrom, K. E. (1981). Testing unconstrained optimization software.
 *  ACM Transactions on Mathematical Software (TOMS), 7(1), 17-41.
 */


const int n = 5;
const int m = 10;

int kroneckerDelta(int i, int j)
{
    if (i == j)
    {
        return 1;
    }
    return 0;
}

void objFuncPart(const Eigen::VectorXd & parameters, Eigen::VectorXd & objFuncPartValue)
{
    double sumParameters = 0;
    for (int j = 0; j < n; j++)
    {
        sumParameters += (j + 1) * parameters(j);
    }
    for (int i = 0; i < m; i++)
    {
        objFuncPartValue(i) = (i + 1) * sumParameters - 1;
    }

    return;
}

void gradFuncPart(const Eigen::VectorXd & parameters, Eigen::MatrixXd & gradFuncPartValue)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            gradFuncPartValue(i, j) = (i + 1) * (j + 1);
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
    Eigen::VectorXd initialParameters = Eigen::VectorXd::Constant(n, 1);
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
