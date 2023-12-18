#include <iostream>

#include<Optimization/LineSearchBackTrack.hpp>
#include <Optimization/BFGS.hpp>
#include <Optimization/SteepestDescent.hpp>


using namespace Optimization;

const int n = 10;

void objFunc(const Eigen::VectorXd & parameters, double & funcValue)
{
    funcValue = 0;
    for (int i = 0; i < n; i++)
    {
        funcValue += (i + 1) * std::pow(parameters(i) - i, 2);
    }

    return;
}

void gradFunc(const Eigen::VectorXd & parameters, Eigen::VectorXd & gradient)
{   
    for (int i = 0; i < n; i++)
    {
        gradient(i) = 2 * (i + 1) * (parameters(i) - i);
    }

    return;
}

int main()
{
    std::shared_ptr<BaseAlgorithm> algorithm;
    Function objFuncInfoExactDerivative(objFunc, gradFunc);
    Function objFuncInfoApproxDerivative(objFunc);
    Eigen::VectorXd initialParameters = Eigen::VectorXd::Constant(n, n);
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
