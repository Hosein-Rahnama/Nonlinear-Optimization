#include <iostream>

#include<Optimization/LineSearchBackTrack.hpp>
#include <Optimization/BFGS.hpp>
#include <Optimization/SteepestDescent.hpp>


using namespace Optimization;


static double objFunc(const Eigen::VectorXd & parameters)
{
    return 100 * std::pow(parameters(1) - std::pow(parameters(0), 2), 2) + std::pow(1 - parameters(0), 2);
}

static void objFuncInfoExactDerivative(const Eigen::VectorXd & parameters,
                        double & funcValue,
                        Eigen::VectorXd & gradient)
{   
    funcValue = objFunc(parameters); 
    gradient(0) = -400 * (parameters(1) - std::pow(parameters(0), 2.0)) * parameters(0) - 2 * (1 - parameters(0));
    gradient(1) = 200 * (parameters(1) - std::pow(parameters(0), 2.0));
}

class ObjFuncApproxDerivative : public Optimization::ApproxDerivative 
{
    public:
        double objectiveFunction(const Eigen::VectorXd & parameters) override 
        {
            return objFunc(parameters);
        }      
};

int main()
{
    std::shared_ptr<BaseAlgorithm> algorithm;
    Function decoratedObjFuncInfo;
    ObjFuncApproxDerivative objFuncInfoApproxDerivative;
    const Eigen::Vector2d initialParameters(-5, 10);
    Result result;

    // Steepest Descent, Nocedal Line Search, Exact Derivative
    SteepestDescent(objFuncInfoExactDerivative, initialParameters).solve(result);
    std::cout << "---------------- Steepest Descent, Nocedal Line Search, Exact Derivative --------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // BFGS, Nocedal Line Search, Exact Derivative
    BFGS(objFuncInfoExactDerivative, initialParameters).solve(result);
    std::cout << "---------------------- BFGS, Nocedal Line Search, Exact Derivative --------------------------" << std::endl;
    std::cout << result << std::endl << std::endl;
    
    // Steepest Descent, Nocedal Line Search, Approximate Derivative
    SteepestDescent(objFuncInfoApproxDerivative, initialParameters).solve(result);
    std::cout << "-------------- Steepest Descent, Nocedal Line Search, Approximate Derivative ----------------" << std::endl;
    std::cout << result << std::endl << std::endl;
    
    // BFGS, Nocedal Line Search, Approximate Derivative
    BFGS(objFuncInfoApproxDerivative, initialParameters).solve(result);
    std::cout << "-------------------- BFGS, Nocedal Line Search, Approximate Derivative ----------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // Steepest Descent, Backtracking Line Search, Exact Derivative
    algorithm = std::make_shared<SteepestDescent>(objFuncInfoExactDerivative, initialParameters);
    decoratedObjFuncInfo = algorithm->getDecoratedObjFuncInfo();
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(decoratedObjFuncInfo));
    algorithm->solve(result);
    std::cout << "--------------- Steepest Descent, Backtracking Line Search, Exact Derivative ----------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // BFGS, Backtracking Line Search, Exact Derivative
    algorithm = std::make_shared<BFGS>(objFuncInfoExactDerivative, initialParameters);
    decoratedObjFuncInfo = algorithm->getDecoratedObjFuncInfo();
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(decoratedObjFuncInfo));
    algorithm->solve(result);
    std::cout << "--------------------- BFGS, Backtracking Line Search, Exact Derivative ----------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // Steepest Descent, Backtracking Line Search, Approximate Derivative
    algorithm = std::make_shared<SteepestDescent>(objFuncInfoApproxDerivative, initialParameters);
    decoratedObjFuncInfo = algorithm->getDecoratedObjFuncInfo();
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(decoratedObjFuncInfo));
    algorithm->solve(result);
    std::cout << "------------ Steepest Descent, Backtracking Line Search, Approximate Derivative -------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    // BFGS, Backtracking Line Search, Approximate Derivative
    algorithm = std::make_shared<BFGS>(objFuncInfoApproxDerivative, initialParameters);
    decoratedObjFuncInfo = algorithm->getDecoratedObjFuncInfo();
    algorithm->setLineSearch(std::make_shared<LineSearchBackTrack>(decoratedObjFuncInfo));
    algorithm->solve(result);
    std::cout << "------------------ BFGS, Backtracking Line Search, Approximate Derivative -------------------" << std::endl;
    std::cout << result << std::endl << std::endl;

    return 0;
}
