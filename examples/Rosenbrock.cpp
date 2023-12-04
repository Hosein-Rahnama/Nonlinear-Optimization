#include <iostream>

#include <Optimization/BFGS.hpp>
#include <Optimization/SteepestDescent.hpp>


static double objFunc(const Eigen::VectorXd & parameters)
{
    return 100 * std::pow(parameters(1) - std::pow(parameters(0), 2), 2) + std::pow(1 - parameters(0), 2);
}

static void objFuncInfo(const Eigen::VectorXd & parameters,
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
    ObjFuncApproxDerivative objFuncInfoApproxDerivative;
    const Eigen::Vector2d initialParameters(-5, 10);
    Optimization::Result result;

    // Use Steepest Descent with exact derivative
    Optimization::SteepestDescent(objFuncInfo, initialParameters).solve(result);
    std::cout << "------------ Steepest Descent and Analytic Derivative ----------------" << std::endl;
    std::cout << result << std::endl;

    // Use BFGS with exact derivative
    Optimization::BFGS(objFuncInfo, initialParameters).solve(result);
    std::cout << "------------------ BFGS and Analytic Derivative ----------------------" << std::endl;
    std::cout << result << std::endl;
    
    // Use Steepest Descent with approximate derivative
    Optimization::SteepestDescent(objFuncInfoApproxDerivative, initialParameters).solve(result);
    std::cout << "------------ Steepest Descent and Approximate Derivative -------------" << std::endl;
    std::cout << result << std::endl;
    
    // Use BFGS with approximate derivative
    Optimization::BFGS(objFuncInfoApproxDerivative, initialParameters).solve(result);
    std::cout << "------------------ BFGS and Approximate Derivative -------------------" << std::endl;
    std::cout << result << std::endl;

    return 0;
}
