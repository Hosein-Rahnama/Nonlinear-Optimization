#include <iostream>
#include <Optimization/BFGS.hpp>

static double objFunc(const Eigen::VectorXd & parameters)
{
    return 100 * std::pow(parameters(1) - std::pow(parameters(0), 2), 2) + std::pow(1 - parameters(0), 2);
}

static void objFuncAnalyticDerivative(const Eigen::VectorXd & parameters,
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
    try 
    {
        const Eigen::Vector2d initialParameters(-5, 10);
        ObjFuncApproxDerivative objFuncApproxDerivative;
        Optimization::BFGS bfgs;
        Optimization::Result result;
        
        // Using anyltic derivative & BFGS
        bfgs(objFuncAnalyticDerivative, initialParameters, result);

        std::cout << "---------------- BFGS and Analytic Derivative ---------------------" << std::endl;
        std::cout << result << std::endl;
        std::cout << "Optimal parameters: " << result.optParameters.transpose() << std::endl << std::endl;
        
        // Using approximative derivative & BFGS
        bfgs(objFuncApproxDerivative, initialParameters, result);
        
        std::cout << "---------------- BFGS and Approximate Derivative ------------------" << std::endl;
        std::cout << result << std::endl;
        std::cout << "Optimal parameters: " << result.optParameters.transpose() << std::endl << std::endl;
    } 
    catch (std::exception & ex) 
    {
        std::cout << "Error: " << ex.what() << std::endl;
        
        return 1;
    }
    return 0;
}
