#include <Optimization/Function.hpp>

namespace Optimization
{

Function::Function(Value objFunc, Gradient gradFunc)
{
    this->objFunc = objFunc;
    this->gradFunc = gradFunc;
    numFuncEvaluations = 0;
    numGradEvaluations = 0;
}

void Function::calcObjFuncValue(const Eigen::VectorXd & parameters,
                                double &                objFuncValue)
{
    numFuncEvaluations++;
    objFunc(parameters, objFuncValue);
}

void Function::calcExactGrad(const Eigen::VectorXd & parameters,
                             Eigen::VectorXd & gradValue)
{
    numGradEvaluations++;
    gradFunc(parameters, gradValue);
}

void Function::calcApproxGrad(const Eigen::VectorXd & parameters,
                              Eigen::VectorXd &       gradValue)
{
    const Eigen::VectorXd::Index numParameters = parameters.size();
    const double epsilon = std::sqrt(DBL_EPSILON);
    const double invEpsilon = 1.0 / epsilon;

    Eigen::VectorXd gradParameters = parameters;
    
    double funcValue;
    double forwardFuncValue;
    calcObjFuncValue(parameters, funcValue);
    
    for (Eigen::VectorXd::Index i = 0;  i < numParameters; ++i)
    {
        // Compute gradient with forward difference.
        gradParameters(i) += epsilon;
        calcObjFuncValue(gradParameters, forwardFuncValue);
        gradValue(i) = (forwardFuncValue - funcValue) * invEpsilon;
        
        // Restore original parameter.
        gradParameters(i) = parameters(i);
    }
}

}