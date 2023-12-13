#include <stdexcept>

#include <Optimization/BaseAlgorithm.hpp>


namespace Optimization 
{

BaseAlgorithm::BaseAlgorithm(Function &              objFunc,
                             const Eigen::VectorXd & initialParameters,
                             double                  gradTol,
                             double                  relTol,
                             unsigned int            maxNumIterations,
                             LineSearch::Ptr         lineSearch)
{
    this->initialParameters = initialParameters;
    numParameters = initialParameters.size();

    setGradientTol(gradTol);
    setRelativeTol(relTol);

    numIterations = 0;
    setMaxNumIterations(maxNumIterations);

    this->objFunc = (&objFunc);

    setLineSearch(lineSearch);
}

BaseAlgorithm::~BaseAlgorithm()
{

}

void BaseAlgorithm::solve(Result & result)
{
    Eigen::VectorXd parameters = initialParameters;
    double funcValue;
    Eigen::VectorXd gradient(numParameters);
    Eigen::VectorXd direction(numParameters);
    double lastGradNorm;

    Eigen::VectorXd lastParameters(numParameters);
    double lastFuncValue;
    Eigen::VectorXd lastGradient(numParameters);
    Eigen::VectorXd lastDirection(numParameters);
    double gradNorm;
    
    // Reset counters of function and gradient evaluations.
    objFunc->resetNumEvaluations();

    // Evaluate the function and its gradient.
    objFunc->calcObjFuncValue(parameters, funcValue);
    objFunc->calcGrad(parameters, gradient);
    
    // Ensure that the initial parameters are not a minimizer.
    gradNorm = computeGradNorm(gradient);
    if (gradNorm <= gradTol)
    {
        result.set(Gradient, parameters, funcValue, gradNorm, numIterations, 
                   objFunc->getNumFuncEvaluations(), objFunc->getNumGradEvaluations());
        return;
    }
    
    // Compute the initial direction.
    initialDirection(gradient, direction);

    double stepLength = 1.0;
    
    while (true)
    {
        ++numIterations;

        lastParameters = parameters;
        lastFuncValue  = funcValue;
        lastGradient   = gradient;
        lastDirection  = direction;
        lastGradNorm   = gradNorm;
        
        // Search for an optimal step length.
        const bool stepLengthFound = lineSearch->search(lastParameters,
                                                        lastGradient,
                                                        direction,
                                                        parameters,
                                                        funcValue,
                                                        gradient,
                                                        stepLength);
        
        if (!stepLengthFound)
        {
            result.set(LineSearchFailed, lastParameters, lastFuncValue, lastGradNorm, numIterations, 
                       objFunc->getNumFuncEvaluations(), objFunc->getNumGradEvaluations());
            return;
        }
        
        // Gradient convergence test.
        gradNorm = computeGradNorm(gradient);
        if (gradNorm <= gradTol)
        {
            result.set(Gradient, parameters, funcValue, gradNorm, numIterations, 
                       objFunc->getNumFuncEvaluations(), objFunc->getNumGradEvaluations());
            return;
        }
        
        // Relative convergence test.
        if (std::fabs(funcValue - lastFuncValue) <= relTol * std::fabs(funcValue))
        {
            result.set(Relative, parameters, funcValue, gradNorm, numIterations, 
                       objFunc->getNumFuncEvaluations(), objFunc->getNumGradEvaluations());
            return;
        }
        
        // Check for maximum number of allowed iterations.
        if (numIterations >= maxNumIterations)
        {
            result.set(MaxNumIterations, parameters, funcValue, gradNorm, numIterations, 
                       objFunc->getNumFuncEvaluations(), objFunc->getNumGradEvaluations());
            return;
        }
        
        // Compute new direction
        updateDirection(parameters, 
                        gradient,
                        lastParameters, 
                        lastGradient,
                        direction);

        // Update trial step length
        stepLength = std::min(1.0, 1.01 * 2 * (funcValue - lastFuncValue) / (lastGradient.dot(lastDirection)));
    }
}

void BaseAlgorithm::setLineSearch(LineSearch::Ptr lineSearch)
{
    if (lineSearch == nullptr)
    {
        this->lineSearch = std::make_shared<LineSearchNocedal>(*objFunc);
    }
    else
    {
        this->lineSearch = lineSearch;
    }
}

LineSearch::Ptr BaseAlgorithm::getLineSearch() const
{
    return lineSearch;
}
 
void BaseAlgorithm::setMaxNumIterations(unsigned int maxNumIterations)
{
    if (maxNumIterations < 1) 
    {
        throw std::invalid_argument("Maximum number of allowed iterations must be greater than zero.");
    }
    
    this->maxNumIterations = maxNumIterations;
}

unsigned int BaseAlgorithm::getMaxNumIterations() const
{
    return maxNumIterations;
}

void BaseAlgorithm::setGradientTol(double gradTol)
{
    if (gradTol < 0.0) 
    {
        throw std::invalid_argument("Gradient tolerance must be greater than or equal to zero.");
    }
    this->gradTol = gradTol;
}

double BaseAlgorithm::getGradientTol() const
{
    return gradTol;
}

void BaseAlgorithm::setRelativeTol(double relTol)
{
    if (relTol < 0.0) 
    {
        throw std::invalid_argument("Relative tolerance must be greater than or equal to zero.");
    }
    this->relTol = relTol;
}

double BaseAlgorithm::getRelativeTol() const
{
    return relTol;
}

}
