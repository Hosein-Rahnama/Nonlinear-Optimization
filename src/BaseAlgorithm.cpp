#include <stdexcept>

#include <Optimization/BaseAlgorithm.hpp>


namespace Optimization 
{

BaseAlgorithm::BaseAlgorithm(const Function &        objFuncInfo,
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

    numFuncEvaluations = 0;
    this->objFuncInfo = objFuncInfo;
    decoratedObjFuncInfo = std::bind(&BaseAlgorithm::evaluateObjFuncInfo, 
                                     this,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3);
    setLineSearch(lineSearch);
}

BaseAlgorithm::~BaseAlgorithm()
{

}

void BaseAlgorithm::solve(Result & result)
{
    Eigen::VectorXd gradient(numParameters);
    Eigen::VectorXd direction(numParameters);
    Eigen::VectorXd lastParameters(numParameters);
    Eigen::VectorXd lastGradient(numParameters);
    double funcValue;
    double lastFuncValue;
    double gradNorm;
    double lastGradNorm;
    
    Eigen::VectorXd parameters = initialParameters;
    // Evaluate the function and its gradient.
    evaluateObjFuncInfo(parameters, funcValue, gradient);
    
    // Ensure that the initial parameters are not a minimizer.
    gradNorm = computeGradNorm(gradient);
    if (gradNorm <= gradTol)
    {
        result.exitFlag           = Gradient;
        result.optParameters      = parameters;
        result.optFuncValue       = funcValue;
        result.optGradNorm        = gradNorm;
        result.numIterations      = numIterations;
        result.numFuncEvaluations = numFuncEvaluations;
        
        return;
    }
    
    // Compute the initial direction.
    initialDirection(gradient, direction);
    
    while (true)
    {
        ++numIterations;
        
        // Store the current parameter and gradient vectors.
        lastParameters = parameters;
        lastGradient   = gradient;
        lastFuncValue  = funcValue;
        lastGradNorm   = gradNorm;
        
        // Search for an optimal step length. Try a step length of 1.0 first.
        double stepLength = 1.0;
        const bool stepLengthFound = lineSearch->search(lastParameters,
                                                        lastGradient,
                                                        direction,
                                                        parameters,
                                                        funcValue,
                                                        gradient,
                                                        stepLength);
        
        if (!stepLengthFound)
        {
            result.exitFlag           = LineSearchFailed;
            result.optParameters      = lastParameters;
            result.optFuncValue       = lastFuncValue;
            result.optGradNorm        = lastGradNorm;
            result.numIterations      = numIterations;
            result.numFuncEvaluations = numFuncEvaluations;
            
            return;
        }
        
        // Gradient convergence test.
        gradNorm = computeGradNorm(gradient);
        if (gradNorm <= gradTol)
        {
            result.exitFlag           = Gradient;
            result.optParameters      = parameters;
            result.optFuncValue       = funcValue;
            result.optGradNorm        = gradNorm;
            result.numIterations      = numIterations;
            result.numFuncEvaluations = numFuncEvaluations;
            
            return;
        }
        
        // Relative convergence test.
        if (std::fabs(funcValue - lastFuncValue) <= relTol * std::fabs(funcValue))
        {
            result.exitFlag           = Relative;
            result.optParameters      = parameters;
            result.optFuncValue       = funcValue;
            result.optGradNorm        = gradNorm;
            result.numIterations      = numIterations;
            result.numFuncEvaluations = numFuncEvaluations;
            
            return;
        }
        
        // Check for maximum number of allowed iterations.
        if (numIterations >= maxNumIterations)
        {
            result.exitFlag           = MaxNumIterations;
            result.optParameters      = parameters;
            result.optFuncValue       = funcValue;
            result.optGradNorm        = gradNorm;
            result.numIterations      = numIterations;
            result.numFuncEvaluations = numFuncEvaluations;

            return;
        }
        
        // Compute new direction
        updateDirection(parameters, 
                        gradient,
                        lastParameters, 
                        lastGradient,
                        direction);
    }
}

void BaseAlgorithm::setLineSearch(LineSearch::Ptr lineSearch)
{
    if (lineSearch == nullptr)
    {
        this->lineSearch = std::make_shared<LineSearchNocedal>(decoratedObjFuncInfo);
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

void BaseAlgorithm::evaluateObjFuncInfo(const Eigen::VectorXd & parameters,
                                        double &                funcValue,
                                        Eigen::VectorXd &       gradient)
{
    numFuncEvaluations++;
    objFuncInfo(parameters, funcValue, gradient);
}

std::ostream & operator<<(std::ostream & os, const Result & result)
{
    os << "------------------------------------------ Result -------------------------------------------\n";
    os << "Exit flag                                 : ";
    
    if (result.exitFlag == Gradient) 
    {
        os << "Reached gradient tolerance\n";
    } 
    else if (result.exitFlag == Relative) 
    {
        os << "Reached relative tolerance\n";
    } 
    else if (result.exitFlag == MaxNumIterations) 
    {
        os << "Reached maximum number of allowed iterations\n";
    } 
    else if (result.exitFlag == LineSearchFailed) 
    {
        os << "Line search failed\n";
    } 
    else 
    {
        os << "Unknown exit flag\n";
    }
    
    os << "Optimal parameters                        : " << result.optParameters.transpose() << std::endl;
    os << "Function value                            : " << result.optFuncValue << std::endl;
    os << "Gradient norm                             : " << result.optGradNorm << std::endl;
    os << "Number of iterations                      : " << result.numIterations << std::endl;
    os << "Number of function or gradient evaluations: " << result.numFuncEvaluations << std::endl;
    os << "---------------------------------------------------------------------------------------------\n";
    
    return os;
}

}
