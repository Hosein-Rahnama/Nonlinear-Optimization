#include <stdexcept>

#include <Optimization/LineSearchNocedal.hpp>


namespace Optimization 
{

LineSearchNocedal::LineSearchNocedal(Function &         decoratedObjFuncInfo,
                                     const double       armijoCoeff,
                                     const double       wolfeCoeff,
                                     const unsigned int maxNumIterations)
                                     :
                                     LineSearch(decoratedObjFuncInfo,
                                                maxNumIterations)
{
    setCoefficients(armijoCoeff, wolfeCoeff);
}

LineSearchNocedal::~LineSearchNocedal()
{

}

/*   
 *  Implements line search Algorithm 3.5 from
 *  Jorge Nocedal and Stephen J. Wright, Numerical Optimization,
 *  Springer, 2nd edition, 2006, Page 60
 */

bool LineSearchNocedal::search(const Eigen::VectorXd & initParameters,
                               const Eigen::VectorXd & initGradient,
                               const Eigen::VectorXd & direction,
                               Eigen::VectorXd &       parameters,
                               double &                funcValue,
                               Eigen::VectorXd &       gradient,
                               double &                stepLength)
{    
    // Step length has to be positive.
    if (stepLength <= 0) 
    {
        throw std::invalid_argument("Initial step length must be greater than zero.");
    }
    
    const double initGradDotDir = initGradient.dot(direction);
    
    // Ensure that the initial direction is a descent direction.
    if (0 < initGradDotDir) 
    {
        throw std::invalid_argument("Direction is not a descent direction.");
    }

    this->initParameters      = &initParameters;
    this->direction           = &direction;
    this->armijoLineIntercept = funcValue;
    this->armijoLineSlope     = armijoCoeff * initGradDotDir;
    this->strongWolfeRHS      = -wolfeCoeff * initGradDotDir;
    this->numIterations       = 0;
    
    double lastStepLength = 0.0;
    double lastFuncValue  = armijoLineIntercept;
    
    while (true) 
    {        
        // Evaluate the function and its gradient values.
        const double gradDotDir = evaluate(stepLength, parameters, funcValue, gradient);
        
        if (!checkArmijo(stepLength, funcValue) || funcValue >= lastFuncValue) 
        {
            return zoom(lastStepLength, stepLength, lastFuncValue, parameters, funcValue, gradient, stepLength);
        }
        
        if (checkStrongWolfe(gradDotDir)) 
        {
            // Line search was successful.
            return true;
        }
        
        if (gradDotDir >= 0.0) 
        {
            return zoom(stepLength, lastStepLength, funcValue, parameters, funcValue, gradient, stepLength);
        }
        
        lastStepLength = stepLength;
        lastFuncValue  = funcValue;
        
        // Extrapolate step length in exponential fashion.
        stepLength = 2.0 * stepLength;
        
        if (std::isinf(stepLength)) 
        {
            // Reached maximum possible step length.
            return false;
        }
    }
}

/*   
 *  Implements line search zoom Algorithm 3.6 from
 *  Jorge Nocedal and Stephen J. Wright, Numerical Optimization,
 *  Springer, 2nd edition, 2006, Page 61
 */

bool LineSearchNocedal::zoom(double            stepLengthLow,  
                             double            stepLengthHigh,
                             double            funcValueLow,
                             Eigen::VectorXd & parameters,
                             double &          funcValue,
                             Eigen::VectorXd & gradient,
                             double &          stepLength)
{    
    while (true) 
    {
        ++numIterations;
        
        // Length of the bracketed interval is too small. More specifically, it is smaller 
        // than the increment used in forward difference.
        if (std::fabs(stepLengthHigh - stepLengthLow) < DBL_EPSILON) 
        {
            return false;
        }
        
        // Bisect current step length interval.
        stepLength = 0.5 * (stepLengthLow + stepLengthHigh);
        
        // Evaluate the function and gradient values.
        const double gradDotDir = evaluate(stepLength, parameters, funcValue, gradient);
        
        if (!checkArmijo(stepLength, funcValue) || funcValue >= funcValueLow) 
        {
            // Change upper bound.
            stepLengthHigh = stepLength;
        } 
        else 
        {
            if (checkStrongWolfe(gradDotDir)) 
            {
                // Line search was successful.
                return true;
            }
            if (gradDotDir * (stepLengthHigh - stepLengthLow) >= 0) 
            {
                // Change upper bound.
                stepLengthHigh = stepLengthLow;
            }
            // Change lower bound.
            stepLengthLow = stepLength;
            funcValueLow  = funcValue;
        }
        if (numIterations > maxNumIterations) 
        {
            // Reached maximum number of allowed iteration.
            return false;
        }
    }
}

void LineSearchNocedal::setCoefficients(double armijoCoeff, double wolfeCoeff)
{
    if (armijoCoeff <= 0.0 || armijoCoeff >= 1.0) 
    {
        throw std::invalid_argument("The Armijo coefficient must be in (0, 1).");
    }
    
    if (wolfeCoeff <= armijoCoeff || wolfeCoeff >= 1.0) 
    {
        throw std::invalid_argument("The Wolfe coefficient must be in (armijoCoeff, 1).");
    }
    
    this->armijoCoeff = armijoCoeff;
    this->wolfeCoeff  = wolfeCoeff;
}

double LineSearchNocedal::getArmijoCoeff() const
{
    return armijoCoeff;
}

double LineSearchNocedal::getWolfeCoeff() const
{
    return wolfeCoeff;
}

}
