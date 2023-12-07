#include <stdexcept>

#include <Optimization/LineSearchBackTrack.hpp>


namespace Optimization 
{

LineSearchBackTrack::LineSearchBackTrack(Function &         decoratedObjFuncInfo,
                                         const double       armijoCoeff,
                                         const double       contractionCoeff,
                                         const unsigned int maxNumIterations)
                                         :
                                         LineSearch(decoratedObjFuncInfo,
                                                    maxNumIterations)
{
    setCoefficients(armijoCoeff, contractionCoeff);
}

LineSearchBackTrack::~LineSearchBackTrack()
{

}

/*   
 *  Implements line search Algorithm 3.1 from
 *  Jorge Nocedal and Stephen J. Wright, Numerical Optimization,
 *  Springer, 2nd edition, 2006, Page 37
 */

bool LineSearchBackTrack::search(const Eigen::VectorXd & initParameters,
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
    this->numIterations       = 0;
    
    while (true) 
    {        
        ++numIterations;

        if (stepLength < DBL_EPSILON) 
        {
            // Current step length is too small.
            return false;
        }

        // Evaluate the function and its gradient values.
        evaluate(stepLength, parameters, funcValue, gradient);
        
        if (checkArmijo(stepLength, funcValue)) 
        {
            return true;
        }
        
        // Decrease step length in exponential fashion.
        stepLength = contractionCoeff * stepLength;

        if (numIterations >= maxNumIterations)
        {
            return false;
        }
    }
}


void LineSearchBackTrack::setCoefficients(double armijoCoeff, double contractionCoeff)
{
    if (armijoCoeff <= 0.0 || armijoCoeff >= 1.0) 
    {
        throw std::invalid_argument("The Armijo coefficient must be in (0, 1).");
    }
    
    if (contractionCoeff <= 0.0 || contractionCoeff >= 1.0) 
    {
        throw std::invalid_argument("The contraction coefficient must be in (0, 1).");
    }
    
    this->armijoCoeff = armijoCoeff;
    this->contractionCoeff  = contractionCoeff;
}

double LineSearchBackTrack::getArmijoCoeff() const
{
    return armijoCoeff;
}

double LineSearchBackTrack::getContractionCoeff() const
{
    return contractionCoeff;
}

}