#include <Optimization/SteepestDescent.hpp>


namespace Optimization 
{

SteepestDescent::SteepestDescent()
{
    
}

SteepestDescent::~SteepestDescent()
{

}

void SteepestDescent::initialDirection(const Eigen::VectorXd & gradient,
                                       Eigen::VectorXd &       direction)
{
    direction = -1 * gradient;
}

void SteepestDescent::updateDirection(const Eigen::VectorXd & parameters,
                                      const Eigen::VectorXd & gradient,
                                      const Eigen::VectorXd & lastParameters,
                                      const Eigen::VectorXd & lastGradient,
                                      unsigned int            numIterations,
                                      Eigen::VectorXd &       direction)
{
    direction = -1 * gradient;
}

}
