#include <Optimization/BFGS.hpp>


namespace Optimization 
{

BFGS::BFGS()
{
    
}

BFGS::~BFGS()
{

}

void BFGS::operator()(const Function &        function,
                      const Eigen::VectorXd & initialParameters,
                      Result &                result)
{
    numParameters = initialParameters.size();
    
    // Call actual quasi Newton optimization
    QuasiNewton::operator()(function, initialParameters, result);
}

void BFGS::initialDirection(const Eigen::VectorXd & gradient,
                            Eigen::VectorXd &       direction)
{
    // Compute initial inverse of Hessian
    inverseHessian = Eigen::MatrixXd::Identity(numParameters, numParameters) / gradient.norm();
    
    // Compute initial direction
    direction = -inverseHessian * gradient;
}

void BFGS::updateDirection(const Eigen::VectorXd & parameters,
                           const Eigen::VectorXd & gradient,
                           const Eigen::VectorXd & lastParameters,
                           const Eigen::VectorXd & lastGradient,
                           unsigned int            numIterations,
                           Eigen::VectorXd &       direction)
{
    /*   
     *   Implements the BFGS algorithm from
     *   Jorge Nocedal and Stephen J. Wright, Numerical Optimization,
     *   Springer, 2nd edition, Springer, 2006, Page 177
     */
    
    // Update approximative inverse Hessian
    s = parameters - lastParameters;
    y = gradient - lastGradient;
    
    double ysInner = y.dot(s);
    
    ysOuter = y * s.transpose();
    ssOuter = s * s.transpose();
    
    A = Eigen::MatrixXd::Identity(numParameters, numParameters) - ysOuter / ysInner;
    B.noalias() = A.transpose() * inverseHessian * A;
    
    inverseHessian = B + ssOuter / ysInner;
    
    // Compute new direction
    direction = -inverseHessian * gradient;
}

}
