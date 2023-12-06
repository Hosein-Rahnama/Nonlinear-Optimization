#include <Optimization/BFGS.hpp>


namespace Optimization 
{

BFGS::BFGS(const Function &        objFuncInfo,
           const Eigen::VectorXd & initialParameters,
           double                  gradTol,
           double                  relTol,
           unsigned int            maxNumIterations,
           LineSearch::Ptr         lineSearch)
           :
           BaseAlgorithm(objFuncInfo,
                         initialParameters,
                         gradTol,
                         relTol,
                         maxNumIterations,
                         lineSearch)
{

}

BFGS::~BFGS()
{

}

void BFGS::initialDirection(const Eigen::VectorXd & gradient,
                            Eigen::VectorXd &       direction)
{
    // Compute initial inverse of Hessian
    inverseHessian = Eigen::MatrixXd::Identity(numParameters, numParameters) / gradient.norm();
    
    // Compute initial direction
    direction = -inverseHessian * gradient;
}

/*   
 *  Implements the BFGS Algorithm 6.1 from
 *  Jorge Nocedal and Stephen J. Wright, Numerical Optimization,
 *  Springer, 2nd edition, 2006, Page 140
 */

void BFGS::updateDirection(const Eigen::VectorXd & parameters,
                           const Eigen::VectorXd & gradient,
                           const Eigen::VectorXd & lastParameters,
                           const Eigen::VectorXd & lastGradient,
                           Eigen::VectorXd &       direction)
{    
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
