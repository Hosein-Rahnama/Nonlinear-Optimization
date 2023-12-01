#pragma once

#include <Optimization/QuasiNewton.hpp>


namespace Optimization 
{

class BFGS : public QuasiNewton 
{
    public:
        BFGS();
        
        ~BFGS();
        
        void operator()(const Function &        function,
                        const Eigen::VectorXd & initialParameters,
                        Result &                result) override;
        
    private:
        void initialDirection(const Eigen::VectorXd & gradient,
                              Eigen::VectorXd &       direction) override;
        
        void updateDirection(const Eigen::VectorXd & parameters,
                             const Eigen::VectorXd & gradient,
                             const Eigen::VectorXd & lastParameters,
                             const Eigen::VectorXd & lastGradient,
                             unsigned int            numIterations,
                             Eigen::VectorXd &       direction) override;
        
    private:
        unsigned int    numParameters;
        Eigen::MatrixXd inverseHessian;
        Eigen::VectorXd s;
        Eigen::VectorXd y;
        Eigen::MatrixXd ysOuter;
        Eigen::MatrixXd ssOuter;
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;    
};

}
