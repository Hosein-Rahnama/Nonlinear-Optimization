#pragma once

#include <Optimization/BaseAlgorithm.hpp>


namespace Optimization 
{

class BFGS : public BaseAlgorithm 
{
    public:
        BFGS(const Function &        objFuncInfo,
             const Eigen::VectorXd & initialParameters,
             double                  gradTol = 1e-9,
             double                  relTol = 1e-9,
             unsigned int            maxNumIterations = 100000,
             LineSearch::Ptr         lineSearch = std::make_shared<LineSearchNocedal>());
        
        ~BFGS();
        
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
        Eigen::MatrixXd inverseHessian;
        Eigen::VectorXd s;
        Eigen::VectorXd y;
        Eigen::MatrixXd ysOuter;
        Eigen::MatrixXd ssOuter;
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;    
};

}
