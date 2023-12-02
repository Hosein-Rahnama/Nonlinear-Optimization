#pragma once

#include <Optimization/QuasiNewton.hpp>


namespace Optimization 
{

class SteepestDescent : public QuasiNewton 
{
    public:
        SteepestDescent();
        
        ~SteepestDescent();
        
    private:
        void initialDirection(const Eigen::VectorXd & gradient,
                              Eigen::VectorXd &       direction) override;
        
        void updateDirection(const Eigen::VectorXd & parameters,
                             const Eigen::VectorXd & gradient,
                             const Eigen::VectorXd & lastParameters,
                             const Eigen::VectorXd & lastGradient,
                             unsigned int            numIterations,
                             Eigen::VectorXd &       direction) override; 
};

}
