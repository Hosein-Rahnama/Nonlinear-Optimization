#pragma once

#include <Optimization/BaseAlgorithm.hpp>


namespace Optimization 
{

class SteepestDescent : public BaseAlgorithm 
{
    public:
        SteepestDescent(Function &        objFunc,
                        const Eigen::VectorXd & initialParameters,
                        double                  gradTol = 1e-9,
                        double                  relTol = 1e-9,
                        unsigned int            maxNumIterations = 100000,
                        LineSearch::Ptr         lineSearch = nullptr);
        
        ~SteepestDescent();
        
    private:
        inline void initialDirection(const Eigen::VectorXd & gradient,
                                     Eigen::VectorXd &       direction) override
        {
            direction = -1 * gradient;
        }

        
        inline void updateDirection(const Eigen::VectorXd & parameters,
                                    const Eigen::VectorXd & gradient,
                                    const Eigen::VectorXd & lastParameters,
                                    const Eigen::VectorXd & lastGradient,
                                    Eigen::VectorXd &       direction) override
        {
            direction = -1 * gradient;
        }
};

}
