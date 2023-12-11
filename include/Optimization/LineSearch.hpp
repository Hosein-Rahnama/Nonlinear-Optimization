#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Optimization/Function.hpp>


namespace Optimization 
{

class LineSearch 
{
    public:
        typedef std::shared_ptr<LineSearch> Ptr;
        
    public:
        LineSearch(Function &   objFunc,
                   unsigned int maxNumIterations);
        
        virtual ~LineSearch() { }
        
        virtual bool search(const Eigen::VectorXd & lastParameters,
                            const Eigen::VectorXd & lastGradient,
                            const Eigen::VectorXd & direction,
                            Eigen::VectorXd &       parameters,
                            double &                funcValue,
                            Eigen::VectorXd &       gradient,
                            double &                stepLength) = 0;

        /* 
         *  The maximum number of allowed line search iterations.
         *  The default value is 1,000.
         */

        void setMaxNumIterations(unsigned int numIterations);
        unsigned int getMaxNumIterations() const;

    protected:
        Function *   objFunc;
        unsigned int maxNumIterations;   
};

}
