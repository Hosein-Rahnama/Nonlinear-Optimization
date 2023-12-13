#pragma once

#include<iostream>
#include<Eigen/Dense>

namespace Optimization
{

enum ExitFlag 
{
     Gradient,
     Relative,
     LineSearchFailed,
     MaxNumIterations
};

class Result 
{
    public:
        Result() { }

        inline void set(const ExitFlag exitFlag,
                        const Eigen::VectorXd & optParameters,
                        const double optFuncValue,
                        const double optGradNorm,
                        const unsigned int numIterations,
                        const unsigned int numFuncEvaluations,
                        const unsigned int numGradEvaluations)
        {
            this->exitFlag           = exitFlag;
            this->optParameters      = optParameters;
            this->optFuncValue       = optFuncValue;
            this->optGradNorm        = optGradNorm;
            this->numIterations      = numIterations;
            this->numFuncEvaluations = numFuncEvaluations;
            this->numGradEvaluations = numGradEvaluations;
        }

        friend std::ostream & operator<<(std::ostream & out, 
                                         const Result & result);

    private:
        ExitFlag        exitFlag;
        Eigen::VectorXd optParameters;
        double          optFuncValue;
        double          optGradNorm;
        unsigned int    numIterations;
        unsigned int    numFuncEvaluations;
        unsigned int    numGradEvaluations;
};

}