#pragma once

#include <cfloat>

#include <Eigen/Dense>


namespace Optimization 
{

class Function 
{
    public:
        typedef void (* Value)(const Eigen::VectorXd & parameters, double & objFuncValue);
        typedef void (* Gradient)(const Eigen::VectorXd & parameters, Eigen::VectorXd & gradValue);

    public:
        Function(Value objFunc, Gradient gradFunc = nullptr);
        
        virtual ~Function() { }

        void calcObjFuncValue(const Eigen::VectorXd & parameters,
                              double & objFuncValue);

        inline void calcGrad(const Eigen::VectorXd & parameters,
                             Eigen::VectorXd &       gradValue)
        {
            if (this->gradFunc == nullptr)
            {
                calcApproxGrad(parameters, gradValue);
            }
            else
            {
                calcExactGrad(parameters, gradValue);
            }
        }

        inline unsigned int getNumFuncEvaluations() const
        {
            return numFuncEvaluations;
        }

        inline unsigned int getNumGradEvaluations() const
        {
            return numGradEvaluations;
        }

        inline void resetNumEvaluations()
        {
            numFuncEvaluations = 0;
            numGradEvaluations = 0;
        }

    private:
        void calcExactGrad(const Eigen::VectorXd & parameters,
                           Eigen::VectorXd &       gradValue);

        void calcApproxGrad(const Eigen::VectorXd & parameters,
                            Eigen::VectorXd &       gradValue);
    
    private:
        Value objFunc;
        Gradient gradFunc;
        unsigned int numFuncEvaluations;
        unsigned int numGradEvaluations;
};

}
