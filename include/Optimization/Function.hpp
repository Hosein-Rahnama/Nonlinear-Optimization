#pragma once

#include <functional>
#include <cfloat>

#include <Eigen/Dense>


namespace Optimization 
{

typedef std::function<void(const Eigen::VectorXd & parameters,
                           double &                funcValue,
                           Eigen::VectorXd &       gradient)> Function;

class ApproxDerivative 
{
    public:
        ApproxDerivative() { }
        
        virtual ~ApproxDerivative() { }
        
        void operator()(const Eigen::VectorXd & parameters,
                        double &                funcValue,
                        Eigen::VectorXd &       gradient)
        {
            const Eigen::VectorXd::Index numParameters = parameters.size();
            const double epsilon = std::sqrt(DBL_EPSILON);
            const double invEpsilon = 1.0 / epsilon;
            funcValue = objectiveFunction(parameters);
            
            gradParameters = parameters;
            
            for (Eigen::VectorXd::Index i = 0;  i < numParameters; ++i)
            {
                // Add epsilon to ith parameter.
                gradParameters(i) += epsilon;
                
                const double value = objectiveFunction(gradParameters);
                
                gradient(i) = (value - funcValue) * invEpsilon;
                
                // Restore original parameter.
                gradParameters(i) = parameters(i);
            }
        }
        
    protected:
        virtual double objectiveFunction(const Eigen::VectorXd & parameters) = 0;
        
    private:
        Eigen::VectorXd gradParameters;      
};

}
