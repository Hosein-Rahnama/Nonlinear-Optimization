#pragma once

#include <Eigen/Dense>
#include <Optimization/LineSearch.hpp>


namespace Optimization 
{

class LineSearchNocedal : public LineSearch 
{
    public:
        LineSearchNocedal(Function &         decoratedObjFuncInfo,
                          const double       armijoCoeff = 1e-4,
                          const double       wolfeCoeff = 0.9,
                          const unsigned int maxNumIterations = 1000);
        
        ~LineSearchNocedal();
        
        bool search(const Eigen::VectorXd & lastParameters,
                    const Eigen::VectorXd & lastGradient,
                    const Eigen::VectorXd & direction,
                    Eigen::VectorXd &       parameters,
                    double &                funcValue,
                    Eigen::VectorXd &       gradient,
                    double &                stepLength) override;
        
        /* 
         *  Set the coefficients for the Armijo and Wolfe conditions.
         *  The armijoCoeff must be in (0, 1). The default value is 1e-4.
         *  The wolfeCoeff must be in (armijoCoeff, 1). The default value is 0.9.
         */
        
        void setCoefficients(double armijoCoeff, double wolfeCoeff);
        double getArmijoCoeff() const;
        double getWolfeCoeff() const;
        
    private:
        bool zoom(double            stepLengthLow,
                  double            stepLengthHigh,
                  double            funcValueLow,
                  Eigen::VectorXd & parameters,
                  double &          funcValue,
                  Eigen::VectorXd & gradient,
                  double &          stepLength);
        
        inline double evaluate(double            stepLength,
                               Eigen::VectorXd & parameters,
                               double &          funcValue,
                               Eigen::VectorXd & gradient) const 
        {
            parameters = (*initParameters) + stepLength * (*direction);
            (*decoratedObjFuncInfo)(parameters, funcValue, gradient);
            const double gradDotDir = gradient.dot(*direction);
            
            return gradDotDir;
        }
        
        inline bool checkArmijo(double stepLength,
                                double funcValue) const 
        {
            // Check the Armijo os sufficient decrease condition.
            return funcValue <= (armijoLineIntercept + stepLength * armijoLineSlope);
        }
        
        inline bool checkStrongWolfe(double gradDotDir) const 
        {
            // Check the Wolfe or curvature condition.
            return std::fabs(gradDotDir) <= strongWolfeRHS;
        }
        
    private:
        double                  armijoCoeff;
        double                  wolfeCoeff;
        
        const Eigen::VectorXd * initParameters;
        const Eigen::VectorXd * direction;
        double                  armijoLineIntercept;
        double                  armijoLineSlope;
        double                  strongWolfeRHS;
        unsigned int            numIterations;        
};

}
