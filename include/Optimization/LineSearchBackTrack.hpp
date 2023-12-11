#pragma once

#include <Eigen/Dense>
#include <Optimization/LineSearch.hpp>


namespace Optimization 
{

class LineSearchBackTrack : public LineSearch 
{
    public:
        LineSearchBackTrack(Function &         objFunc,
                            const double       armijoCoeff = 1e-4,
                            const double       contractionCoeff = 0.5,
                            const unsigned int maxNumIterations = 1000);
        
        ~LineSearchBackTrack();
        
        bool search(const Eigen::VectorXd & lastParameters,
                    const Eigen::VectorXd & lastGradient,
                    const Eigen::VectorXd & direction,
                    Eigen::VectorXd &       parameters,
                    double &                funcValue,
                    Eigen::VectorXd &       gradient,
                    double &                stepLength) override;
        
        void setCoefficients(double armijoCoeff, double contractionCoeff);
        double getArmijoCoeff() const;
        double getContractionCoeff() const;
        
    private:        
        inline void evaluate(double            stepLength,
                             Eigen::VectorXd & parameters,
                             double &          funcValue,
                             Eigen::VectorXd & gradient) const 
        {
            parameters = (*initParameters) + stepLength * (*direction);
            objFunc->calcObjFuncValue(parameters, funcValue);
            objFunc->calcGrad(parameters, gradient);

            return;
        }
        
        inline bool checkArmijo(double stepLength,
                                double funcValue) const 
        {
            // Check the Armijo os sufficient decrease condition.
            return funcValue <= (armijoLineIntercept + stepLength * armijoLineSlope);
        }
        
    private:
        double                  armijoCoeff;
        double                  contractionCoeff;
        
        const Eigen::VectorXd * initParameters;
        const Eigen::VectorXd * direction;
        double                  armijoLineIntercept;
        double                  armijoLineSlope;
        unsigned int            numIterations;        
};

}
