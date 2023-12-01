#pragma once

#include <memory>

#include <Optimization/Function.hpp>
#include <Eigen/Dense>


namespace Optimization 
{

class LineSearch 
{
    public:
        typedef std::shared_ptr<LineSearch> Ptr;
        
    public:
        LineSearch() { }
        
        virtual ~LineSearch() { }
        
        virtual bool search(Function &              function,
                            const Eigen::VectorXd & lastParameters,
                            const Eigen::VectorXd & lastGradient,
                            const Eigen::VectorXd & direction,
                            Eigen::VectorXd &       parameters,
                            double &                funcValue,
                            Eigen::VectorXd &       gradient,
                            double &                stepLength) = 0;       
};

class LineSearchNocedal : public LineSearch 
{
    public:
        LineSearchNocedal();
        
        ~LineSearchNocedal();
        
        bool search(Function &               function,
                    const Eigen::VectorXd &  lastParameters,
                    const Eigen::VectorXd &  lastGradient,
                    const Eigen::VectorXd &  direction,
                    Eigen::VectorXd &        parameters,
                    double &                 funcValue,
                    Eigen::VectorXd &        gradient,
                    double &                 stepLength) override;
        
        /* 
         *  The maximum number of allowed line search iterations.
         *  The default value is 1,000.
         */
        void setMaxNumIterations(unsigned int numIterations);
        
        unsigned int getMaxNumIterations() const;
        
        /* 
         *  Set the coefficients for the Armijo and Wolfe conditions.
         *  The armijoCoeff must be in (0, 1).
         *  The default value is 1e-4.
         *  The wolfeCoeff must be in (armijoCoeff, 1).
         *  The default value is 0.9.
         */
        void setCoefficients(double armijoCoeff,
                             double wolfeCoeff);
        
        double getArmijoCoeff() const;
        
        double getWolfeCoeff() const;
        
    private:
        bool zoom(double            stepLengthLo,
                  double            funcValueLo,
                  double            stepLengthHi,
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
            
            (*function)(parameters, funcValue, gradient);
            
            const double gradDotDir = gradient.dot(*direction);
            
            return gradDotDir;
        }
        
        inline bool checkArmijo(double stepLength,
                                double funcValue) const 
        {
            // Check the Armijo condition (sufficient decrease condition).
            return funcValue <= (initFuncValue + stepLength * testArmijo);
        }
        
        inline bool checkWolfe(double gradDotDir) const 
        {
            // Check the Wolfe condition (curvature condition).
            return gradDotDir >= testWolfe;
        }
        
    private:
        unsigned int            maxNumIterations;
        double                  armijoCoeff;
        double                  wolfeCoeff;
        
        Function *              function;
        const Eigen::VectorXd * initParameters;
        const Eigen::VectorXd * direction;
        double                  initFuncValue;
        unsigned int            numIterations;
        double                  testArmijo;
        double                  testWolfe;
        
};

}
