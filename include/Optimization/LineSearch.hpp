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
        LineSearch() { }
        
        virtual ~LineSearch() { }
        
        virtual bool search(Function &              decoratedObjFuncInfo,
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
        
        bool search(Function &               decoratedObjFuncInfo,
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
            (*decoratedObjFuncInfo)(parameters, funcValue, gradient);
            const double gradDotDir = gradient.dot(*direction);
            
            return gradDotDir;
        }
        
        inline bool checkArmijo(double stepLength,
                                double funcValue) const 
        {
            // Check the Armijo os sufficient decrease condition.
            return funcValue <= (initFuncValue + stepLength * testArmijo);
        }
        
        inline bool checkWolfe(double gradDotDir) const 
        {
            // Check the Wolfe or curvature condition.
            return gradDotDir >= testWolfe;
        }
        
    private:
        double                  armijoCoeff;
        double                  wolfeCoeff;
        unsigned int            maxNumIterations;
        
        Function *              decoratedObjFuncInfo;
        const Eigen::VectorXd * initParameters;
        const Eigen::VectorXd * direction;
        double                  initFuncValue;
        unsigned int            numIterations;
        double                  testArmijo;
        double                  testWolfe;        
};

}
