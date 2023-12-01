#pragma once

#include <Optimization/LineSearch.hpp>


namespace Optimization 
{

enum ExitFlag 
{
     Gradient,
     Relative,
     LineSearchFailed,
     MaxNumIterations
};

struct Result 
{
    ExitFlag        exitFlag;
    Eigen::VectorXd optParameters;
    double          optFuncValue;
    double          firstOrderOptimality;
    unsigned int    numIterations;
    unsigned int    numFuncEvaluations;
};

class QuasiNewton 
{
    public:
        QuasiNewton();

        virtual ~QuasiNewton();

        virtual void operator()(const Function &        function,
                                const Eigen::VectorXd & initialParameters,
                                Result &                result);

        void setLineSearch(LineSearch::Ptr lineSearch);

        LineSearch::Ptr getLineSearch() const;

        void setMaxNumIterations(unsigned int maxNumIterations);

        unsigned int getMaxNumIterations() const;

        void setGradientTol(double gradTol);

        double getGradientTol() const;

        void setRelativeTol(double relTol);

        double getRelativeTol() const;
        
    private:
        virtual void initialDirection(const Eigen::VectorXd & gradient,
                                      Eigen::VectorXd &       direction) = 0;

        virtual void updateDirection(const Eigen::VectorXd & parameters,
                                     const Eigen::VectorXd & gradient,
                                     const Eigen::VectorXd & lastParameters,
                                     const Eigen::VectorXd & lastGradient,
                                     unsigned int            numIterations,
                                     Eigen::VectorXd &       direction) = 0;

        void evaluateObjectiveFunction(const Eigen::VectorXd & parameters,
                                       double &                funcValue,
                                       Eigen::VectorXd &       gradient);

        static inline double computeFirstOrderOpt(const Eigen::VectorXd & gradient) 
        {
            return gradient.lpNorm<Eigen::Infinity>();
        }
        
    private:
        LineSearch::Ptr lineSearch;
        unsigned int    maxNumIterations;
        double          gradTol;
        double          relTol;
        
        Function        objFunc;
        Function        evalObjFunc;
        unsigned int    numFuncEvaluations;
        
};

std::ostream & operator<<(std::ostream & os,
                          const Result & result);

}
