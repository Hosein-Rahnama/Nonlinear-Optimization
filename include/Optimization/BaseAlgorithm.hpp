#pragma once

#include <string>

#include <Optimization/LineSearchNocedal.hpp>


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
    double          optGradNorm;
    unsigned int    numIterations;
    unsigned int    numFuncEvaluations;
};

class BaseAlgorithm 
{
    public:
        BaseAlgorithm(const Function &        objFuncInfo,
                      const Eigen::VectorXd & initialParameters,
                      double                  gradTol,
                      double                  relTol,
                      unsigned int            maxNumIterations,
                      LineSearch::Ptr         lineSearch = nullptr);

        virtual ~BaseAlgorithm();

        virtual void solve(Result & result);

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
                                     Eigen::VectorXd &       direction) = 0;

        void evaluateObjFuncInfo(const Eigen::VectorXd & parameters,
                                 double &                funcValue,
                                 Eigen::VectorXd &       gradient);

        static inline double computeGradNorm(const Eigen::VectorXd & gradient) 
        {
            return gradient.lpNorm<Eigen::Infinity>();
        }
        
    protected:
        Eigen::VectorXd        initialParameters;
        Eigen::VectorXd::Index numParameters;

        double                 gradTol;
        double                 relTol;
        unsigned int           numIterations;
        unsigned int           maxNumIterations;
        unsigned int           numFuncEvaluations;
        
        LineSearch::Ptr        lineSearch;

        Function               objFuncInfo;
        Function               decoratedObjFuncInfo;
};

std::ostream & operator<<(std::ostream & os, const Result & result);

}
