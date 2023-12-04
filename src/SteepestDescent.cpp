#include <Optimization/SteepestDescent.hpp>


namespace Optimization 
{

SteepestDescent::SteepestDescent(const Function &        objFuncInfo,
                                 const Eigen::VectorXd & initialParameters,
                                 double                  gradTol,
                                 double                  relTol,
                                 unsigned int            maxNumIterations,
                                 LineSearch::Ptr         lineSearch)
                                 :
                                 BaseAlgorithm(objFuncInfo,
                                               initialParameters,
                                               gradTol,
                                               relTol,
                                               maxNumIterations,
                                               lineSearch)
{

}

SteepestDescent::~SteepestDescent()
{

}

}
