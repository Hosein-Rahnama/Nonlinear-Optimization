#include <Optimization/SteepestDescent.hpp>


namespace Optimization 
{

SteepestDescent::SteepestDescent(Function &              objFunc,
                                 const Eigen::VectorXd & initialParameters,
                                 double                  gradTol,
                                 double                  relTol,
                                 unsigned int            maxNumIterations,
                                 LineSearch::Ptr         lineSearch)
                                 :
                                 BaseAlgorithm(objFunc,
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
