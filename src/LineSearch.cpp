#include <stdexcept>

#include <Optimization/LineSearch.hpp>


namespace Optimization 
{

LineSearch::LineSearch(Function &   decoratedObjFuncInfo,
                       unsigned int maxNumIterations)
{ 
    this->decoratedObjFuncInfo = &decoratedObjFuncInfo;
    setMaxNumIterations(maxNumIterations);
}

void LineSearch::setMaxNumIterations(unsigned int maxNumIterations)
{
    if (maxNumIterations < 1) 
    {
        throw std::invalid_argument("Maximum number of iterations must be greater than zero.");
    }
    this->maxNumIterations = maxNumIterations;
}

unsigned int LineSearch::getMaxNumIterations() const
{
    return maxNumIterations;
}

}
