#include <Optimization/Result.hpp>

namespace Optimization
{

std::ostream & operator<<(std::ostream & out, 
                          const Result & result)
{
    out << "---------------------------------------- Result ----------------------------------------\n";
    out << "               Exit flag                     : ";
    
    if (result.exitFlag == Gradient) 
    {
        out << "Reached gradient tolerance\n";
    } 
    else if (result.exitFlag == Relative) 
    {
        out << "Reached relative tolerance\n";
    } 
    else if (result.exitFlag == MaxNumIterations) 
    {
        out << "Reached maximum number of allowed iterations\n";
    } 
    else if (result.exitFlag == LineSearchFailed) 
    {
        out << "Line search failed\n";
    } 
    else 
    {
        out << "Unknown exit flag\n";
    }
    
    out << "               Gradient norm                 : " << result.optGradNorm << std::endl;
    out << "               Number of iterations          : " << result.numIterations << std::endl;
    out << "               Number of function evaluations: " << result.numFuncEvaluations << std::endl;
    out << "               Number of gradient evaluations: " << result.numGradEvaluations << std::endl;
    out << "               Function value                : " << result.optFuncValue << std::endl;
    out << "               Optimal parameters            : ";
    int n = result.optParameters.size();
    for (int i = 0; i < n; i++)
    {
        out << result.optParameters(i) << ((i == n - 1) ? "" : ", ");
    }
    out << std::endl;
    out << "----------------------------------------------------------------------------------------\n";
    
    return out;
}

}