#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;
 
int main()
{
    MatrixXd m = MatrixXd::Random(3, 3);
    m = (m + MatrixXd::Constant(3, 3, 1.2)) * 50;

    VectorXd v(3);
    v << 1, 2, 3;

    std::cout << "m =" << std::endl << m << "\n" << std::endl;
    std::cout << "v =" << std::endl << v << "\n" << std::endl;
    std::cout << "m * v =" << std::endl << m * v << std::endl;
}