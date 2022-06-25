#ifndef SOLVEQXY_H
#define SOLVEQXY_H

#include "solver.h"
#include <ceres/ceres.h>


/*
 *  Rotation matrix to euler
*/
template<typename T>
void mat2RPY2(const Eigen::Matrix<T, 3, 3>& m, T roll[2], T pitch[2], T yaw[2])
{
//    std::cout<<"R31: "<<m(2,0)<<std::endl;
    double pitch1 = -asin(m(2,0));
    double pitch2 = M_PI + asin(m(2,0));
    double roll1 = atan2(m(2,1)/cos(pitch1), m(2,2)/cos(pitch1));
    double roll2 = atan2(m(2,1)/cos(pitch2), m(2,2)/cos(pitch2));
    double yaw1 = atan2(m(1,0)/cos(pitch1), m(0,0)/cos(pitch1));
    double yaw2 = atan2(m(1,0)/cos(pitch2), m(0,0)/cos(pitch2));
//    std::cout<<"roll pitch yaw:\n";
    double c = 180/M_PI;
    roll[0] = roll1*c; pitch[0] = pitch1*c; yaw[0] = yaw1*c;
    roll[1] = roll2*c; pitch[1] = pitch2*c; yaw[1] = yaw2*c;

//    std::cout<<roll1*c<<" "<<pitch1*c<<" "<<yaw1*c<<std::endl;
//    std::cout<<roll2*c<<" "<<pitch2*c<<" "<<yaw2*c<<std::endl;
}
template<typename T>
void mat2RPY(const Eigen::Matrix<T, 3, 3>& m, T& roll, T& pitch, T& yaw)
{
    roll = atan2(m(2,1), m(2,2));
    pitch = atan2(-m(2,0), sqrt(m(2,1) * m(2,1) + m(2,2) * m(2,2)));
    yaw = atan2(m(1,0), m(0,0));
}

class SolveQyx
{
public:
    SolveQyx();

    bool estimateRyx_cam(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx);
    bool estimateRyx_imu(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx);

    void q2Euler_zyx(Eigen::Quaterniond q, Eigen::Vector3d &res);

private:

    bool SolveConstraintqyx(const Eigen::Vector4d t1, const Eigen::Vector4d t2, double& x1, double& x2);
    bool SolveConstraintqyx_test(const Eigen::Vector4d t1,
                                 const Eigen::Vector4d t2,
                                 double& x1, double& x2);

};
#endif