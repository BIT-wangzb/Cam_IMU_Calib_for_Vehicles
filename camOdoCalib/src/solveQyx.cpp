#include "solveQyx.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define Pi 3.1415926
#define DEG2ANG 57.295828
#define ANG2DEG 0.017453292

using namespace std;

SolveQyx::SolveQyx(){}

//
bool SolveQyx::estimateRyx_cam(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx)
{
    size_t motionCnt = sync_result.size();
    Eigen::MatrixXd M(motionCnt*4, 4);
    M.setZero();
    //1.构建最小二乘矩阵Ａ
    for(size_t i = 0; i < sync_result.size(); ++i)
    {
        double t12_length = sync_result[i].cam_t12.norm();
        if(t12_length < 1e-4)//相机移动距离不小于１微米
            continue;
        if(sync_result[i].axis(1) > -0.96)//最好只绕y轴旋转，即要接近于-1，axis(1)是相机y轴，且均为负数
            continue;
        //axis[i]为x,y,z旋转轴，与deltaTheta相乘就是具体绕[i]轴旋转的角度
        const Eigen::Vector3d& axis = sync_result[i].axis;

        Eigen::Matrix4d M_tmp;
        M_tmp << 0, -1-axis[2], axis[1], -axis[0],
                axis[2]+1, 0, -axis[0], -axis[1],
                -axis[1], axis[0], 0 , 1-axis[2],
                axis[0], axis[1], -1+axis[2], 0;
//    M_tmp << 0, -1-axis[2], axis[1], axis[0],
//              axis[2]+1, 0, -axis[0], axis[1],
//              -axis[1], axis[0], 0 , axis[2]-1,
//              -axis[0], -axis[1], 1-axis[2], 0;
        //block右值操作，往里面写数据
        M.block<4,4>(i*4, 0) = sin(sync_result[i].angle/2)*M_tmp;
    }
    //M.conservativeResize((id-1)*4,4);
    //TODO:: M^T * M
    //2.对A矩阵进行SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU |Eigen::ComputeFullV);

    //3.find two optimal solutions(最小奇异值对应的列向量)
    Eigen::Vector4d v1 = svd.matrixV().block<4,1>(0,2);
    Eigen::Vector4d v2 = svd.matrixV().block<4,1>(0,3);

//        std::cout<<"singularValues: "<<svd.singularValues().transpose()<<std::endl;

    //4.求解带有约束的线性二元二次方程，得到可以恢复qyx的线性参数解 lambda1 lambda2
    double lambda[2];      // solution from  ax^2+ bx + c = 0
    if( !SolveConstraintqyx(v1,v2,lambda[0],lambda[1]))
    {
        std::cout << "# ERROR: Quadratic equation cannot be solved due to negative determinant." << std::endl;
        return false;
    }

    // choose one lambda
    Eigen::Matrix3d R_yxs[2];
    double yaw[2];
    double roll[2],pitch[2];

        std::cout<<"----------Cam-------------\n";
    for( int i = 0; i<2;++i)
    {
        //获取的第1个系数有两个解，第二个系数为1
        double t = lambda[i] * lambda[i] * v1.dot(v1) + 2 * lambda[i] * v1.dot(v2) + v2.dot(v2);

        // solve constraint ||q_yx|| = 1
        //通过此约束可以获得第二个系数的解析解
        double lambda2 = sqrt(1.0/t);
        double lambda1 = lambda[i] * lambda2;

        Eigen::Quaterniond q_yx;
        q_yx.coeffs() = lambda1 * v1 + lambda2 * v2; // x,y,z,w

        R_yxs[i] = q_yx.toRotationMatrix();
        mat2RPY(R_yxs[i], roll[i], pitch[i], yaw[i]);
        roll[i] = roll[i] * DEG2ANG;
        pitch[i] = pitch[i] * DEG2ANG;
        std::cout<<"roll: "<<roll[i]<<" pitch: "<<pitch[i]<<" yaw: "<<yaw[i]<<std::endl;

    }

    // q_yx  means yaw is zero. we choose the smaller yaw
    if(fabs(yaw[0]) < fabs(yaw[1]) )
    {
        Ryx = R_yxs[0];
    }else
    {
        Ryx = R_yxs[1];
    }
    return true;
}
bool SolveQyx::estimateRyx_imu(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx)
{   
    size_t motionCnt = sync_result.size( );
    Eigen::MatrixXd M(motionCnt*4, 4);
    M.setZero();
    //1.构建最小二乘矩阵Ａ
    for(size_t i = 0; i < sync_result.size(); ++i)
    {
        double t12_length = sync_result[i].cam_t12.norm();
        if(t12_length < 1e-4)//相机移动距离大于阈值
            continue;
        if(sync_result[i].axis_imu(2) > -0.96)//此处依然是相机，最好只绕y轴旋转，即要接近于-1，axis(1)是相机y轴，且均为负数
            continue;

        Eigen::Quaterniond q1 = sync_result[i].q21_cam;
        Eigen::Quaterniond q2 = sync_result[i].q21_imu;
        Eigen::Matrix3d Rc_g = Ric_test.inverse() * q2 * Ric_test;
        Eigen::Quaterniond r2(Rc_g);

        double angular_distance = 180 / M_PI * q1.angularDistance(r2);
//        std::cout<<"angular distance: "<<angular_distance<<std::endl;
        double huber = angular_distance > 1.5 ? 1.5/angular_distance : 1.0;
        //axis[i]为x,y,z旋转轴，与deltaTheta相乘就是具体绕[i]轴旋转的角度
        const Eigen::Vector3d& axis = sync_result[i].axis_imu;

        //此处四元数是JPL表述形式
        Eigen::Matrix4d M_tmp;
//        M_tmp << 0, -1-axis[2], axis[1], -axis[0],
//                axis[2]+1, 0, -axis[0], -axis[1],
//                -axis[1], axis[0], 0 , 1-axis[2],
//                axis[0], axis[1], -1+axis[2], 0;
        //Hamilton形式，对应约束为: SolveConstraintqyx_test
        M_tmp << 0, axis[0], axis[1], -1+axis[2],
                -axis[0], 0, -1-axis[2], axis[1],
                -axis[1], 1+axis[2], 0 , -axis[0],
                axis[2], -axis[1], axis[0], 0;

        //block右值操作，往里面写数据
//        M.block<4,4>(i*4, 0) = huber*sin(sync_result[i].angle_imu/2)*M_tmp;
        M.block<4,4>(i*4, 0) = sin(sync_result[i].angle_imu/2)*M_tmp;
    }
    //M.conservativeResize((id-1)*4,4);

    //TODO:: M^T * M
    //2.对Ａ矩阵进行ｓｖｄ分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU |Eigen::ComputeFullV);
    /*Eigen::Matrix<double,4,1> x = svd.matrixV().col(3);
    Eigen::Vector3d roi_cov;
    roi_cov = svd.singularValues().tail(3);
    std::cout<<"-----------IMU------------\n"<<std::endl;
    std::cout<<"roi_cov: "<<roi_cov.transpose()<<std::endl;
    if (roi_cov(1)>0.01)
    {
        double roll,pitch;
        double yaw;
        Eigen::Quaterniond estimated_R_yx(x);
        Eigen::Matrix3d R = estimated_R_yx.toRotationMatrix();
        std::cout<<"R_yx = \n"<<R<<std::endl;
        Eigen::Vector3d e1 = R.eulerAngles(2,1,0);
        std::cout<<"e1: "<<e1.transpose() * DEG2ANG<<std::endl;
        mat2RPY(R, roll, pitch, yaw);
//        if (fabs(roll) > 6.0/4.0*M_PI);
//            roll = roll - 2*M_PI;
        std::cout<<"roll_: "<<roll<<" pitch: "<<pitch<<" yaw: "<<yaw<<std::endl;
        Eigen::Matrix3d R_test;
        R_test = Eigen::AngleAxisd(0,Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX());
//        Eigen::Vector3d e2 = R_test.inverse().eulerAngles(2,1,0);
        Ryx = R_test;
//        std::cout<<"R_yx_new: \n"<<R_test<<std::endl;
    }*/

    //3.find two optimal solutions(最小奇异值对应的列向量)
    Eigen::Vector4d v1 = svd.matrixV().block<4,1>(0,2);
    Eigen::Vector4d v2 = svd.matrixV().block<4,1>(0,3);
//    std::cout<<"singularValues: "<<svd.singularValues().transpose()<<std::endl;
//    std::cout<<"singularValues: "<<svd.singularValues().transpose()<<std::endl;

    //4.求解带有约束的线性二元二次方程，得到可以恢复qyx的线性参数解 lambda1 lambda2
    double lambda[2];      // solution from  ax^2+ bx + c = 0
//    if( !SolveConstraintqyx(v1,v2,lambda[0],lambda[1]))
//    {
//        std::cout << "# ERROR: Quadratic equation cannot be solved due to negative determinant." << std::endl;
//        return false;
//    }
    if( !SolveConstraintqyx_test(v1,v2,lambda[0],lambda[1]))
    {
        std::cout << "# ERROR: Quadratic equation cannot be solved due to negative determinant." << std::endl;
        return false;
    }

    // choose one lambda
    Eigen::Matrix3d R_yxs[2];
    double yaw[2];

    std::cout<<"----------IMU-------------\n";
    for( int i = 0; i<2;++i)
    {
        double t = lambda[i] * lambda[i] * v1.dot(v1) + 2 * lambda[i] * v1.dot(v2) + v2.dot(v2);

        // solve constraint ||q_yx|| = 1
        double lambda2 = sqrt(1.0/t);
        double lambda1 = lambda[i] * lambda2;

        Eigen::Quaterniond q_yx;
//        q_yx.coeffs() = lambda1 * v1 + lambda2 * v2; // x,y,z,w
        Eigen::Vector4d x = lambda1 * v1 + lambda2 * v2;
        q_yx.w() = x(0);
        q_yx.x() = x(1);
        q_yx.y() = x(2);
        q_yx.z() = x(3);
//        double alpha1 = atan2(q_yx.x(), q_yx.w());
//        double alpha2 = atan2(-q_yx.z(), q_yx.y());
//        cout<<"alpha1 = "<<2*alpha1*DEG2ANG<<", alpha2 = "<<2*alpha2*DEG2ANG<<endl;
//        double beta1 = atan2(q_yx.y(), q_yx.w());
//        double beta2 = atan2(-q_yx.z(), q_yx.x());
//        cout<<"beta1 = "<<2*beta1*DEG2ANG<<", alpha2 = "<<2*beta2*DEG2ANG<<endl;
//        Eigen::Vector3d test1;
//        q2Euler_zyx(q_yx,test1);
//        std::cout<<"test1: "<<test1.transpose()*DEG2ANG<<std::endl;

//        double tan_roll = q_yx.x() / q_yx.w();
//        std::cout<<"tan_roll: "<<tan_roll<<std::endl;
//        double tan_pitch = -q_yx.z() / q_yx.y();
//        std::cout<<"tan_pitch: "<<tan_pitch<<std::endl;
//        double roll_test = atan(q_yx.x() / q_yx.w()) * DEG2ANG;
//        double pitch_test = atan(-q_yx.z() / q_yx.y()) * DEG2ANG;
//        std::cout<<"roll_test: "<<roll_test<<" pitch_test: "<<pitch_test<<std::endl;
        R_yxs[i] = q_yx.toRotationMatrix();

        Eigen::Vector3d e1 = R_yxs[i].eulerAngles(2,1,0);
        std::cout<<"e1: "<<e1.transpose() * DEG2ANG<<std::endl;
        double roll,pitch;
//        mat2RPY2(R_yxs[i], roll, pitch, yaw[i]);
        mat2RPY(R_yxs[i], roll, pitch, yaw[i]);
        roll = roll * DEG2ANG;
        pitch = pitch * DEG2ANG;
        std::cout<<"roll: "<<roll<<" pitch: "<<pitch<<" yaw: "<<yaw[i]<<std::endl;
        //zed_imu
        if (fabs(roll) > 170)
        {
            if (roll > 0)
                roll = roll-180;
            else
                roll = 180 - fabs(roll);
        }
        R_yxs[i] = Eigen::AngleAxisd(0,Eigen::Vector3d::UnitZ())*
                Eigen::AngleAxisd(-pitch*ANG2DEG,Eigen::Vector3d::UnitY())*
                Eigen::AngleAxisd(roll*ANG2DEG,Eigen::Vector3d::UnitX());
        //oxts_imu
//        R_yxs[i] = Eigen::AngleAxisd(0,Eigen::Vector3d::UnitZ())*
//                Eigen::AngleAxisd(pitch*ANG2DEG,Eigen::Vector3d::UnitY())*
//                Eigen::AngleAxisd(roll*ANG2DEG,Eigen::Vector3d::UnitX());
    }

    // q_yx  means yaw is zero. we choose the smaller yaw
    if(fabs(yaw[0]) < fabs(yaw[1]) )
    {
        Ryx = R_yxs[0];

    }else
    {
        Ryx = R_yxs[1];
    }
    return true;
}

void SolveQyx::q2Euler_zyx(Eigen::Quaterniond q, Eigen::Vector3d &res)
{
  double r11 = 2*(q.x()*q.y() + q.w()*q.z());
  double r12 = q.w()*q.w() + q.x()*q.x() - q.y()*q.y() - q.z()*q.z();
  double r21 = -2*(q.x()*q.z() - q.w()*q.y());
  double r31 = 2*(q.y()*q.z() + q.w()*q.x());
  double r32 =  q.w()*q.w() - q.x()*q.x() - q.y()*q.y() + q.z()*q.z();
  res[2] = atan2( r31, r32 );//yaw
  res[1] = asin ( r21 );//pitch
  res[0] = atan2( r11, r12 );//roll
  //test
  res[0] = atan2(q.x(),q.w());
}

/*
 *    constraint for q_yx:
 *              xy = -zw
 *     this can transform to a  equation  ax^2+ bx + c = 0
 *     假设第2个系数为1，因为任意线性组合都是方程的解
 */
bool SolveQyx::SolveConstraintqyx(const Eigen::Vector4d t1, const Eigen::Vector4d t2, double& x1, double& x2)
{
    double a = t1(0) * t1(1)+ t1(2)*t1(3);
    double b = t1(0) * t2(1)+ t1(1)*t2(0)+t1(2)*t2(3)+ t1(3)*t2(2);
    double c = t2(0) * t2(1)+ t2(2)*t2(3);
    if ( std::fabs(a) < 1e-10)
    {
        x1 = x2 = -c/b;//假设第2个系数为1，因为任意线性组合都是方程的解
        return true;
    }
    double delta2 = b*b - 4.0 * a * c;

    if(delta2 < 0.0) return false;

    double delta = sqrt(delta2);

    x1 = (-b + delta)/(2.0 * a);
    x2 = (-b - delta)/(2.0 * a);

    return true;

}

bool SolveQyx::SolveConstraintqyx_test(const Eigen::Vector4d t1, const Eigen::Vector4d t2, double& x1, double& x2)
{
    double a = t1(0) * t1(2)+ t1(1)*t1(2);
    double b = t1(0) * t2(3)+ t1(3)*t2(0)+
               t1(1)*t2(2)+ t1(2)*t2(1);
    double c = t2(0) * t2(3)+ t2(1)*t2(2);
    if ( std::fabs(a) < 1e-10)
    {
        x1 = x2 = -c/b;//假设第2个系数为1，因为任意线性组合都是方程的解
        return true;
    }
    double delta2 = b*b - 4.0 * a * c;

    if(delta2 < 0.0) return false;

    double delta = sqrt(delta2);

    x1 = (-b + delta)/(2.0 * a);
    x2 = (-b - delta)/(2.0 * a);
//    std::cout<<"x1: "<<x1<<", x2: "<<x2<<std::endl;

    return true;

}