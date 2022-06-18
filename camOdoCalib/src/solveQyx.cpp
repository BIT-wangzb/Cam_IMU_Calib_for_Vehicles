#include "solveQyx.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define Pi 3.1415926
#define DEG2ANG 57.295828
#define ANG2DEG 0.017453292

using namespace std;

SolveQyx::SolveQyx(){}

//查看收敛
/*bool SolveQyx::estimateRyx(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx)
{
    static ofstream f_yx;
    f_yx.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/camRyx.txt");
    f_yx<<std::fixed<<std::setprecision(6);
    f_yx <<"estimateRyx\n";
    size_t start_num = sync_result.size() - 15;
    double time0 = 0.0;
    bool first = true;
    for (size_t k = start_num; k > 15; --k)
    {
        size_t motionCnt = sync_result.size()-k;
        Eigen::MatrixXd M(motionCnt*4, 4);
        M.setZero();
        //1.构建最小二乘矩阵Ａ
        for(size_t i = 0; i < sync_result.size()-k; ++i)
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
        if (first)
        {
            first = false;
            time0 = sync_result[sync_result.size()-k].startTime;
        }
        double dt = sync_result[sync_result.size()-k].startTime - time0;
        f_yx <<dt<<" ";
        //TODO:: M^T * M
        //2.对Ａ矩阵进行ｓｖｄ分解
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

//        std::cout<<"----------Cam-------------\n";
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
//            std::cout<<"roll: "<<roll[i]<<" pitch: "<<pitch[i]<<" yaw: "<<yaw[i]<<std::endl;

        }

        // q_yx  means yaw is zero. we choose the smaller yaw
        if(fabs(yaw[0]) < fabs(yaw[1]) )
        {
            Ryx = R_yxs[0];
            f_yx << roll[0]<<" "<<pitch[0]<<endl;
        }else
        {
            Ryx = R_yxs[1];
            f_yx << roll[0]<<" "<<pitch[0]<<endl;
        }
    }


    return true;
}*/
bool SolveQyx::estimateRyx(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx)
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
    //2.对Ａ矩阵进行ｓｖｄ分解
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
//    Eigen::Matrix3d Rci_test;
//    Rci_test << 0.003160590246273326, -0.9999791454805219, -0.005631986624785923,
//            -0.002336363271321251, 0.0056246151765369234, -0.9999814523833838,
//            0.9999922760081504, 0.0031736899915518757, -0.0023185374437218464;

//    Eigen::Matrix3d Ric_test = Rci_test.inverse();
    Eigen::Matrix3d Ric_test;
//    R = Eigen::AngleAxisd(95*DEG2RAD,Eigen::Vector3d::UnitZ()) *
//        Eigen::AngleAxisd(180*DEG2RAD,Eigen::Vector3d::UnitY()) *
//        Eigen::AngleAxisd(95*DEG2RAD,Eigen::Vector3d::UnitX());
    Ric_test.setIdentity();

    size_t motionCnt = sync_result.size( );
    Eigen::MatrixXd M(motionCnt*4, 4);
    M.setZero();
    //1.构建最小二乘矩阵Ａ
    for(size_t i = 0; i < sync_result.size(); ++i)
    {
        double t12_length = sync_result[i].cam_t12.norm();
        if(t12_length < 1e-4)//相机移动距离不小于１微米
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
void SolveQyx::correctCamera(std::vector<data_selection::sync_data> &sync_result, std::vector<data_selection::cam_data> &camDatas,Eigen::Matrix3d Ryx)
{
    //1.检查数据是否正确(同步数据与相机的数量是一致的)
  if(sync_result.size() != camDatas.size())
  {
    std::cerr << "ERROR!! correctCamera: sync_result.size() != camDatas.size()" << std::endl;
    return;
  }
  std::vector<data_selection::sync_data> sync_tmp;
  std::vector<data_selection::cam_data> cam_tmp;
  for (unsigned int i = 0; i < sync_result.size(); ++i)
  {
    //2.获得 t(c1, c1c2)
    Eigen::Vector3d tlc_cam = camDatas[i].t12;//
    //3.将t左乘Rxy旋转使之成为一个整体
    Eigen::Vector3d tlc_corrected = Ryx * tlc_cam;
//    std::cout<<"tlc_correct1: "<<tlc_corrected.transpose()<<std::endl;
    Eigen::Vector3d n;
    n = Ryx.row(2);

    Eigen::Vector3d pi = Ryx * (tlc_cam - tlc_cam.dot(n) * n);
//    std::cout<<"tlc_correct2: "<<pi.transpose()<<std::endl;

    //4.保证相机整体坐标的分量是合理的（合理性需要再考虑）
    //tlc_corrected(1)与tlc_cam(2)理论上应该相等，都是车辆向前的方向
    if(tlc_corrected(1)*tlc_cam(2) < 0)
      continue;

    //5.去掉整体中ｚ轴的分量
    sync_result[i].scan_match_results[0] = tlc_corrected[0];
    sync_result[i].scan_match_results[1] = tlc_corrected[1];
    sync_tmp.push_back(sync_result[i]);
    cam_tmp.push_back(camDatas[i]);
  }
  sync_result.swap(sync_tmp);
  camDatas.swap(cam_tmp);
}

void SolveQyx::refineExPara(std::vector<data_selection::sync_data> sync_result,
                    cSolver::calib_result &internelPara,Eigen::Matrix3d Ryx)
{
    std::cout << std::endl << "there are  "<< sync_result.size() << " datas for refining extrinsic paras" << std::endl;
    std::vector<Eigen::Quaterniond> q_cam , q_odo;
    std::vector<Eigen::Vector3d> t_cam , t_odo;
    double r_L = internelPara.radius_l, r_R = internelPara.radius_r, axle = internelPara.axle;
    Eigen::Vector2d trc;
    trc << internelPara.l[0] , internelPara.l[1];
    for (int i = 0; i < int(sync_result.size()) - 3; ++i)
    {
        q_cam.push_back(sync_result[i].q21_cam);
        t_cam.push_back(sync_result[i].t21_cam);

        double vel_L = sync_result[i].velocity_left, vel_R = sync_result[i].velocity_right;
        double v = 0.5 * (r_L* vel_L + r_R * vel_R)                                 ,   omega = (r_R * vel_R - r_L * vel_L) / axle;
        Eigen::Quaterniond qlc_odo;
        Eigen::Vector3d tlc_odo, tcl_odo;
        double o_theta = omega * sync_result[i].T;
        double t1,t2;
        if (fabs(o_theta) > 1e-12) 
        {
          t1 = sin(o_theta) / o_theta;
          t2 = (1 - cos(o_theta)) / o_theta;
        }
        else {
          t1 = 1;
          t2 = 0;
        }
        Eigen::Vector3d eulerAngle(0.0,0.0,o_theta);
        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(0),Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1),Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(2),Eigen::Vector3d::UnitZ()));             
        qlc_odo = yawAngle*pitchAngle*rollAngle;
        tlc_odo = {v * sync_result[i].T * t1, v * sync_result[i].T * t2, 0.0};
        tcl_odo = -qlc_odo.matrix().inverse() * tlc_odo;
        Eigen::Quaterniond qcl_odo(qlc_odo.matrix().inverse());
        
        q_odo.push_back(qcl_odo);
        t_odo.push_back(tcl_odo);

    }

    Eigen::Matrix3d Rrc = Eigen::AngleAxisd(internelPara.l[2], Eigen::Vector3d::UnitZ() ) * Ryx;
    Eigen::Vector3d rotation_vector_rc ;
    q2Euler_zyx(Eigen::Quaterniond(Rrc) , rotation_vector_rc);
    std::cout << std::endl;
    std::cout <<  "before refine: Rrc(YPR) = " << rotation_vector_rc[0] <<  " " <<rotation_vector_rc[1] << " " << rotation_vector_rc[2] <<std::endl;
    std::cout << "before refine: trc =  " << trc[0] << "  " << trc[1] << std::endl;

    Eigen::Matrix4d Trc = Eigen::Matrix4d::Identity();
    Trc.block<3,3>(0,0) = Rrc;
    Trc.block<2,1>(0,3) = trc;
    refineEstimate(Trc, 1.0 ,q_odo,t_odo,q_cam,t_cam);
    Eigen::Vector3d Rrc_zyx;
    q2Euler_zyx(Eigen::Quaterniond(Trc.block<3,3>(0,0)) , Rrc_zyx);
    std::cout << std::endl << "after refine: Rrc(YPR) = " << Rrc_zyx[0] << "  " << Rrc_zyx[1] << "  " << Rrc_zyx[2] << std::endl;
    std::cout << "after refine trc = " << Trc(0,3) << "  " << Trc(1,3) << std::endl;

    //2019.06.22
    internelPara.l[0] = Trc(0,3);
    internelPara.l[1] = Trc(1,3);
    internelPara.l[2] = Rrc_zyx[0];

    double laser_std_x, laser_std_y, laser_std_th;
    cSolver cs;
    cs.estimate_noise(sync_result, internelPara, laser_std_x, laser_std_y, laser_std_th);

    /* Now compute the FIM */
    // 论文公式 9 误差的协方差
//  std::cout <<'\n' << "Noise: " << '\n' << laser_std_x << ' ' << laser_std_y
//            << ' ' << laser_std_th << std::endl;

    Eigen::Matrix3d laser_fim = Eigen::Matrix3d::Zero();
    laser_fim(0,0) = (float)1 / (laser_std_x * laser_std_x);
    laser_fim(1,1) = (float)1 / (laser_std_y * laser_std_y);
    laser_fim(2,2) = (float)1 / (laser_std_th * laser_std_th);

    Eigen::Matrix3d laser_cov = laser_fim.inverse();

    std::cout << '\n' << "-------Errors (std dev)-------" << '\n'
        << "cam-odom x: " << 1000 * sqrt(laser_cov(0,0)) << " mm" << '\n'
        << "cam-odom y: " << 1000 * sqrt(laser_cov(1,1)) << " mm" << '\n'
        << "cam-odom yaw: " << rad2deg(sqrt(laser_cov(2,2))) << " deg" << std::endl;

  // TODO
  // Compute 6*6 FIM
    Eigen::MatrixXd fim = Eigen::MatrixXd::Zero(6,6);
    fim = cs.compute_fim(sync_result,internelPara,laser_fim);
    Eigen::Matrix<double, 6,6> state_cov = fim.inverse();
    std::cout << '\n' << "-------Uncertainty-------" << '\n';
    std::cout << "Uncertainty Left wheel radius : "<< 1000 * sqrt(state_cov(0,0)) <<" mm \n";
    std::cout << "Uncertainty Right wheel radius : "<< 1000 * sqrt(state_cov(1,1)) <<" mm \n";
    std::cout << "Uncertainty Axle between wheels : "<< 1000 * sqrt(state_cov(2,2)) <<" mm \n";
    std::cout << "Uncertainty cam-odom-x : "<< 1000 * sqrt(state_cov(3,3)) <<" mm \n";
    std::cout << "Uncertainty cam-odom-y : "<< 1000 * sqrt(state_cov(4,4)) <<" mm \n";
    std::cout << "Uncertainty cam-odom-yaw : "<< rad2deg( sqrt(state_cov(5,5)) )<<" deg \n";
    std::cout << std::endl;
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

//#define LOSSFUNCTION
void SolveQyx::refineEstimate(Eigen::Matrix4d &Trc, double scale,
    const std::vector<Eigen::Quaterniond > &quats_odo,
    const std::vector<Eigen::Vector3d> &tvecs_odo,
    const std::vector<Eigen::Quaterniond> &quats_cam,
    const std::vector<Eigen::Vector3d> &tvecs_cam)
{
  Eigen::Quaterniond q(Trc.block<3,3>(0,0));
  double q_coeffs[4] = {q.w(),q.x(),q.y(),q.z()};
  double t_coeffs[3] = {Trc(0,3),Trc(1,3),Trc(2,3)};
  ceres::Problem problem;
  for(size_t i = 0; i< quats_odo.size(); ++i)
  {
      ceres::CostFunction * costfunction =
      new ceres::AutoDiffCostFunction<CameraOdomErr, 6,4,3>(
            new CameraOdomErr(quats_odo.at(i) , tvecs_odo.at(i), quats_cam.at(i) , tvecs_cam.at(i) ) );   //  residual : 6 ,  rotation: 4

      #ifdef LOSSFUNCTION
        //ceres::LossFunctionWrapper* loss_function(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);
        ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(costfunction, loss_function, q_coeffs, t_coeffs);
      #else
        problem.AddResidualBlock(costfunction, NULL, q_coeffs, t_coeffs);
      #endif

  }
  ceres::LocalParameterization* quaternionParameterization = new ceres::QuaternionParameterization;
  problem.SetParameterization(q_coeffs,quaternionParameterization);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, & summary);
  q = Eigen::Quaterniond(q_coeffs[0],q_coeffs[1],q_coeffs[2],q_coeffs[3]);

  Trc.block<3,3>(0,0) = q.toRotationMatrix();
  Trc.block<3,1>(0,3) << t_coeffs[0],t_coeffs[1],t_coeffs[2];

}

bool SolveQyx::estimateRyx2(const std::vector<Eigen::Quaterniond> &quats_odo,
  const std::vector<Eigen::Vector3d> &tvecs_odo,
  const std::vector<Eigen::Quaterniond > &quats_cam,
  const std::vector<Eigen::Vector3d > &tvecs_cam,
  Eigen::Matrix3d &R_yx)
{
  size_t motionCnt = quats_odo.size( );

  Eigen::MatrixXd M(motionCnt*4, 4);
  M.setZero();

  for(size_t i = 0; i < quats_odo.size(); ++i)
  {
    const Eigen::Quaterniond& q_odo = quats_odo.at(i);
          //const Eigen::Vector3d& t_odo = tvecs_odo.at(i);
    const Eigen::Quaterniond& q_cam = quats_cam.at(i);
          //const Eigen::Vector3d& t_cam = tvecs_cam.at(i);

    M.block<4,4>(i*4, 0) = QuaternionMultMatLeft(q_odo) - QuaternionMultMatRight(q_cam);
  }

          //TODO:: M^T * M
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU |Eigen::ComputeFullV);

    Eigen::Vector4d v1 = svd.matrixV().block<4,1>(0,2);
    Eigen::Vector4d v2 = svd.matrixV().block<4,1>(0,3);

    double lambda[2];      // solution from  ax^2+ bx + c = 0
    if( !SolveConstraintqyx(v1,v2,lambda[0],lambda[1]))
    {
        std::cout << "# ERROR: Quadratic equation cannot be solved due to negative determinant." << std::endl;
        return false;
    }

   // choose one lambda
    Eigen::Matrix3d R_yxs[2];
    double yaw[2];

    for( int i = 0; i<2;++i)
    {
        double t = lambda[i] * lambda[i] * v1.dot(v1) + 2 * lambda[i] * v1.dot(v2) + v2.dot(v2);

              // solve constraint ||q_yx|| = 1
        double lambda2 = sqrt(1.0/t);
        double lambda1 = lambda[i] * lambda2;

        Eigen::Quaterniond q_yx;
        q_yx.coeffs() = lambda1 * v1 + lambda2 * v2; // x,y,z,w

        R_yxs[i] = q_yx.toRotationMatrix();
        double roll,pitch;
        mat2RPY(R_yxs[i], roll, pitch, yaw[i]);
        std::cout<<"roll: "<<roll<<" pitch: "<<pitch<<" yaw: "<<yaw[i]<<std::endl;
    }

    // q_yx  means yaw is zero. we choose the smaller yaw
    if(fabs(yaw[0]) < fabs(yaw[1]) )
    {
      R_yx = R_yxs[0];
    }else
    {
      R_yx = R_yxs[1];
    }
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

//for test
bool SolveQyx::estimateRyx_cam_test(std::vector<data_selection::sync_data_test> sync_result,
                                Eigen::Matrix3d &Ryx) {

    size_t motionCnt = sync_result.size( );
    Eigen::MatrixXd M(motionCnt*4, 4);
    M.setZero();
    //1.构建最小二乘矩阵Ａ
    for(size_t i = 0; i < sync_result.size(); ++i)
    {
        double tlc_length = sync_result[i].cam_t12.norm();
        if(tlc_length < 1e-4)//相机移动距离不小于１微米
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
        //block右值操作，往里面写数据
        M.block<4,4>(i*4, 0) = sin(sync_result[i].angle)*M_tmp;
    }
    //M.conservativeResize((id-1)*4,4);

    //TODO:: M^T * M
    //2.对Ａ矩阵进行ｓｖｄ分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU |Eigen::ComputeFullV);

    //3.find two optimal solutions(最小奇异值对应的列向量)
    Eigen::Vector4d v1 = svd.matrixV().block<4,1>(0,2);
    Eigen::Vector4d v2 = svd.matrixV().block<4,1>(0,3);


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

    std::cout<<"----------Cam-------------\n";
    for( int i = 0; i<2;++i)
    {
        double t = lambda[i] * lambda[i] * v1.dot(v1) + 2 * lambda[i] * v1.dot(v2) + v2.dot(v2);

        // solve constraint ||q_yx|| = 1
        double lambda2 = sqrt(1.0/t);
        double lambda1 = lambda[i] * lambda2;

        Eigen::Quaterniond q_yx;
        q_yx.coeffs() = lambda1 * v1 + lambda2 * v2; // x,y,z,w

        R_yxs[i] = q_yx.toRotationMatrix();
        double roll,pitch;
        mat2RPY(R_yxs[i], roll, pitch, yaw[i]);
        roll = roll * DEG2ANG;
        pitch = pitch * DEG2ANG;
        std::cout<<"roll: "<<roll<<" pitch: "<<pitch<<" yaw: "<<yaw[i]<<std::endl;
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
//for test
bool SolveQyx::estimateRyx_odo_test(std::vector<data_selection::sync_data_test> sync_result,
                                    Eigen::Matrix3d &Ryx) {

    size_t motionCnt = sync_result.size( );
    Eigen::MatrixXd M(motionCnt*4, 4);
    M.setZero();
    //1.构建最小二乘矩阵Ａ
    for(size_t i = 0; i < sync_result.size(); ++i)
    {
        double t12_length = sync_result[i].t12_odo.norm();
        if(t12_length < 1e-4)//相机移动距离不小于１微米
            continue;
        if(sync_result[i].axis_odo(2) > -0.96)//最好只绕z轴旋转，即要接近于-1，axis(1)是相机y轴，且均为负数
            continue;
        //axis[i]为x,y,z旋转轴，与deltaTheta相乘就是具体绕[i]轴旋转的角度
        const Eigen::Vector3d& axis = sync_result[i].axis_odo;

        Eigen::Matrix4d M_tmp;
        M_tmp << 0, -1-axis[2], axis[1], -axis[0],
                axis[2]+1, 0, -axis[0], -axis[1],
                -axis[1], axis[0], 0 , 1-axis[2],
                axis[0], axis[1], -1+axis[2], 0;
        //block右值操作，往里面写数据
        M.block<4,4>(i*4, 0) = sin(sync_result[i].angle_odo)*M_tmp;
    }
    //M.conservativeResize((id-1)*4,4);

    //TODO:: M^T * M
    //2.对Ａ矩阵进行ｓｖｄ分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU |Eigen::ComputeFullV);

    //3.find two optimal solutions(最小奇异值对应的列向量)
    Eigen::Vector4d v1 = svd.matrixV().block<4,1>(0,2);
    Eigen::Vector4d v2 = svd.matrixV().block<4,1>(0,3);


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

    std::cout<<"----------odo-------------\n";
    for( int i = 0; i<2;++i)
    {
        double t = lambda[i] * lambda[i] * v1.dot(v1) + 2 * lambda[i] * v1.dot(v2) + v2.dot(v2);

        // solve constraint ||q_yx|| = 1
        double lambda2 = sqrt(1.0/t);
        double lambda1 = lambda[i] * lambda2;

        Eigen::Quaterniond q_yx;
        q_yx.coeffs() = lambda1 * v1 + lambda2 * v2; // x,y,z,w
//        std::cout<<"q coeffs: "<<q_yx.coeffs().transpose()<<std::endl;

        R_yxs[i] = q_yx.toRotationMatrix();
        double roll,pitch;
        mat2RPY(R_yxs[i], roll, pitch, yaw[i]);
        roll = roll * DEG2ANG;
        pitch = pitch * DEG2ANG;
        std::cout<<"roll: "<<roll<<" pitch: "<<pitch<<" yaw: "<<yaw[i]<<std::endl;
        //test
        Eigen::Vector3d euler;
        toEulerAngle(q_yx,euler);
//        std::cout<<"euler: "<<euler.transpose() * DEG2ANG<<std::endl;
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

void SolveQyx::toEulerAngle(Eigen::Quaterniond q,Eigen::Vector3d &euler)

{
    double roll, pitch, yaw;
    //roll
    double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
    double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
    if (fabs(sinp) >= 1)
        pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        pitch = asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    yaw = atan2(siny_cosp, cosy_cosp);
    euler(0) = roll;
    euler(1) = pitch;
    euler(2) = yaw;
}