#include <iostream>
#include "solver.h"
#include "solveQyx.h"
//#include "utils.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <sophus/so3.hpp>

using namespace std;
#define DEG2RAD 0.01745329
#define RAD2DEG 57.29578

using namespace Eigen;
cSolver::cSolver()
{
    cv::FileStorage fs("../config.yaml",cv::FileStorage::READ);
    ws = fs["sliding_window"];
    K1 = fs["K1"];
    K2 = fs["K2"];
    cout<<"ws: "<<ws<<endl;
    fs.release();
}

//求解陀螺仪Bias
void cSolver::solveGyroscopeBias(std::vector<data_selection::sync_data> &calib_data,
                                 Eigen::Matrix3d Rcb)
{
    //选择连续数据，并且符合条件
    static ofstream f_bias;
    f_bias.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/bias.txt");
    f_bias<<std::fixed<<std::setprecision(6);
    f_bias <<"bias\n";
//    Eigen::Matrix3d Rcb;
//    Rcb << 0.003160590246273326, -0.9999791454805219, -0.005631986624785923,
//            -0.002336363271321251, 0.0056246151765369234, -0.9999814523833838,
//            0.9999922760081504, 0.0031736899915518757, -0.0023185374437218464;
    int frame_count = 15;
    int data_size = 0;
    while (true)
    {
        std::vector<data_selection::sync_data> tmp_data;
        bool first_ = false;
        int num = 0;
        int start_i = data_size*frame_count;
        int end_i = start_i + frame_count;
        if ((calib_data.size() - end_i) < frame_count)
            break;
//            end_i = calib_data.size();
        for (int i = start_i; i < end_i; ++i)
            tmp_data.push_back(calib_data[i]);
        /*for (int i = start_i; i < end_i; ++i)
        {
            double tlc_length = calib_data[i].cam_t12.norm();
            if (first_)
            {
                tmp_data.push_back(calib_data[i]);
                if (tlc_length > 1e-4 && calib_data[i].axis(1) < -0.96)
                    num++;
            }
            else
            {
                if(tlc_length > 1e-4 && calib_data[i].axis(1) < -0.96)
                {
                    first_ = true;
                    tmp_data.push_back(calib_data[i]);
                    num++;
                    continue;
                }
            }

        }*/
        data_size++;

        //开始求解bias
        Eigen::Matrix3d A;
        Eigen::Vector3d b;
        Eigen::Vector3d delta_bg;
        A.setZero();
        b.setZero();
        for (int j = 0; j < tmp_data.size(); ++j)
        {
            Eigen::Matrix3d tmp_A;
            tmp_A.setZero();
            Eigen::Vector3d tmp_b;
            tmp_b.setZero();
            tmp_A = tmp_data[j].jacobian_.template block<3,3>(3,12);
            Eigen::Matrix3d Rwb1 = tmp_data[j].Rwc1_cam * Rcb;
            Eigen::Matrix3d Rwb2 = tmp_data[j].Rwc2_cam * Rcb;
            Eigen::Quaterniond qwb1(Rwb1);
            Eigen::Quaterniond qwb2(Rwb2);
            tmp_b = 2*(tmp_data[j].q21_imu.inverse() * qwb1.inverse() * qwb2).vec();

            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }

        delta_bg = A.ldlt().solve(b);
//        cout<<"gyroscope bias: "<<delta_bg.transpose()<<endl;
        f_bias << delta_bg(0)<<" ";
        f_bias << delta_bg(1)<<" ";
        f_bias << delta_bg(2)<<endl;

        for (int i = start_i; i < end_i; ++i)
            calib_data[i].Bg = delta_bg;

    }

//    return;
    //开始求解bias

    f_bias.close();
    return;
}

//测试，求解重力
void cSolver::solveGravity(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                               Eigen::Matrix3d Rci,
                               Eigen::Vector3d &gc)
{
    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }

    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        cout<<"segmentCount: "<<segmentCount<<endl;
        int num = calib_data[segmentId].size()-1;
        cout<<"num: "<<num<<endl;
        Eigen::MatrixXd tmp_G(num*3,3);
        tmp_G.setZero();
        Eigen::MatrixXd tmp_b(num*3,1);
        Eigen::Vector3d tcb;
        tcb.setZero();
//        tcb(0) = 0.02241856;
//        tcb(1) = -0.01121906;
//        tcb(2) = -0.01653902;
        for (int motionId = 0; motionId < calib_data[segmentId].size()-1; ++motionId)
        {
            Eigen::Vector3d pc1 = calib_data[segmentId][motionId].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[segmentId][motionId].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[segmentId][motionId+1].twc2_cam;
//            cout<<"pc1: "<<pc1.transpose()<<endl;

            Eigen::Vector3d t12_imu = calib_data[segmentId][motionId].P;
            Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+1].P;
            Eigen::Vector3d v12_imu = calib_data[segmentId][motionId].V;
            Eigen::Vector3d t12_cam = calib_data[segmentId][motionId].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+1].Rwc2_cam;
            double dT12 = calib_data[segmentId][motionId].T;
            double dT23 = calib_data[segmentId][motionId+1].T;
            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
            Eigen::Vector3d phi = (Rwc2_cam - Rwc3_cam)*tcb*dT12 +
                    (Rwc2_cam - Rwc1_cam)*tcb*dT23;//pcb
            Eigen::Vector3d gamma = -Rwc1_cam*Rci*t12_imu*dT23 +
                    Rwc2_cam*Rci*t23_imu*dT12 + Rwc1_cam*Rci*v12_imu*dT12*dT23;//Rcb
            tmp_G.block<3,3>(3*motionId,0) = beta;
            tmp_b.block<3,1>(3*motionId,0) = phi + gamma - lambda;

        }
        //求解
        Eigen::VectorXd x;
        x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);
//        A = A * 1000;
//        b = b * 1000;
//        Eigen::VectorXd x(6);
//        x = A.ldlt().solve(b);
//        Eigen::Vector2d t(x(0), x(1));
//        double alpha = atan2(x(3), x(2));
//        double lx = x(2);
//        double lz = x(3);
        std::cout<<"--------result--------\n";
        std::cout<<"Gravity: "<<x.transpose()<<std::endl;
//        std::cout<<"alpha = "<<alpha * DEG2RAD<<", lx = "<<lx<<", lz = "<<lz<<std::endl;

    }
}


//设Rcb已知，求解gc和pcb 非滑窗
void cSolver::solvePcb(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                               const Eigen::Matrix3d Rcb,
                               Eigen::Vector3d &gc)
{
    static ofstream f_pcb;
    f_pcb.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/tcb_estimation.txt");
    f_pcb<<std::fixed<<std::setprecision(6);
    f_pcb <<"solveOtherResult_test2 非滑窗\n";
    f_pcb<<"solveOtherResult_test2 zed_tci = [0.0224, -0.0112, -0.0165]\n";
    f_pcb<<"solveOtherResult_test2 oxts_tci = [-0.081, 1.03, -1.45]\n";
    f_pcb <<"gcx, gcy, gcz, pcb_x, pcb_y, pcb_z\n";

    bool first_solve = true;
    int segmentCount = calib_data.size();
    Eigen::Vector3d gc_init = Eigen::Vector3d::Zero();
    Eigen::Vector3d pcb_init = Eigen::Vector3d::Zero();
    vector<double> x1,x2,x3,x4,x5,x6;
    double time0 = 0.0;
    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        f_pcb <<"---------\n";
        int num = calib_data[segmentId].size()-1;
        Eigen::MatrixXd tmp_G(num*3,6);
        tmp_G.setZero();
        Eigen::MatrixXd tmp_b(num*3,1);
        tmp_b.setZero();
        for (int motionId = 0; motionId < calib_data[segmentId].size()-1; ++motionId)
        {
            Eigen::Vector3d pc1 = calib_data[segmentId][motionId].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[segmentId][motionId].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[segmentId][motionId+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[segmentId][motionId].P;
            Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+1].P;
            Eigen::Vector3d v12_imu = calib_data[segmentId][motionId].V;
            Eigen::Vector3d t12_cam = calib_data[segmentId][motionId].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+1].Rwc2_cam;
            double dT12 = calib_data[segmentId][motionId].T;
            double dT23 = calib_data[segmentId][motionId+1].T;
            if (dT12 != calib_data[segmentId][motionId].sum_dt)
                cout<<"sum_dt: "<<calib_data[segmentId][motionId].sum_dt<<", dT12: "<<dT12<<endl;
            if (dT23 != calib_data[segmentId][motionId+1].sum_dt)
                cout<<"sum_dt: "<<calib_data[segmentId][motionId+1].sum_dt<<", dT23: "<<dT23<<endl;

            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
            Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*dT23;//pcb

            //gamma
            Eigen::Vector3d gamma_1 =  Rwc1_cam * Rcb * t12_imu * dT23;
            Eigen::Vector3d gamma_2 =  -Rwc2_cam * Rcb * t23_imu * dT12;
            Eigen::Vector3d gamma_3 = -Rwc1_cam * Rcb * v12_imu * dT12 * dT23;
            Eigen::Vector3d gamma = gamma_1 + gamma_2 + gamma_3;
            Eigen::Vector3d test_b = -lambda + gamma;

//            计算权重
            if (!first_solve)
            {
                Eigen::Vector3d pred = beta * gc_init + phi * pcb_init;
                Eigen::Vector3d e = pred - test_b;
                double wi = exp(-e.norm()*K2);
//                if (e.norm() > 1.2)
//                    wi = 0.0;
                beta *= wi;
                phi *= wi;
                test_b *= wi;
                f_pcb <<"e.norm="<<e.norm()<<" ";
                f_pcb <<"wi="<<wi<<" ";
            }

            tmp_G.block<3,3>(motionId*3,0) = beta;
            tmp_G.block<3,3>(motionId*3,3) = phi;
            tmp_b.block<3,1>(motionId*3,0) = test_b;

            if (motionId > 15)
            {
                if (first_solve)
                {
                    first_solve = false;
                    time0 = calib_data[segmentId][motionId].startTime;
                }

                Eigen::MatrixXd G(motionId*3+3,6);
                G = tmp_G.topRows(motionId*3+3);
                Eigen::VectorXd b(motionId*3+3);
                b = tmp_b.topRows(motionId*3+3);
                //求解
                Eigen::VectorXd x;
                x = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                gc_init(0) = x(0);
                gc_init(1) = x(1);
                gc_init(2) = x(2);
                pcb_init(0) = x(3);
                pcb_init(1) = x(4);
                pcb_init(2) = x(5);
                double time = calib_data[segmentId][motionId].startTime-time0;
//                double time = calib_data[segmentId][motionId].startTime;
                f_pcb <<time<<" ";
                f_pcb << x(0) <<" ";
                f_pcb << x(1) <<" ";
                f_pcb << x(2) <<" ";
                f_pcb << x(3) <<" ";
                f_pcb << x(4) <<" ";
                f_pcb << x(5) <<endl;


                x1.emplace_back(x(0));
                x2.emplace_back(x(1));
                x3.emplace_back(x(2));
                x4.emplace_back(x(3));
                x5.emplace_back(x(4));
                x6.emplace_back(x(5));
            }
        }
    }
    f_pcb.close();

    size_t res_num = x1.size();
    int n = 10;
    double condition1 = 0.005;
    double condition2 = 0.05;
    static ofstream f_dev1;
    f_dev1.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/tcb_estimation_dev.txt");
    f_dev1<<std::fixed<<std::setprecision(6);
    double x_min = std::numeric_limits<double>::max();
    double x1_min = std::numeric_limits<double>::max();
    double x2_min = std::numeric_limits<double>::max();
    double x3_min = std::numeric_limits<double>::max();
    double x4_min = std::numeric_limits<double>::max();
    double x5_min = std::numeric_limits<double>::max();
    double x6_min = std::numeric_limits<double>::max();
    double x1_max = std::numeric_limits<double>::min();
    double x2_max = std::numeric_limits<double>::min();
    double x3_max = std::numeric_limits<double>::min();
    double x4_max = std::numeric_limits<double>::min();
    double x5_max = std::numeric_limits<double>::min();
    double x6_max = std::numeric_limits<double>::min();
    vector<double> best_res;
    best_res.resize(6);
    for (int k = 0; k < res_num-n-5; ++k)
    {
        vector<double> x1_(x1.begin()+k, x1.begin()+k+n);
        vector<double> x2_(x2.begin()+k, x2.begin()+k+n);
        vector<double> x3_(x3.begin()+k, x3.begin()+k+n);
        vector<double> x4_(x4.begin()+k, x4.begin()+k+n);
        vector<double> x5_(x5.begin()+k, x5.begin()+k+n);
        vector<double> x6_(x6.begin()+k, x6.begin()+k+n);
        double dev1 = std::sqrt(GetVariance(x1_));
        double dev2 = std::sqrt(GetVariance(x2_));
        double dev3 = std::sqrt(GetVariance(x3_));
        double dev4 = std::sqrt(GetVariance(x4_));
        double dev5 = std::sqrt(GetVariance(x5_));
        double dev6 = std::sqrt(GetVariance(x6_));
        f_dev1 <<dev1<<" ";
        f_dev1 <<dev2<<" ";
        f_dev1 <<dev3<<" ";
        f_dev1 <<dev4<<" ";
        f_dev1 <<dev5<<" ";
        f_dev1 <<dev6<<endl;
        if (dev1 < condition1 && dev2 < condition1 && dev3 < condition1 &&
            dev4 < condition2 && dev5 < condition2 && dev6 < condition2)
        {
            f_dev1<<"condition: "<<x1_[n-1]<<" "<<x2_[n-1]<<" "<<x3_[n-1]
                  <<" "<<x4_[n-1]<<" "<<x5_[n-1]<<" "<<x6_[n-1]<<endl;
        }

        x1_min = dev1 < x1_min ? dev1 : x1_min;
        x2_min = dev2 < x2_min ? dev2 : x2_min;
        x3_min = dev3 < x3_min ? dev3 : x3_min;
        x4_min = dev4 < x4_min ? dev4 : x4_min;
        x5_min = dev5 < x5_min ? dev5 : x5_min;
        x6_min = dev6 < x6_min ? dev6 : x6_min;
        x1_max = dev1 > x1_max ? dev1 : x1_max;
        x2_max = dev2 > x2_max ? dev2 : x2_max;
        x3_max = dev3 > x3_max ? dev3 : x3_max;
        x4_max = dev4 > x4_max ? dev4 : x4_max;
        x5_max = dev5 > x5_max ? dev5 : x5_max;
        x6_max = dev6 > x6_max ? dev6 : x6_max;
        double _x_min = x1_min + x2_min + x3_min +
                        x4_min + x5_min + x6_min;
        if (_x_min < x_min)
        {
            x_min = _x_min;
            gc(0) = x1_[n-1];
            gc(1) = x2_[n-1];
            gc(2) = x3_[n-1];
            best_res[0] = x1_[n-1];
            best_res[1] = x2_[n-1];
            best_res[2] = x3_[n-1];
            best_res[3] = x4_[n-1];
            best_res[4] = x5_[n-1];
            best_res[5] = x6_[n-1];
        }

    }
    f_dev1 <<"best result: "<<best_res[0]<<" "<<best_res[1]<<" "
            <<best_res[2]<<" "<<best_res[3]<<" "
            <<best_res[4]<<" "<<best_res[5]<<endl;
    f_dev1<<"min: "<<x1_min<<" "<<x2_min<<" "<<x3_min<<" "<<x4_min<<" "<<x5_min<<" "<<x6_min<<endl;
    f_dev1<<"max: "<<x1_max<<" "<<x2_max<<" "<<x3_max<<" "<<x4_max<<" "<<x5_max<<" "<<x6_max<<endl;
    f_dev1.close();
}

//求解yaw，gc两个分量和pcb3个分量
//测试该方法，估计pcb两个平移分量，该方法无法成功估计
void cSolver::solvePcb_test2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                               Eigen::Matrix3d Ryx_cam,
                               Eigen::Matrix3d Ryx_imu,Eigen::Matrix3d &Rci)
{
    static ofstream f_gamma4;
    f_gamma4.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/pcb.txt");
    f_gamma4<<std::fixed<<std::setprecision(6);
    f_gamma4 <<"solveOtherResult1\n";

    bool first_solve = true;
    Eigen::Vector3d gc_init = Eigen::Vector3d::Zero();
    Eigen::Vector3d gamma_init = Eigen::Vector3d::Zero();
    Eigen::Vector3d pcb_init = Eigen::Vector3d::Zero();
    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }
    Eigen::Matrix3d Ryx_imu_inv = Ryx_imu.transpose();
    Eigen::Matrix3d Ryx_cam_inv = Ryx_cam.transpose();

    vector<double> x1,x2,x3,x4,x5,x6,x7,x8,x9;
    double gcy = 9.7975;
    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        int num = calib_data[segmentId].size()-1;
        //Ax=b, x=[gcx,gcz,cos_gamma, sin_gamma, pcx, pcy, pcz]
        Eigen::MatrixXd tmp_G(num*3,7);
        tmp_G.setZero();
        Eigen::MatrixXd tmp_b(num*3,1);
        tmp_b.setZero();
        for (int motionId = 0; motionId < calib_data[segmentId].size()-1; ++motionId)
        {
            Eigen::Vector3d pc1 = calib_data[segmentId][motionId].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[segmentId][motionId].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[segmentId][motionId+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[segmentId][motionId].P;
            Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+1].P;
            Eigen::Vector3d v12_imu = calib_data[segmentId][motionId].V;
            Eigen::Vector3d t12_cam = calib_data[segmentId][motionId].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+1].Rwc2_cam;
            double dT12 = calib_data[segmentId][motionId].T;
            double dT23 = calib_data[segmentId][motionId+1].T;
            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+1].q21_cam.toRotationMatrix();


            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
            Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*dT23;//pcb

            //gamma
            Eigen::Vector3d pi_12 = Ryx_imu * t12_imu;
            Eigen::Matrix3d tmp_R12;
            tmp_R12<< pi_12(0), -pi_12(1),0,
                    pi_12(1),pi_12(0),0,
                    0,0,pi_12(2);
            Eigen::Matrix3d gamma_1 = -Rwc1_cam * Ryx_cam_inv * tmp_R12 * dT23;

            Eigen::Vector3d pi_23 = Ryx_imu * t23_imu;
            Eigen::Matrix3d tmp_R23;
            tmp_R23<< pi_23(0), -pi_23(1),0,
                    pi_23(1),pi_23(0),0,
                    0,0,pi_23(2);
            Eigen::Matrix3d gamma_2 = Rwc2_cam * Ryx_cam_inv * tmp_R23 * dT12;
            Eigen::Vector3d vi_12 = Ryx_imu * v12_imu;
            Eigen::Matrix3d tmp_v12;
            tmp_v12<< vi_12(0), -vi_12(1),0,
                    vi_12(1),vi_12(0),0,
                    0,0,vi_12(2);
            Eigen::Matrix3d gamma_3 = Rwc1_cam * Ryx_cam_inv * tmp_v12 * dT12 * dT23;
            Eigen::Matrix3d gamma = gamma_1 + gamma_2 + gamma_3;

//            计算权重
            if (!first_solve)
            {
                Eigen::Vector3d pred = beta*gc_init + gamma * gamma_init + phi * pcb_init;
                Eigen::Vector3d e = pred + lambda;
                double wi = exp(-e.norm()*200.0);
//                cout<<"e norm: "<<e.norm()<<endl;
                if (e.norm() > 0.01)
                    wi = 0.0;
                beta *= wi;
                gamma *=wi;
                lambda *= wi;
                phi *= wi;
            }

            tmp_G.block<3,1>(motionId*3,0) = beta.block<3,1>(0,0);
            tmp_G.block<3,1>(motionId*3,1) = beta.block<3,1>(0,2);
            tmp_G.block<3,2>(motionId*3,2) = gamma.block<3,2>(0,0);
            tmp_G.block<3,3>(motionId*3,4) = phi;
            tmp_b.block<3,1>(motionId*3,0) = -lambda -
                                                gamma.block<3,1>(0,2) -
                                                gcy*beta.block<3,1>(0,1);

            if (motionId > 15)
            {
                //求解
                Eigen::VectorXd x;
                x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);
                f_gamma4 << motionId <<" ";
                f_gamma4 << x(0)<<" ";
                f_gamma4 << x(1)<<" ";
                double alpha = atan2(x(3), x(2));
                f_gamma4 << alpha*RAD2DEG<<" ";
                Eigen::Matrix3d Rz;
                Rz << cos(alpha), -sin(alpha), 0,
                        sin(alpha), cos(alpha),0,
                        0,0,1;
                Eigen::Matrix3d Rcb = Ryx_cam_inv * Rz * Ryx_imu;
                Eigen::Vector3d euler = Rcb.inverse().eulerAngles(2,1,0);
                f_gamma4 << euler(0)*RAD2DEG<<" ";
                f_gamma4 << euler(1)*RAD2DEG<<" ";
                f_gamma4 << euler(2)*RAD2DEG<<" ";
                f_gamma4 << x(4)<<" ";
                f_gamma4 << x(5)<<" ";
                f_gamma4 << x(6)<<endl;

                gc_init(0) = x(0);
                gc_init(1) = gcy;
                gc_init(2) = x(1);
                gamma_init(0) = x(2);
                gamma_init(1) = x(3);
                gamma_init(2) = 1;
                pcb_init(0) = x(4);
                pcb_init(1) = x(5);
                pcb_init(2) = x(6);
                if (first_solve)
                    first_solve = false;

                x1.emplace_back(x(0));
                x2.emplace_back(x(1));
                x3.emplace_back(alpha*RAD2DEG);
                x4.emplace_back(x(4));
                x5.emplace_back(x(5));
                x6.emplace_back(x(6));
                x7.emplace_back(euler(0)*RAD2DEG);
                x8.emplace_back(euler(1)*RAD2DEG);
                x9.emplace_back(euler(2)*RAD2DEG);
            }
        }
    }
    f_gamma4.close();

    size_t res_num = x1.size();
    int n = 20;
    double condition1 = 0.006;
    double condition2 = 0.2;
    static ofstream f_dev2;
    f_dev2.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/stdDevPcb.txt");
    f_dev2<<std::fixed<<std::setprecision(6);
    double x_min = std::numeric_limits<double>::max();
    double x1_min = std::numeric_limits<double>::max();
    double x2_min = std::numeric_limits<double>::max();
    double x3_min = std::numeric_limits<double>::max();
    double x1_max = std::numeric_limits<double>::min();
    double x2_max = std::numeric_limits<double>::min();
    double x3_max = std::numeric_limits<double>::min();
    vector<double> best_res;
    best_res.resize(6);
    for (int k = 0; k < res_num-n-5; ++k)
    {
        vector<double> x1_(x1.begin()+k, x1.begin()+k+n);
        vector<double> x2_(x2.begin()+k, x2.begin()+k+n);
        vector<double> x3_(x3.begin()+k, x3.begin()+k+n);
        vector<double> x4_(x4.begin()+k, x4.begin()+k+n);
        vector<double> x5_(x5.begin()+k, x5.begin()+k+n);
        vector<double> x6_(x6.begin()+k, x6.begin()+k+n);
        vector<double> x7_(x7.begin()+k, x7.begin()+k+n);
        vector<double> x8_(x8.begin()+k, x8.begin()+k+n);
        vector<double> x9_(x8.begin()+k, x9.begin()+k+n);
        double dev1 = std::sqrt(GetVariance(x1_));
        double dev2 = std::sqrt(GetVariance(x2_));
        double dev3 = std::sqrt(GetVariance(x3_));
        f_dev2 <<dev1<<" ";
        f_dev2 <<dev2<<" ";
        f_dev2 <<dev3<<endl;
        if (dev1 < condition1 && dev2 < condition1 && dev3 < condition2)
            f_dev2<<"best result: "<<x1_[n-1]<<" "<<x2_[n-1]<<" "<<x3_[n-1]<<" "
                <<x4_[n-1]<<" "<<x5_[n-1]<<x6_[n-1]<<endl;
        if (dev1 < x1_min)
            x1_min = dev1;
        if (dev2 < x2_min)
            x2_min = dev2;
        if (dev3 < x3_min)
            x3_min = dev3;
        if (dev1 > x1_max)
            x1_max = dev1;
        if (dev2 > x2_max)
            x2_max = dev2;
        if (dev3 > x3_max)
            x3_max = dev3;
        double _x_min = x1_min + x2_min + x3_min;
        if (_x_min < x_min)
        {
            x_min = _x_min;
            best_res[0] = x4_[n-1];
            best_res[1] = x5_[n-1];
            best_res[2] = x6_[n-1];
            best_res[3] = x7_[n-1];
            best_res[4] = x8_[n-1];
            best_res[5] = x9_[n-1];
        }
    }
    f_dev2<<"min: "<<x1_min<<" "<<x2_min<<" "<<x3_min<<endl;
    f_dev2<<"max: "<<x1_max<<" "<<x2_max<<" "<<x3_max<<endl;
    f_dev2 <<"pcb best result: "<<best_res[0]<<" "<<best_res[1]<<" "
           <<best_res[2]<<endl;
    f_dev2 <<"Ric best result: "<<best_res[3]<<" "<<best_res[4]<<" "
           <<best_res[5]<<endl;

    f_dev2.close();
}


//设Rcb已知，求解gc和pcb的两个分量,平面运动
void cSolver::solvePcb_test3(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                                     const Eigen::Matrix3d Rcb,
                                     Eigen::Vector3d &gc)
{
    static ofstream f_pcb;
    f_pcb.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/tcb_estimation.txt");
    f_pcb<<std::fixed<<std::setprecision(6);
    f_pcb<<"solveOtherResult_test3 zed_tci = [0.0224, -0.0112, -0.0165]\n";
    f_pcb<<"solveOtherResult_test3 oxts_tci = [-0.081, 1.03, -1.45]\n";
    f_pcb <<"gcx, gcy, gcz, pcb_x, pcb_y, pcb_z\n";

    bool first_solve = true;
    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }
    Eigen::Vector3d gc_init = Eigen::Vector3d::Zero();
    Eigen::Vector3d pcb_init = Eigen::Vector3d::Zero();
    vector<double> x1,x2,x3,x4,x5,x6;
    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        f_pcb <<"---------\n";
        int num = calib_data[segmentId].size()-1;
        Eigen::MatrixXd tmp_G(num*3,5);
        tmp_G.setZero();
        Eigen::MatrixXd tmp_b(num*3,1);
        tmp_b.setZero();
        for (int motionId = 0; motionId < calib_data[segmentId].size()-1; ++motionId)
        {
            Eigen::Vector3d pc1 = calib_data[segmentId][motionId].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[segmentId][motionId].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[segmentId][motionId+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[segmentId][motionId].P;
            Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+1].P;
            Eigen::Vector3d v12_imu = calib_data[segmentId][motionId].V;
            Eigen::Vector3d t12_cam = calib_data[segmentId][motionId].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+1].Rwc2_cam;
            double dT12 = calib_data[segmentId][motionId].T;
            double dT23 = calib_data[segmentId][motionId+1].T;
            if (dT12 != calib_data[segmentId][motionId].sum_dt)
                cout<<"sum_dt: "<<calib_data[segmentId][motionId].sum_dt<<", dT12: "<<dT12<<endl;
            if (dT23 != calib_data[segmentId][motionId+1].sum_dt)
                cout<<"sum_dt: "<<calib_data[segmentId][motionId+1].sum_dt<<", dT23: "<<dT23<<endl;
//            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId].q21_cam.toRotationMatrix();
//            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
            Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*dT23;//pcb

            //gamma
            Eigen::Vector3d gamma_1 =  Rwc1_cam * Rcb * t12_imu * dT23;
            Eigen::Vector3d gamma_2 =  -Rwc2_cam * Rcb * t23_imu * dT12;
            Eigen::Vector3d gamma_3 = -Rwc1_cam * Rcb * v12_imu * dT12 * dT23;
            Eigen::Vector3d gamma = gamma_1 + gamma_2 + gamma_3;
//            Eigen::Vector3d test_b = -lambda + gamma;
            Eigen::Vector3d test_b = -lambda + gamma - 1.03 * phi.block<3,1>(0,1);

//            计算权重
            if (!first_solve)
            {
                Eigen::Vector3d pred = beta * gc_init + phi * pcb_init;
                Eigen::Vector3d e = pred + lambda - gamma;
                double wi = exp(-e.norm()*100.0);
                if (e.norm() > 0.05)
                    wi = 0.0;
                beta *= wi;
                phi *= wi;
                test_b *= wi;
                f_pcb <<"e.norm="<<e.norm()<<" ";
                f_pcb <<"wi="<<wi<<" ";
            }
            tmp_G.block<3,3>(motionId*3,0) = beta;
//            tmp_G.block<3,3>(motionId*3,3) = phi;
            tmp_G.block<3,1>(motionId*3,3) = phi.leftCols(1);
            tmp_G.block<3,1>(motionId*3,4) = phi.rightCols(1);
            tmp_b.block<3,1>(motionId*3,0) = test_b;

            if (motionId > 15)
            {
                Eigen::MatrixXd G(motionId*3+3,5);
                G = tmp_G.topRows(motionId*3+3);
                Eigen::VectorXd b(motionId*3+3);
                b = tmp_b.topRows(motionId*3+3);
                //求解
                Eigen::VectorXd x;
//                x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);
                x = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                gc_init(0) = x(0);
                gc_init(1) = x(1);
                gc_init(2) = x(2);
                pcb_init(0) = x(3);
                pcb_init(1) = 1.03;
                pcb_init(2) = x(4);
//                pcb_init(2) = x(5);
                f_pcb << x(0) <<" ";
                f_pcb << x(1) <<" ";
                f_pcb << x(2) <<" ";
                f_pcb << x(3) <<" ";
                f_pcb << x(4) <<endl;
//                f_pcb << x(5) <<endl;
                if (first_solve)
                    first_solve = false;

                x1.emplace_back(x(0));
                x2.emplace_back(x(1));
                x3.emplace_back(x(2));
                x4.emplace_back(x(3));
                x5.emplace_back(x(4));
//                x6.emplace_back(x(5));
            }
        }
    }
    f_pcb.close();

    size_t res_num = x1.size();
    int n = 10;
    double condition1 = 0.005;
    double condition2 = 0.005;
    static ofstream f_dev1;
    f_dev1.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/tcb_StaDev.txt");
    f_dev1<<std::fixed<<std::setprecision(6);
    double x_min = std::numeric_limits<double>::max();
    double x1_min = std::numeric_limits<double>::max();
    double x2_min = std::numeric_limits<double>::max();
    double x3_min = std::numeric_limits<double>::max();
    double x4_min = std::numeric_limits<double>::max();
    double x5_min = std::numeric_limits<double>::max();
    double x6_min = std::numeric_limits<double>::max();
    double x1_max = std::numeric_limits<double>::min();
    double x2_max = std::numeric_limits<double>::min();
    double x3_max = std::numeric_limits<double>::min();
    double x4_max = std::numeric_limits<double>::min();
    double x5_max = std::numeric_limits<double>::min();
    double x6_max = std::numeric_limits<double>::min();
    vector<double> best_res;
    best_res.resize(6);
    for (int k = 0; k < res_num-n-5; ++k)
    {
        vector<double> x1_(x1.begin()+k, x1.begin()+k+n);
        vector<double> x2_(x2.begin()+k, x2.begin()+k+n);
        vector<double> x3_(x3.begin()+k, x3.begin()+k+n);
        vector<double> x4_(x4.begin()+k, x4.begin()+k+n);
        vector<double> x5_(x5.begin()+k, x5.begin()+k+n);
        vector<double> x6_(x6.begin()+k, x6.begin()+k+n);
        double dev1 = std::sqrt(GetVariance(x1_));
        double dev2 = std::sqrt(GetVariance(x2_));
        double dev3 = std::sqrt(GetVariance(x3_));
        double dev4 = std::sqrt(GetVariance(x4_));
        double dev5 = std::sqrt(GetVariance(x5_));
        double dev6 = std::sqrt(GetVariance(x6_));
        f_dev1 <<dev1<<" ";
        f_dev1 <<dev2<<" ";
        f_dev1 <<dev3<<" ";
        f_dev1 <<dev4<<" ";
        f_dev1 <<dev5<<" ";
        f_dev1 <<dev6<<endl;
        if (dev1 < condition1 && dev2 < condition1 && dev3 < condition1 &&
            dev4 < condition2 && dev5 < condition2 && dev6 < condition2)
        {
            f_dev1<<"condition: "<<x1_[n-1]<<" "<<x2_[n-1]<<" "<<x3_[n-1]
                  <<" "<<x4_[n-1]<<" "<<x5_[n-1]<<" "<<x6_[n-1]<<endl;
        }

        x1_min = dev1 < x1_min ? dev1 : x1_min;
        x2_min = dev2 < x2_min ? dev2 : x2_min;
        x3_min = dev3 < x3_min ? dev3 : x3_min;
        x4_min = dev4 < x4_min ? dev4 : x4_min;
        x5_min = dev5 < x5_min ? dev5 : x5_min;
        x6_min = dev6 < x6_min ? dev6 : x6_min;
        x1_max = dev1 > x1_max ? dev1 : x1_max;
        x2_max = dev2 > x2_max ? dev2 : x2_max;
        x3_max = dev3 > x3_max ? dev3 : x3_max;
        x4_max = dev4 > x4_max ? dev4 : x4_max;
        x5_max = dev5 > x5_max ? dev5 : x5_max;
        x6_max = dev6 > x6_max ? dev6 : x6_max;
        double _x_min = x1_min + x2_min + x3_min +
                        x4_min + x5_min + x6_min;
        if (_x_min < x_min)
        {
            x_min = _x_min;
            gc(0) = x1_[n-1];
            gc(1) = x2_[n-1];
            gc(2) = x3_[n-1];
            best_res[0] = x1_[n-1];
            best_res[1] = x2_[n-1];
            best_res[2] = x3_[n-1];
            best_res[3] = x4_[n-1];
            best_res[4] = x5_[n-1];
            best_res[5] = x6_[n-1];
        }

    }
    f_dev1 <<"best result: "<<best_res[0]<<" "<<best_res[1]<<" "
           <<best_res[2]<<" "<<best_res[3]<<" "
           <<best_res[4]<<" "<<best_res[5]<<endl;
    f_dev1<<"min: "<<x1_min<<" "<<x2_min<<" "<<x3_min<<" "<<x4_min<<" "<<x5_min<<" "<<x6_min<<endl;
    f_dev1<<"max: "<<x1_max<<" "<<x2_max<<" "<<x3_max<<" "<<x4_max<<" "<<x5_max<<" "<<x6_max<<endl;
    f_dev1.close();
}

//求解bias，优化重力，平移,非滑窗
void cSolver::RefinePcb_test3(std::vector<std::vector<data_selection::sync_data>> &calib_data,
                                 Eigen::Matrix3d Rcb,
                                 Eigen::Vector3d &gc)
{
    Eigen::Vector3d theta_xy_last;
    Eigen::Vector3d bias_last, pcb_last;
    theta_xy_last.setZero();
    bias_last.setZero();
    pcb_last.setZero();
    double err_min = std::numeric_limits<double>::max();
    double best_x1, best_x2, best_x3;
    bool first_solve = true;

    static ofstream f_pcb3;
    f_pcb3.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/refine_tcb_test3.txt");
    f_pcb3<<std::fixed<<std::setprecision(6);
    f_pcb3<<"非滑窗\n";
    f_pcb3<<"RefineResult_test1 zed_tci = [0.0224, -0.0112, -0.0165]\n";
    f_pcb3<<"RefineResult_test1 oxts_tci = [-0.081, 1.03, -1.45]\n";
    f_pcb3<<"# theta_x theta_y bias_x bias_y bias_z pcb_x pcb_y pcb_z\n";

    Eigen::Vector3d GI = Eigen::Vector3d::Zero();
    GI(2) = -9.810;
    Eigen::Matrix3d skewGI;
    skewGI << 0, -GI(2), GI(1),
            GI(2), 0, -GI(0),
            -GI(1), GI(0), 0;
    Eigen::Vector3d GIxgc = GI.cross(gc);
    Eigen::Vector3d vhat = GIxgc / GIxgc.norm();
    double theta = std::atan2(GIxgc.norm(),GI.dot(gc));
    f_pcb3 << "theta = "<<theta<<endl;
    Eigen::Matrix3d RWI = Sophus::SO3d::exp(vhat*theta).matrix();
    int N = calib_data[0].size() - 1;
    Eigen::MatrixXd A(3*N,8);
    A.setZero();
    Eigen::MatrixXd b(3*N,1);
    b.setZero();
    for (int i = 0; i < calib_data[0].size()-1; ++i)
    {
        Eigen::Vector3d pc1 = calib_data[0][i].twc1_cam;
        Eigen::Vector3d pc2 = calib_data[0][i].twc2_cam;
        Eigen::Vector3d pc3 = calib_data[0][i+1].twc2_cam;

        Eigen::Vector3d t12_imu = calib_data[0][i].P;
        Eigen::Vector3d t23_imu = calib_data[0][i+1].P;
        Eigen::Vector3d v12_imu = calib_data[0][i].V;
        Eigen::Vector3d t12_cam = calib_data[0][i].cam_t12;
        Eigen::Vector3d t23_cam = calib_data[0][i+1].cam_t12;

        Eigen::Matrix3d Rwc1_cam = calib_data[0][i].Rwc1_cam;
        Eigen::Matrix3d Rwc2_cam = calib_data[0][i].Rwc2_cam;
        Eigen::Matrix3d Rwc3_cam = calib_data[0][i+1].Rwc2_cam;
        double dT12 = calib_data[0][i].T;
        double dT23 = calib_data[0][i+1].T;
        Eigen::Matrix3d R21_cam = calib_data[0][i].q21_cam.toRotationMatrix();
        Eigen::Matrix3d R32_cam = calib_data[0][i+1].q21_cam.toRotationMatrix();

        Eigen::Matrix3d Jpba12 = calib_data[0][i].jacobian_.block<3,3>(0,9);
        Eigen::Matrix3d Jpba23 = calib_data[0][i+1].jacobian_.block<3,3>(0,9);
        Eigen::Matrix3d Jvba12 = calib_data[0][i].jacobian_.block<3,3>(6,9);


        Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
        Eigen::Matrix3d beta = -0.5*(dT12*dT12*dT23 + dT12*dT23*dT23)*RWI*skewGI;//gc
        Eigen::Matrix3d bias = Rwc2_cam*Rcb*Jpba23*dT12 -
                               Rwc1_cam*Rcb*Jpba12*dT23 +
                               Rwc1_cam*Rcb*Jvba12*dT12*dT23;//bias
        Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                              (Rwc1_cam - Rwc2_cam)*dT23;//pcb

        //gamma
        Eigen::Vector3d gamma_1 =  Rwc1_cam * Rcb * t12_imu * dT23;
        Eigen::Vector3d gamma_2 =  -Rwc2_cam * Rcb * t23_imu * dT12;
        Eigen::Vector3d gamma_3 = -Rwc1_cam * Rcb * v12_imu * dT12 * dT23;
        Eigen::Vector3d gamma_4 = -0.5*RWI*GI*(dT12*dT12*dT23 + dT12*dT23*dT23);
        Eigen::Vector3d gamma = gamma_1 + gamma_2 + gamma_3 + gamma_4;

        if (!first_solve)
        {
            Eigen::Vector3d pred = beta*theta_xy_last +
                                    bias * bias_last +
                                    phi * pcb_last;
            Eigen::Vector3d e = pred + lambda - gamma;
            double wi = std::exp(-e.norm()*10);
            if (e.norm() > 1.2)
                wi = 0.0;
            f_pcb3 <<"e.norm="<<e.norm()<<" ";
            f_pcb3 <<"wi="<<wi<<" ";
            beta *=wi;
            bias *=wi;
            phi *= wi;
            lambda *= wi;
//            gamma *=wi;
        }
        A.block<3,2>(3*i,0) = beta.block<3,2>(0,0);
        A.block<3,3>(3*i,2) = bias;
        A.block<3,3>(3*i,5) = phi;
        Eigen::Vector3d test_b = -lambda + gamma;
        b.block<3,1>(3*i,0) = test_b;
        if (i > 15)
        {
            Eigen::MatrixXd A_(i*3+3,8);
            A_ = A.topRows(i*3+3);
            Eigen::VectorXd b_(i*3+3);
            b_ = b.topRows(i*3+3);

            Eigen::VectorXd x;
            x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
//            x = A_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_);
//            theta_xy_last = x.head(2);
            theta_xy_last(0) = x(0);
            theta_xy_last(1) = x(1);
            theta_xy_last(2) = 0;
            bias_last = x.block<3,1>(2,0);
            pcb_last = x.tail(3);

            f_pcb3 << x(0)<<" ";
            f_pcb3 << x(1)<<" ";
            f_pcb3 << x(2)<<" ";
            f_pcb3 << x(3)<<" ";
            f_pcb3 << x(4)<<" ";
            f_pcb3 << x(5)<<" ";
            f_pcb3 << x(6)<<" ";
            f_pcb3 << x(7)<<endl;
            if (first_solve)
                first_solve = false;
        }



        //计算误差

    }
    f_pcb3.close();

}
//优化重力，平移 非滑窗
void cSolver::RefinePcb(std::vector<std::vector<data_selection::sync_data>> &calib_data,
                           Eigen::Matrix3d Rcb,
                           Eigen::Vector3d &gc)
{
    double s_last;
    Eigen::Vector3d theta_xy_last;
    Eigen::Vector3d bias_last, pcb_last;

    theta_xy_last.setZero();
    bias_last.setZero();
    pcb_last.setZero();
    vector<double> x1,x2,x3,x4,x5,x6;
    double time0 = 0.0;

    static ofstream f_pcb3;
    f_pcb3.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/refine_tcb.txt");
    f_pcb3<<std::fixed<<std::setprecision(6);
    f_pcb3<<"非滑窗 RefineResult zed_tci = [0.0224, -0.0112, -0.0165]\n";
    f_pcb3<<"RefineResult oxts_tci = [-0.081, 1.03, -1.45]\n";
    f_pcb3<<"# theta_x theta_y pcb_x pcb_y pcb_z\n";

    Eigen::Vector3d GI = Eigen::Vector3d::Zero();
    GI(2) = -9.810;
    Eigen::Matrix3d skewGI;
    skewGI << 0, -GI(2), GI(1),
            GI(2), 0, -GI(0),
            -GI(1), GI(0), 0;
    Eigen::Vector3d GIxgc = GI.cross(gc);
    Eigen::Vector3d vhat = GIxgc / GIxgc.norm();
    double theta = std::atan2(GIxgc.norm(),GI.dot(gc));
    double theta2 = std::acos(GI.dot(gc));
    f_pcb3 << "theta = "<<theta*RAD2DEG<<endl;
    f_pcb3 << "theta2 = "<<theta2*RAD2DEG<<endl;
    Eigen::Matrix3d RWI = Sophus::SO3d::exp(vhat*theta).matrix();
    int N = calib_data[0].size() - 1;

//    Eigen::MatrixXd A(3*N,6);//s
    Eigen::MatrixXd A(3*N,5);
    A.setZero();
    Eigen::MatrixXd b(3*N,1);
    b.setZero();
    bool first_solve = true;
    for (int i = 0; i < calib_data[0].size()-1; ++i)
    {
        Eigen::Vector3d pc1 = calib_data[0][i].twc1_cam;
        Eigen::Vector3d pc2 = calib_data[0][i].twc2_cam;
        Eigen::Vector3d pc3 = calib_data[0][i+1].twc2_cam;

        Eigen::Vector3d t12_imu = calib_data[0][i].P;
        Eigen::Vector3d t23_imu = calib_data[0][i+1].P;
        Eigen::Vector3d v12_imu = calib_data[0][i].V;
        Eigen::Vector3d t12_cam = calib_data[0][i].cam_t12;
        Eigen::Vector3d t23_cam = calib_data[0][i+1].cam_t12;

        Eigen::Matrix3d Rwc1_cam = calib_data[0][i].Rwc1_cam;
        Eigen::Matrix3d Rwc2_cam = calib_data[0][i].Rwc2_cam;
        Eigen::Matrix3d Rwc3_cam = calib_data[0][i+1].Rwc2_cam;
        double dT12 = calib_data[0][i].T;
        double dT23 = calib_data[0][i+1].T;
        if (dT12 != calib_data[0][i].sum_dt)
            cout<<"sum_dt: "<<calib_data[0][i].sum_dt<<", dT12: "<<dT12<<endl;
        if (dT23 != calib_data[0][i+1].sum_dt)
            cout<<"sum_dt: "<<calib_data[0][i+1].sum_dt<<", dT23: "<<dT23<<endl;
//        Eigen::Matrix3d R21_cam = calib_data[0][i].q21_cam.toRotationMatrix();
//        Eigen::Matrix3d R32_cam = calib_data[0][i+1].q21_cam.toRotationMatrix();

//        Eigen::Matrix3d Jpba12 = calib_data[0][i].jacobian_.block<3,3>(0,9);
//        Eigen::Matrix3d Jpba23 = calib_data[0][i+1].jacobian_.block<3,3>(0,9);
//        Eigen::Matrix3d Jvba12 = calib_data[0][i].jacobian_.block<3,3>(6,9);


        Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
        Eigen::Matrix3d beta = -0.5*(dT12*dT12*dT23 + dT12*dT23*dT23)*RWI*skewGI;//theta_xy
        Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                              (Rwc1_cam - Rwc2_cam)*dT23;//pcb

        //gamma
        Eigen::Vector3d gamma_1 =  Rwc1_cam * Rcb * t12_imu * dT23;
        Eigen::Vector3d gamma_2 =  -Rwc2_cam * Rcb * t23_imu * dT12;
        Eigen::Vector3d gamma_3 = -Rwc1_cam * Rcb * v12_imu * dT12 * dT23;
        Eigen::Vector3d gamma_4 = -0.5*RWI*GI*(dT12*dT12*dT23 + dT12*dT23*dT23);
        Eigen::Vector3d gamma = gamma_1 + gamma_2 + gamma_3 + gamma_4;
//        Eigen::Vector3d test_b = gamma;

//        A.block<3,1>(3*i,0) = lambda;


        if (!first_solve)
        {
//            Eigen::Vector3d pred = s_last*lambda + beta.block<3,2>(0,0)*theta_xy_last +
//                                    phi * pcb_last;
            Eigen::Vector3d pred = beta*theta_xy_last +
                                   phi * pcb_last;
            Eigen::Vector3d e = pred + lambda - gamma;
            double wi = std::exp(-e.norm()*K2);
//            if (e.norm() > 1.2)
//                wi = 0.0;

            f_pcb3 <<"e.norm="<<e.norm()<<" ";
            f_pcb3 <<"wi="<<wi<<" ";
            beta *=wi;
            phi *= wi;
            lambda *= wi;
            gamma *=wi;
        }

        A.block<3,2>(3*i,0) = beta.block<3,2>(0,0);
        A.block<3,3>(3*i,2) = phi;
        Eigen::Vector3d test_b = -lambda + gamma;
        b.block<3,1>(3*i,0) = test_b;
        if (i > 15)
        {
            if (first_solve)
            {
                first_solve = false;
                time0 = calib_data[0][i].startTime;
            }
            Eigen::MatrixXd A_(i*3+3,5);
            A_ = A.topRows(i*3+3);
            Eigen::VectorXd b_(i*3+3);
            b_ = b.topRows(i*3+3);
//
            Eigen::VectorXd x;
            x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
//            x = A_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_);
//            s_last = x(0);
//            theta_xy_last = x.block<2,1>(1,0);
//            theta_xy_last = x.head(2);
//            pcb_last = x.tail(3);
            theta_xy_last(0) = x(0);
            theta_xy_last(1) = x(1);
            theta_xy_last(2) = 0;
            pcb_last(0) = x(2);
            pcb_last(1) = x(3);
            pcb_last(2) = x(4);
            x1.emplace_back(x(0));
            x2.emplace_back(x(1));
            x3.emplace_back(x(2));
            x4.emplace_back(x(3));
            x5.emplace_back(x(4));
//            x6.emplace_back(x(5));
            double time = calib_data[0][i].startTime-time0;
//                double time = calib_data[segmentId][motionId].startTime;
            f_pcb3 <<time<<" ";
            f_pcb3 << x(0)<<" ";
            f_pcb3 << x(1)<<" ";
            f_pcb3 << x(2)<<" ";
            f_pcb3 << x(3)<<" ";
            f_pcb3 << x(4)<<endl;
//            f_pcb3 << x(5)<<endl;


//            Eigen::Vector3d theta_xy;
//            theta_xy.setZero();
//            theta_xy(0) = x(0);
//            theta_xy(1) = x(1);
//            Eigen::Matrix3d Rwi_ = Sophus::SO3d::exp(theta_xy).matrix();
//            Eigen::Vector3d gw_refined = Rwi_*GI;
//            cout<<"gw norm: "<<gc.norm()<<endl;
//            cout<<"gw refine: "<<gw_refined.transpose()<<endl;
        }
    }
    f_pcb3.close();

    size_t res_num = x1.size();
    int n = 10;
    double condition1 = 0.005;
    double condition2 = 0.05;
    static ofstream f_dev2;
    f_dev2.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/refine_tcb_dev.txt");
    f_dev2<<std::fixed<<std::setprecision(6);
    double x_min = std::numeric_limits<double>::max();
    double x1_min = std::numeric_limits<double>::max();
    double x2_min = std::numeric_limits<double>::max();
    double x3_min = std::numeric_limits<double>::max();
    double x4_min = std::numeric_limits<double>::max();
    double x5_min = std::numeric_limits<double>::max();
    double x1_max = std::numeric_limits<double>::min();
    double x2_max = std::numeric_limits<double>::min();
    double x3_max = std::numeric_limits<double>::min();
    double x4_max = std::numeric_limits<double>::min();
    double x5_max = std::numeric_limits<double>::min();
    vector<double> best_res;
    best_res.resize(5);
    for (int k = 0; k < res_num-n-5; ++k)
    {
        vector<double> x1_(x1.begin()+k, x1.begin()+k+n);
        vector<double> x2_(x2.begin()+k, x2.begin()+k+n);
        vector<double> x3_(x3.begin()+k, x3.begin()+k+n);
        vector<double> x4_(x4.begin()+k, x4.begin()+k+n);
        vector<double> x5_(x5.begin()+k, x5.begin()+k+n);
        double dev1 = std::sqrt(GetVariance(x1_));
        double dev2 = std::sqrt(GetVariance(x2_));
        double dev3 = std::sqrt(GetVariance(x3_));
        double dev4 = std::sqrt(GetVariance(x4_));
        double dev5 = std::sqrt(GetVariance(x5_));
        f_dev2 <<dev1<<" ";
        f_dev2 <<dev2<<" ";
        f_dev2 <<dev3<<" ";
        f_dev2 <<dev4<<" ";
        f_dev2 <<dev5<<endl;
//        if (dev1 < condition1 && dev2 < condition1 && dev3 < condition2 &&
//            dev4 < condition2 && dev5 < condition2)
//            f_dev2<<"best result: "<<x1_[n-1]<<" "<<x2_[n-1]<<" "<<x3_[n-1]
//                  <<" "<<x4_[n-1]<<" "<<x5_[n-1]<<" "<<endl;
        x1_min = dev1 < x1_min ? dev1 : x1_min;
        x2_min = dev2 < x2_min ? dev2 : x2_min;
        x3_min = dev3 < x3_min ? dev3 : x3_min;
        x4_min = dev4 < x4_min ? dev4 : x4_min;
        x5_min = dev5 < x5_min ? dev5 : x5_min;
        x1_max = dev1 > x1_max ? dev1 : x1_max;
        x2_max = dev2 > x2_max ? dev2 : x2_max;
        x3_max = dev3 > x3_max ? dev3 : x3_max;
        x4_max = dev4 > x4_max ? dev4 : x4_max;
        x5_max = dev5 > x5_max ? dev5 : x5_max;
        double _x_min = x1_min + x2_min + x3_min +
                        x4_min + x5_min;
        if (_x_min < x_min)
        {
            x_min = _x_min;
            best_res[0] = x1_[n-1];
            best_res[1] = x2_[n-1];
            best_res[2] = x3_[n-1];
            best_res[3] = x4_[n-1];
            best_res[4] = x5_[n-1];
        }

    }
    f_dev2 <<"pcb best: "<<best_res[2]<<" "
           <<best_res[3]<<" "<<best_res[4]<<endl;
    f_dev2<<"min: "<<x1_min<<" "<<x2_min<<" "<<x3_min<<" "<<x4_min<<" "<<x5_min<<endl;
    f_dev2<<"max: "<<x1_max<<" "<<x2_max<<" "<<x3_max<<" "<<x4_max<<" "<<x5_max<<endl;
    f_dev2.close();
}

//求解Rcb
//yaw的判断应该使用收敛条件，这里使用的误差最小来判断，不太合适
//判断yaw和gc的两个分量的标准差
void cSolver::solveRcb(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                               Eigen::Matrix3d Ryx_cam,
                               Eigen::Matrix3d Ryx_imu,Eigen::Matrix3d &Rci)
{
    static ofstream f_gamma;
    f_gamma.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/Rcb_estimation.txt");
    f_gamma<<std::fixed<<std::setprecision(6);
    f_gamma <<"SolveOtherResult\n";
    static ofstream f_cam;
    f_cam.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/cam.txt");
    f_cam<<std::fixed<<std::setprecision(6);

    bool first_solve = true;
    Eigen::Vector2d gc_init = Eigen::Vector2d::Zero();
    Eigen::Vector2d gamma_init = Eigen::Vector2d::Zero();
    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }
    Eigen::Matrix3d Ryx_imu_inv = Ryx_imu.transpose();
    Eigen::Matrix3d Ryx_cam_inv = Ryx_cam.transpose();

    vector<double> x1,x2,x3,x4,x5,x6,x_time;
    double time0 = 0.0;
    cout<<"yaw,pitch,roll: "<<Ryx_cam.eulerAngles(2,1,0).transpose()*RAD2DEG<<endl;
    double beta = Ryx_cam.eulerAngles(2,1,0)(1);
    double alpha = Ryx_cam.eulerAngles(2,1,0)(2);
    double gcy = fabs(cos(beta)*cos(M_PI_2 + alpha)*9.81);
    cout<<"gcy: "<<gcy<<endl;

    //x = [gc,cos,sin]
//    double gcy = 9.7975;
    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        Eigen::Vector3d tcb;
        tcb.setZero();
        //zed_imu
//        tcb(0) = 0.02241856;
//        tcb(1) = -0.01121906;
//        tcb(2) = -0.01653902;
        //oxts_imu
//        tcb(0) = -0.021;
//        tcb(1) = 1.012;
//        tcb(2) = -1.1163;
        int num = calib_data[segmentId].size()-1;
        Eigen::MatrixXd tmp_G(num*3,4);
        tmp_G.setZero();
        Eigen::MatrixXd tmp_b(num*3,1);
        tmp_b.setZero();
        for (int motionId = 0; motionId < calib_data[segmentId].size()-1; ++motionId)
        {
            Eigen::Vector3d pc1 = calib_data[segmentId][motionId].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[segmentId][motionId].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[segmentId][motionId+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[segmentId][motionId].P;
            Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+1].P;
            Eigen::Vector3d v12_imu = calib_data[segmentId][motionId].V;
            Eigen::Vector3d t12_cam = calib_data[segmentId][motionId].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+1].cam_t12;
            f_cam << t12_cam(0)<<" ";
            f_cam << t12_cam(1)<<" ";
            f_cam << t12_cam(2)<<endl;

            Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+1].Rwc2_cam;
            double dT12 = calib_data[segmentId][motionId].T;
            double dT23 = calib_data[segmentId][motionId+1].T;
            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+1].q21_cam.toRotationMatrix();


            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
            Eigen::Vector3d phi = (Rwc2_cam - Rwc3_cam)*tcb*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*tcb*dT23;//pcb

            //gamma
            Eigen::Vector3d pi_12 = Ryx_imu * t12_imu;
            Eigen::Matrix3d tmp_R12;
            tmp_R12<< pi_12(0), -pi_12(1),0,
                    pi_12(1),pi_12(0),0,
                    0,0,pi_12(2);
            Eigen::Matrix3d gamma_1 = -Rwc1_cam * Ryx_cam_inv * tmp_R12 * dT23;

            Eigen::Vector3d pi_23 = Ryx_imu * t23_imu;
            Eigen::Matrix3d tmp_R23;
            tmp_R23<< pi_23(0), -pi_23(1),0,
                    pi_23(1),pi_23(0),0,
                    0,0,pi_23(2);
            Eigen::Matrix3d gamma_2 = Rwc2_cam * Ryx_cam_inv * tmp_R23 * dT12;
            Eigen::Vector3d vi_12 = Ryx_imu * v12_imu;
            Eigen::Matrix3d tmp_v12;
            tmp_v12<< vi_12(0), -vi_12(1),0,
                    vi_12(1),vi_12(0),0,
                    0,0,vi_12(2);
            Eigen::Matrix3d gamma_3 = Rwc1_cam * Ryx_cam_inv * tmp_v12 * dT12 * dT23;
            Eigen::Matrix3d gamma = gamma_1 + gamma_2 + gamma_3;

//            计算权重
            if (!first_solve)
            {
                Eigen::Vector3d tmp_gc;
                tmp_gc(0) = gc_init(0);
                tmp_gc(1) = gcy;
                tmp_gc(2) = gc_init(1);
                Eigen::Vector3d tmp_gamma;
                tmp_gamma(0) = gamma_init(0);
                tmp_gamma(1) = gamma_init(1);
                tmp_gamma(2) = 1;
                Eigen::Vector3d pred = beta*tmp_gc + gamma * tmp_gamma;
                Eigen::Vector3d e = pred + lambda - phi;
                double wi = exp(-e.norm()*300.0);
                if (e.norm() > 0.05)
                    wi = 0.0;
//                f_gamma <<"e.norm="<<e.norm()<<" ";
//                f_gamma <<"wi="<<wi<<" ";
                beta *= wi;
//                beta.block<3,1>(0,0) *= wi;
//                beta.block<3,1>(0,2) *= wi;
                gamma.block<3,2>(0,0) *=wi;
//                gamma *= wi;
//                lambda *= wi;
//                phi *= wi;
            }

            tmp_G.block<3,1>(motionId*3,0) = beta.block<3,1>(0,0);
            tmp_G.block<3,1>(motionId*3,1) = beta.block<3,1>(0,2);
            tmp_G.block<3,2>(motionId*3,2) = gamma.block<3,2>(0,0);
            tmp_b.block<3,1>(motionId*3,0) = -lambda - phi
                            - gamma.block<3,1>(0,2)
                            - gcy*beta.block<3,1>(0,1);

            if (motionId > 15)
            {
                if (first_solve)
                {
                    time0 = calib_data[segmentId][motionId].startTime;
                    first_solve = false;
                }
                //求解
                Eigen::VectorXd x;
                x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);
                gamma_init(0) = x(2);
                gamma_init(1) = x(3);
                double alpha = atan2(x(3), x(2));
                double time = calib_data[segmentId][motionId].startTime-time0;
//                double time = calib_data[segmentId][motionId].startTime;
                f_gamma <<time<<" ";
//                f_gamma << motionId <<" ";
                f_gamma << x(0)<<" ";
                f_gamma << x(1)<<" ";
                f_gamma << alpha*RAD2DEG<<" ";
                Eigen::Matrix3d Rz;
                Rz << cos(alpha), -sin(alpha), 0,
                    sin(alpha), cos(alpha),0,
                    0,0,1;
                Eigen::Matrix3d Rcb = Ryx_cam_inv * Rz * Ryx_imu;
                Eigen::Vector3d euler = Rcb.inverse().eulerAngles(2,1,0);
//                Eigen::Vector3d euler1 = Rcb.eulerAngles(2,1,0);
                Eigen::Matrix3d R_test[1];
                R_test[0] = Rcb.inverse();
                double roll[2], pitch[2], yaw[2];
                mat2RPY2(R_test[0], roll, pitch, yaw);
                double roll_, pitch_, yaw_;
                if (fabs(pitch[0] < fabs(pitch[1])))
                {
                    roll_ = roll[0];
                    pitch_ = pitch[0];
                    yaw_ = yaw[0];
                }
                else
                {
                    roll_ = roll[1];
                    pitch_ = pitch[1];
                    yaw_ = yaw[1];
                }

                f_gamma << yaw_<<" ";
                f_gamma << pitch_<<" ";
                f_gamma << roll_<<endl;

//                f_gamma << euler(0)*RAD2DEG<<" ";
//                f_gamma << euler(1)*RAD2DEG<<" ";
//                f_gamma << euler(2)*RAD2DEG<<endl;
                Eigen::Vector2d tmp_gc;
                tmp_gc(0) = x(0);
                tmp_gc(1) = x(1);
                gc_init(0) = x(0);
                gc_init(1) = x(1);
                x1.emplace_back(x(0));
                x2.emplace_back(x(1));
                x3.emplace_back(alpha*RAD2DEG);
//                x4.emplace_back(euler(0)*RAD2DEG);
//                x5.emplace_back(euler(1)*RAD2DEG);
//                x6.emplace_back(euler(2)*RAD2DEG);
                x4.emplace_back(yaw_);
                x5.emplace_back(pitch_);
                x6.emplace_back(roll_);
                x_time.emplace_back(time);
                //
            }
        }
    }
    f_gamma.close();
    f_cam.close();

    //判断收敛
    size_t res_num = x1.size();
    int n = 10;
    double condition1 = 0.005;
    double condition2 = 0.1;
    static ofstream f_dev;
    f_dev.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/Rcb_estimation_dev.txt");
    f_dev<<std::fixed<<std::setprecision(6);
    double x_min = std::numeric_limits<double>::max();
    double x1_min = std::numeric_limits<double>::max();
    double x2_min = std::numeric_limits<double>::max();
    double x3_min = std::numeric_limits<double>::max();
    double x1_max = std::numeric_limits<double>::min();
    double x2_max = std::numeric_limits<double>::min();
    double x3_max = std::numeric_limits<double>::min();
    vector<double> best_res;
    best_res.resize(3);
    for (int k = 0; k < res_num-n-5; ++k)
    {
        vector<double> x1_(x1.begin()+k, x1.begin()+k+n);
        vector<double> x2_(x2.begin()+k, x2.begin()+k+n);
        vector<double> x3_(x3.begin()+k, x3.begin()+k+n);
        vector<double> x4_(x4.begin()+k, x4.begin()+k+n);
        vector<double> x5_(x5.begin()+k, x5.begin()+k+n);
        vector<double> x6_(x6.begin()+k, x6.begin()+k+n);
        double dev1 = std::sqrt(GetVariance(x1_));
        double dev2 = std::sqrt(GetVariance(x2_));
        double dev3 = std::sqrt(GetVariance(x3_));
        double dev4 = std::sqrt(GetVariance(x4_));
        double dev5 = std::sqrt(GetVariance(x5_));
        double dev6 = std::sqrt(GetVariance(x6_));
        f_dev << x_time[k]<<" ";
        f_dev <<dev1<<" ";
        f_dev <<dev2<<" ";
        f_dev <<dev3<<" ";
        f_dev <<dev4<<" ";
        f_dev <<dev5<<" ";
        f_dev <<dev6<<endl;
//        if (dev1 < condition1 && dev2 < condition1 && dev3 < condition2)
//            f_dev<<"condition: "<<x1_[n-1]<<" "<<x2_[n-1]<<" "<<x3_[n-1]<<endl;
        if (dev1 < x1_min)
            x1_min = dev1;
        if (dev2 < x2_min)
            x2_min = dev2;
        if (dev3 < x3_min)
            x3_min = dev3;
        if (dev1 > x1_max)
            x1_max = dev1;
        if (dev2 > x2_max)
            x2_max = dev2;
        if (dev3 > x3_max)
            x3_max = dev3;
        double _x_min = x1_min + x2_min + x3_min;
        if (_x_min < x_min)
        {
            x_min = _x_min;
            best_res[0] = x4_[n-1];
            best_res[1] = x5_[n-1];
            best_res[2] = x6_[n-1];
        }

    }
    f_dev<<"min: "<<x1_min<<" "<<x2_min<<", "<<x3_min<<endl;
    f_dev<<"max: "<<x1_max<<" "<<x2_max<<" "<<x3_max<<endl;
    f_dev<<"Ric best result: "<<best_res[0]<<" "<<best_res[1]<<" "<<best_res[2]<<endl;
    Eigen::Matrix3d Ric_best;
    Ric_best = Eigen::AngleAxisd(best_res[0]*DEG2RAD,Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(best_res[1]*DEG2RAD,Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(best_res[2]*DEG2RAD,Eigen::Vector3d::UnitX());
    Rci = Ric_best.inverse();
    f_dev.close();
}


//测试，估计gc
void cSolver::solveOtherResult_gc(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                               Eigen::Matrix3d Ryx_cam,
                               Eigen::Matrix3d Ryx_imu)
{
    static ofstream f_gc;
    f_gc.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/gc.txt");
    f_gc<<std::fixed<<std::setprecision(6);
    f_gc <<"gc\n";
    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }
    Eigen::Matrix3d Ryx_imu_inv = Ryx_imu.transpose();
    Eigen::Matrix3d Ryx_cam_inv = Ryx_cam.transpose();

    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        cout<<"segmentCount: "<<segmentCount<<endl;
        int num = calib_data[segmentId].size()-1;
        cout<<"num: "<<num<<endl;
        Eigen::MatrixXd tmp_G(num*3,3);
        tmp_G.setZero();
        Eigen::MatrixXd tmp_b(num*3,1);
        Eigen::Matrix3d Rcb;
        Rcb << 0.003160590246273326, -0.9999791454805219, -0.005631986624785923,
                -0.002336363271321251, 0.0056246151765369234, -0.9999814523833838,
                0.9999922760081504, 0.0031736899915518757, -0.0023185374437218464;
        Eigen::Vector3d tcb;
//        tcb.setZero();
        tcb(0) = 0.02241856;
        tcb(1) = -0.01121906;
        tcb(2) = -0.01653902;
        for (int motionId = 0; motionId < calib_data[segmentId].size()-1; ++motionId)
        {
            Eigen::Vector3d pc1 = calib_data[segmentId][motionId].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[segmentId][motionId].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[segmentId][motionId+1].twc2_cam;
//            cout<<"pc1: "<<pc1.transpose()<<endl;

            Eigen::Vector3d t12_imu = calib_data[segmentId][motionId].P;
            Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+1].P;
            Eigen::Vector3d v12_imu = calib_data[segmentId][motionId].V;
            Eigen::Vector3d t12_cam = calib_data[segmentId][motionId].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+1].Rwc2_cam;
            double dT12 = calib_data[segmentId][motionId].T;
            double dT23 = calib_data[segmentId][motionId+1].T;
            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
            Eigen::Vector3d phi = (Rwc2_cam - Rwc3_cam)*tcb*dT12 +
                    (Rwc2_cam - Rwc1_cam)*tcb*dT23;//pcb
            Eigen::Vector3d gamma = -Rwc1_cam*Rcb*t12_imu*dT23 +
                    Rwc2_cam*Rcb*t23_imu*dT12 + Rwc1_cam*Rcb*v12_imu*dT12*dT23;//Rcb
            tmp_G.block<3,3>(3*motionId,0) = beta;
            tmp_b.block<3,1>(3*motionId,0) = phi + gamma - lambda;

        }
        //求解
        Eigen::VectorXd x;
        x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);
        f_gc << x(0)<<" ";
        f_gc << x(1)<<" ";
        f_gc << x(2)<<endl;
//        A = A * 1000;
//        b = b * 1000;
//        Eigen::VectorXd x(6);
//        x = A.ldlt().solve(b);
        std::cout<<"--------result--------\n";
        std::cout<<"x: "<<x.transpose()<<std::endl;
        f_gc.close();

    }
}

void cSolver::solveOtherResult_test(std::vector<data_selection::sync_data_test> &calib_data,
                                    Eigen::Matrix3d Ryx_cam,
                                    Eigen::Matrix3d Ryx_odo)
{
    Eigen::MatrixXd G(4,4);
    Eigen::VectorXd w(4);
    Eigen::Matrix3d Ryx_odo_inv = Ryx_odo.transpose();
    double a11, a21, a22;
    a11 = Ryx_odo_inv(0,0);
    a21 = Ryx_odo_inv(1,0);
    a22 = Ryx_odo_inv(1,1);
    for (int i = 0; i < calib_data.size(); ++i)
    {
        double t12_length = calib_data[i].t12_odo.norm();
        if(t12_length < 1e-4)//相机移动距离不小于１微米
            continue;
        if(calib_data[i].axis_odo(2) > -0.96)//最好只绕z轴旋转，即要接近于-1，axis(1)是相机y轴，且均为负数
            continue;

        Eigen::MatrixXd tmp_G(2,4);
        Eigen::Quaterniond qcl_odo = calib_data[i].q21_odo;
        Eigen::Matrix2d J;
        J = Eigen::Matrix2d::Identity() -
            qcl_odo.toRotationMatrix().block<2,2>(0,0);
//        J = qcl_odo.toRotationMatrix().block<2,2>(0,0) -
//            Eigen::Matrix2d::Identity();

        tmp_G.block<2,2>(0,0) = J;
        //
        Eigen::Vector3d tvec_cam = calib_data[i].t21_cam;
        Eigen::Vector3d n;
        n = Ryx_cam.row(2);
        Eigen::Vector3d pi = Ryx_cam * (tvec_cam - tvec_cam.dot(n)*n);

        Eigen::Matrix2d K;
        K << a11*pi(0), -a11*pi(1),
                a21*pi(0)+a22*pi(1), a22*pi(0)-a21*pi(1);
        tmp_G.block<2,2>(0, 2) = K;

        G += tmp_G.transpose()*tmp_G;

        Eigen::Vector3d t12_odo = calib_data[i].t12_odo;
        Eigen::Vector2d tmp_b(t12_odo(0), t12_odo(1));

        w += tmp_G.transpose() * tmp_b;
    }
    Eigen::VectorXd x(4);
    x = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(w);
    double alpha = atan2(x(3), x(2));
    std::cout<<"alpha: "<<alpha * RAD2DEG <<std::endl;
    std::cout<<"result: "<<x.transpose()<<std::endl;

    Eigen::Matrix3d R_z;
    R_z = Eigen::AngleAxisd(alpha,Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX());
    Eigen::Matrix3d Ric;
    Ric = Ryx_odo_inv * R_z * Ryx_cam;
    Eigen::Vector3d euler = Ric.eulerAngles(2,1,0);
    std::cout<<"roll, pitch, yaw: "<<euler.transpose() * RAD2DEG <<std::endl;
}

//Tool
double cSolver::GetVariance(std::vector<double> &vector_)
{
    double sum = std::accumulate(std::begin(vector_), std::end(vector_), 0.0);
    double mean = sum / vector_.size();

    double accum = 0.0;
    std::for_each(std::begin(vector_), std::end(vector_), [&](const double d){
        accum += (d-mean)*(d-mean);
    });

    return accum / vector_.size();
}