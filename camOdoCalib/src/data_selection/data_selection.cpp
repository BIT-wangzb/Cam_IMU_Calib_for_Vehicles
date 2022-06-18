#include "data_selection.h"
#include <algorithm>
#include <fstream>

#include <sophus/so3.hpp>

using namespace std;
using namespace Eigen;
using namespace Sophus;
#define PI 3.1415926
typedef std::vector<data_selection::imu_data> imuPtr;
typedef std::vector<data_selection::imu_preIntegration> imuPreInegrationPtr;
typedef std::vector<data_selection::cam_data> CamPtr;
typedef std::vector<data_selection::sync_data> SyncPtr;

data_selection::data_selection()
{
    g = Eigen::Vector3d(0,0,9.81);
//    sum_dt = 0.0;
//    delta_p.setZero();
//    delta_q.setIdentity();
//    delta_v.setZero();
//    linearized_ba = Eigen::Vector3d(0.157602,0.47403,0.0179254);//zed
//    linearized_ba = Eigen::Vector3d(0.03,0.03,0.001);//oxts
//    linearized_bg = Eigen::Vector3d(-0.000596544,0.000468732,-0.00114165);

    Init();
}

void data_selection::Init()
{
    jacobian.setIdentity();
    covariance.setZero();
    cv::FileStorage fs("../config.yaml",cv::FileStorage::READ);
    if (!fs.isOpened())
        std::cout<<"open config file failed!\n";
    double ACC_N, GYR_N, ACC_W, GYR_W;
    ACC_N = fs["acc_n"];
    ACC_W = fs["acc_w"];
    GYR_N = fs["gyr_n"];
    GYR_W = fs["gyr_w"];
    acc_bias(0) = fs["acc_bias_x"];
    acc_bias(1) = fs["acc_bias_y"];
    acc_bias(2) = fs["acc_bias_z"];
    gyr_bias(0) = fs["gyr_bias_x"];
    gyr_bias(1) = fs["gyr_bias_x"];
    gyr_bias(2) = fs["gyr_bias_x"];
    fs.release();

    noise.setZero();
    noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
}

//将筛选好对的数据存放在odoDatas\camDatas\sync_result，下标一一对应
void data_selection::selectData(imuPtr &imuDatas, CamPtr &camDatas,std::vector<data_selection::sync_data> &sync_result)
{
    //1.数据切齐
    startPosAlign(imuDatas,camDatas);//以时间最新的数据基准，得到数据头对齐的两组数据缓存,超出的部分删除
    //get rid of cam data ( deltaTheta = nan, tlc_length<1e-4,axis(1)<0.96), and align cam and odo data
    //2.数据配对
    imuPreInegrationPtr imuPreIntegrationDatas;
    camOdoAlign(imuDatas,camDatas,imuPreIntegrationDatas, sync_result);

    //get rid of the odo data whose distance is less than 1e-4 and whose theta sign isn't same
    std::vector<data_selection::sync_data> vec_sync_tmp;
    imuPreInegrationPtr imu_matches;///
    CamPtr cam_matches;
    int size = std::min(camDatas.size(), imuDatas.size());

    //3.筛选数据
    int nFlagDiff = 0;
    for (int i = 0; i < size; ++i)
    {
//        if(fabs(odoDatas[i].v_left) < 1e-3 || fabs(odoDatas[i].v_right) < 1e-3)//以里程计平均速度为衡量，将近似静止的帧去除
//            continue;
        //the sign is opposite
//        if((odoDatas[i].v_right - odoDatas[i].v_left) * camDatas[i].deltaTheta < 0.0)// rL == rR  去除相机和里程计旋转测量值方向相反的帧
//            {nFlagDiff++; continue; }
        imu_matches.push_back(imuPreIntegrationDatas[i]);
        cam_matches.push_back(camDatas[i]);
        vec_sync_tmp.push_back(sync_result[i]);
    }
    std::cout << "nFlagDiff = " << nFlagDiff <<std::endl;
    sync_result.swap(vec_sync_tmp);
    camDatas.swap(cam_matches);
    imuPreIntegrationDatas.swap(imu_matches);
}

void data_selection::ResetState(double time,
                                Eigen::Vector3d imu_acc,
                                Eigen::Vector3d imu_vel,
                                Eigen::Vector3d Ba, Eigen::Vector3d Bg)
{
    sum_dt = 0.0;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    prev_acc = imu_acc;
    prev_gyr = imu_vel;
    prev_time = time;
    linearized_ba = Ba;
    linearized_bg = Bg;
    jacobian.setIdentity();
    covariance.setZero();

}

void data_selection::midPointIntegration(double T,
                         const Eigen::Vector3d &curr_acc,
                         const Eigen::Vector3d &curr_gyr,
                         Eigen::Vector3d &result_delta_p,
                         Eigen::Quaterniond &result_delta_q,
                         Eigen::Vector3d &result_delta_v,
                         Eigen::Vector3d &result_linearized_ba,
                         Eigen::Vector3d &result_linearized_bg,
                         bool update_jacobian)
{
    //ROS_INFO("midpoint integration");
    //delta_q为相对预积分参考系的旋转四元数，线加速度的测量值减去偏差然后和旋转四元数相乘表示将线加速度从世界坐标系下转到了body(IMU)坐标系下
    //delta_q初始为单位四元数, delta_p初始为0
    //计算平均角速度
    Vector3d un_gyr = 0.5 * (prev_gyr + curr_gyr) - linearized_bg;
    //对平均角速度和时间的乘积构成的旋转值组成的四元数左乘旋转四元数，获得当前时刻body中的旋转向量（四元数表示）
//    Vector3d omega = un_gyr * T;
//    Eigen::Matrix3d dR = SO3d::exp(omega).matrix();
    result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * T / 2,
                                            un_gyr(1) * T / 2,
                                            un_gyr(2) * T / 2);
    result_delta_q.normalize();
    //用计算出来的旋转向量左乘当前的加速度，表示将线加速度从当前body系下转到预积分初始body坐标系下
    Vector3d un_acc_0 = delta_q * (prev_acc - linearized_ba);
    Vector3d un_acc_1 = result_delta_q * (curr_acc - linearized_ba);
    //计算两个时刻的平均加速度
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    //当前的位移：当前位移=前一次的位移+(速度×时间)+1/2×加速度的×时间的平方
    //匀加速度运动的位移公式：s_1 = s_0 + v_0 * t + 1/2 * a * t^2
    result_delta_p = delta_p + delta_v * T + 0.5 * un_acc * T * T;
    //速度计算公式：v_1 = v_0 + a*t
    result_delta_v = delta_v + un_acc * T;
    // 预积分的过程中Bias并未发生改变，所以还保存在result当中
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    //是否更新IMU预积分测量关于IMU Bias的雅克比矩阵
    if(update_jacobian)
    {
        //计算平均角速度
        Vector3d w_x = 0.5 * (prev_gyr + curr_gyr) - linearized_bg;
        //计算_acc_0这个观测线加速度对应的实际加速度
        Vector3d a_0_x = prev_acc - linearized_ba;
        //计算_acc_1这个观测线加速度对应的实际加速度
        Vector3d a_1_x = curr_acc - linearized_ba;
        Matrix3d R_w_x, R_a_0_x, R_a_1_x;
        /**
         *         | 0      -W_z     W_y |
         * [W]_x = | W_z     0      -W_x |
         *         | -W_y   W_x       0  |
        */

        //反对称矩阵
        R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
        R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
        R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

        //F是一个15行15列的数据类型为double，数据全部为0的矩阵，Matrix创建的矩阵默认按列存储
        MatrixXd F = MatrixXd::Zero(15, 15);
        /**
        * matrix.block(i,j, p, q) : 表示返回从矩阵(i, j)开始，每行取p个元素，每列取q个元素所组成的临时新矩阵对象，原矩阵的元素不变；
        * matrix.block<p,q>(i, j) :<p, q>可理解为一个p行q列的子矩阵，该定义表示从原矩阵中第(i, j)开始，获取一个p行q列的子矩阵，
        * 返回该子矩阵组成的临时矩阵对象，原矩阵的元素不变；
       */
        //从F矩阵的(0,0)位置的元素开始，将前3行3列的元素赋值为单位矩阵
        /**
         * 下面F和V矩阵为什么这样构造，是需要进行相关推导的。这里的F、V矩阵的构造理解可以参考论文附录中给出的公式
        */
        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * T * T +
                              -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * T) * T * T;
        F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * T;
        F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * T * T;
        F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * T * T * -T;
        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * T;
        F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * T;
        F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * T +
                              -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * T) * T;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * T;
        F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * T * -T;
        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();
        //cout<<"A"<<endl<<A<<endl;

        MatrixXd V = MatrixXd::Zero(15,18);
        V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * T * T;
        V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * T * T * 0.5 * T;
        V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * T * T;
        V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * T;
        V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * T;
        V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * T;
        V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * T * 0.5 * T;
        V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * T;
        V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * T;
        V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * T;

        //step_jacobian = F;
        //step_V = V;
        /**
         * 求矩阵的转置、共轭矩阵、伴随矩阵：可以通过成员函数transpose()、conjugate()、adjoint()来完成。注意：这些函数返回操作后的结果，
         * 而不会对原矩阵的元素进行直接操作，如果要让原矩阵进行转换，则需要使用响应的InPlace函数，如transpoceInPlace()等
        */
        //雅克比jacobian的迭代公式：J_(k+1)​=F*J_k​，J_0​=I
        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
    return;
}

void data_selection::propagate(double T, const Eigen::Vector3d &curr_acc,
                               const Eigen::Vector3d &curr_gyr)
{
    Eigen::Vector3d result_delta_p;
    Eigen::Quaterniond result_delta_q;
    Eigen::Vector3d result_delta_v;
    Eigen::Vector3d result_linearized_ba;
    Eigen::Vector3d result_linearized_bg;

    midPointIntegration(T, curr_acc, curr_gyr,
                        result_delta_p, result_delta_q,
                        result_delta_v, result_linearized_ba,
                        result_linearized_bg, true);

    //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
    //                    linearized_ba, linearized_bg);
    //更新PQV
    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;

    //更新偏置
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    //时间进行累加
    sum_dt += T;
    return;
}

void data_selection::processIMU(double curr_time, const Eigen::Vector3d &linear_acceleration,
                                const Eigen::Vector3d &angular_velocity)
{
    //1.判断是不是第一个imu消息，如果是第一个imu消息，则将当前传入的线加速度和角速度作为初始的加速度和角速度
   double T = curr_time - prev_time;
   prev_time = curr_time;
   propagate(T,linear_acceleration,angular_velocity);
   //预积分完后，更新当前的线加速度和角速度为上一时刻的线加速度和角速度
   prev_acc = linear_acceleration;
   prev_gyr = angular_velocity;
   return;
}

void data_selection::reprogate(SyncPtr &tmp_sync_data)
{
    for (auto &tmp_data : tmp_sync_data)
    {
        int id_imu = 0;
        double start_t = tmp_data.startTime;
        double end_t = tmp_data.endTime;
        while (_imuDatas[id_imu].time < start_t)
            id_imu++;
        int imu_start_id = id_imu;
        while (_imuDatas[id_imu].time < end_t)
            id_imu++;
        int imu_end_id = id_imu;

        //5.排除无用数据
        if(imu_end_id - imu_start_id <= 1)
            continue;

        //状态清除
        for (int j = imu_start_id; j <= imu_end_id; ++j)
        {
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            if (j == imu_start_id)
            {
                //imu插值
                double T = _imuDatas[j+1].time - _imuDatas[j].time;
                double dt1 = _imuDatas[j+1].time - start_t;
                double dt2 = start_t - _imuDatas[j].time;
                double front_scale = dt1 / T;
                double back_scale = dt2 / T;
                dx = _imuDatas[j].acc_x * front_scale + _imuDatas[j+1].acc_x * back_scale;
                dy = _imuDatas[j].acc_y * front_scale + _imuDatas[j+1].acc_y * back_scale;
                dz = _imuDatas[j].acc_z * front_scale + _imuDatas[j+1].acc_z * back_scale;
                rx = _imuDatas[j].angular_x * front_scale + _imuDatas[j+1].angular_x * back_scale;
                ry = _imuDatas[j].angular_y * front_scale + _imuDatas[j+1].angular_y * back_scale;
                rz = _imuDatas[j].angular_z * front_scale + _imuDatas[j+1].angular_z * back_scale;
                ResetState(start_t,Eigen::Vector3d(dx,dy,dz),
                           Eigen::Vector3d(rx,ry,rz),
                           acc_bias,
                           tmp_data.Bg);
            }
            else if (j == imu_end_id)
            {
                //imu插值
                double T = _imuDatas[j].time - _imuDatas[j-1].time;
                double dt1 = _imuDatas[j].time - end_t;
                double dt2 = end_t - _imuDatas[j-1].time;
                double front_scale = dt1 / T;
                double back_scale = dt2 / T;
                dx = _imuDatas[j].acc_x * front_scale + _imuDatas[j+1].acc_x * back_scale;
                dy = _imuDatas[j].acc_y * front_scale + _imuDatas[j+1].acc_y * back_scale;
                dz = _imuDatas[j].acc_z * front_scale + _imuDatas[j+1].acc_z * back_scale;
                rx = _imuDatas[j].angular_x * front_scale + _imuDatas[j+1].angular_x * back_scale;
                ry = _imuDatas[j].angular_y * front_scale + _imuDatas[j+1].angular_y * back_scale;
                rz = _imuDatas[j].angular_z * front_scale + _imuDatas[j+1].angular_z * back_scale;
                processIMU(end_t,
                           Eigen::Vector3d(dx,dy,dz),
                           Eigen::Vector3d(rx,ry,rz));
            }
            else
            {
                dx = _imuDatas[j].acc_x;
                dy = _imuDatas[j].acc_y;
                dz = _imuDatas[j].acc_z;
                rx = _imuDatas[j].angular_x;
                ry = _imuDatas[j].angular_y;
                rz = _imuDatas[j].angular_z;
                //imu预积分
                processIMU(_imuDatas[j].time,
                           Eigen::Vector3d(dx,dy,dz),
                           Eigen::Vector3d(rx,ry,rz));
            }
        }

        Eigen::Quaterniond q_21 = delta_q.conjugate();

        Eigen::Matrix3d R21 = q_21.toRotationMatrix();
        Eigen::AngleAxisd rotation_vector(q_21);//3:旋转向量表示
        axis = rotation_vector.axis();//旋转轴
        deltaTheta_21 = rotation_vector.angle(); //c2->c1，旋转角度（弧度表示）
        if(axis(2)>0)//使轴和角有统一方向的计量(y轴)？
        {
            deltaTheta_21 *= -1;
            axis *= -1;
        }
        tmp_data.P = delta_p;
        tmp_data.R = delta_q;
        tmp_data.V = delta_v;
        tmp_data.jacobian_ = jacobian;
        tmp_data.axis_imu = axis;
        tmp_data.angle_imu = deltaTheta_21;
        tmp_data.t21_imu = -R21 * delta_p;
        tmp_data.q21_imu = q_21;
    }
}

//数据配对-----数据同步算法中最有价值的部分
//配好对的数据存放在odoDatas\camDatas\sync_result，下标一一对应(相机的前三帧不使用)
void data_selection::camOdoAlign(imuPtr &imuDatas,
                                 CamPtr &camDatas,
                                 imuPreInegrationPtr &preIntegrationDatas,
                                 std::vector<data_selection::sync_data> &sync_result)
{
    int id_imu = 0;
    imuPtr imuDatas_tmp;
    CamPtr camDatas_tmp;
    imuPreInegrationPtr preIntegrationDatas_tmp;

    static ofstream f_imu_pre;
    f_imu_pre.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/imu_pre.txt");
    f_imu_pre<<std::fixed<<std::setprecision(6);
    f_imu_pre <<"start_t end_t px py pz qx qy qz qw vx vy vz\n";
    //1.以相机帧为计量单位，配对数据为从第四帧到倒数第２帧段循环
    for (int i = 0; unsigned(i) < camDatas.size(); ++i)
    {
        //2.数据粗筛
        if( std::isnan(camDatas[i].deltaTheta_21) )
            continue;

        //3. 找到距离相机i帧start_time最近的imu时间戳imu_start
        while (imuDatas[id_imu].time < camDatas[i].start_t)
            id_imu++;
        int imu_start_id = id_imu-1;
        while (imuDatas[id_imu].time < camDatas[i].end_t)
            id_imu++;
        int imu_end_id = id_imu;
        id_imu--;

        //5.排除无用数据
        if(imu_end_id - imu_start_id <= 1)
            continue;
//        std::cout<<"start_id: "<<imu_start_id<<std::endl;
//        std::cout<<"end_id: "<<imu_end_id<<std::endl;
        f_imu_pre << imuDatas[imu_start_id].time <<" ";
        f_imu_pre << imuDatas[imu_end_id].time <<" ";
        //6,计算预积分
        for (int j = imu_start_id; j <= imu_end_id; ++j)
        {
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            if (j == imu_start_id)
            {
                //imu插值
                double T = imuDatas[j+1].time - imuDatas[j].time;
                double dt1 = imuDatas[j+1].time - camDatas[i].start_t;
                double dt2 = camDatas[i].start_t - imuDatas[j].time;
                double front_scale = dt1 / T;
                double back_scale = dt2 / T;
                dx = imuDatas[j].acc_x * front_scale + imuDatas[j+1].acc_x * back_scale;
                dy = imuDatas[j].acc_y * front_scale + imuDatas[j+1].acc_y * back_scale;
                dz = imuDatas[j].acc_z * front_scale + imuDatas[j+1].acc_z * back_scale;
                rx = imuDatas[j].angular_x * front_scale + imuDatas[j+1].angular_x * back_scale;
                ry = imuDatas[j].angular_y * front_scale + imuDatas[j+1].angular_y * back_scale;
                rz = imuDatas[j].angular_z * front_scale + imuDatas[j+1].angular_z * back_scale;
                ResetState(camDatas[i].start_t,Eigen::Vector3d(dx,dy,dz),
                           Eigen::Vector3d(rx,ry,rz),
                           acc_bias,
                           gyr_bias);
            }
            else if (j == imu_end_id)
            {
                //imu插值
                double T = imuDatas[j].time - imuDatas[j-1].time;
                double dt1 = imuDatas[j].time - camDatas[i].end_t;
                double dt2 = camDatas[i].end_t - imuDatas[j-1].time;
                double front_scale = dt1 / T;
                double back_scale = dt2 / T;
                dx = imuDatas[j].acc_x * front_scale + imuDatas[j+1].acc_x * back_scale;
                dy = imuDatas[j].acc_y * front_scale + imuDatas[j+1].acc_y * back_scale;
                dz = imuDatas[j].acc_z * front_scale + imuDatas[j+1].acc_z * back_scale;
                rx = imuDatas[j].angular_x * front_scale + imuDatas[j+1].angular_x * back_scale;
                ry = imuDatas[j].angular_y * front_scale + imuDatas[j+1].angular_y * back_scale;
                rz = imuDatas[j].angular_z * front_scale + imuDatas[j+1].angular_z * back_scale;
                processIMU(camDatas[i].end_t,
                           Eigen::Vector3d(dx,dy,dz),
                           Eigen::Vector3d(rx,ry,rz));
            }
            else
            {
                dx = imuDatas[j].acc_x;
                dy = imuDatas[j].acc_y;
                dz = imuDatas[j].acc_z;
                rx = imuDatas[j].angular_x;
                ry = imuDatas[j].angular_y;
                rz = imuDatas[j].angular_z;
                //imu预积分
                processIMU(imuDatas[j].time,
                           Eigen::Vector3d(dx,dy,dz),
                           Eigen::Vector3d(rx,ry,rz));
            }

        }

        f_imu_pre << sum_dt<<" ";
        f_imu_pre << delta_p(0)<<" ";
        f_imu_pre << delta_p(1)<<" ";
        f_imu_pre << delta_p(2)<<" ";
        f_imu_pre << delta_q.x()<<" ";
        f_imu_pre << delta_q.y()<<" ";
        f_imu_pre << delta_q.z()<<" ";
        f_imu_pre << delta_q.w()<<" ";
        f_imu_pre << delta_v(0)<<" ";
        f_imu_pre << delta_v(1)<<" ";
        f_imu_pre << delta_v(2)<<std::endl;
        Eigen::Quaterniond q12_cam(camDatas[i].R12);
        //test
        Eigen::AngleAxisd rotation_vector1(delta_q);//3:旋转向量表示
        Eigen::Vector3d axis_test = rotation_vector1.axis();//旋转轴
        double deltaTheta_12 = rotation_vector1.angle(); //c2->c1，旋转角度（弧度表示）

        Eigen::Quaterniond q_21(delta_q.toRotationMatrix().inverse());
        Eigen::Matrix3d R21 = q_21.toRotationMatrix();
        Eigen::AngleAxisd rotation_vector(q_21);//3:旋转向量表示
        axis = rotation_vector.axis();//旋转轴
        deltaTheta_21 = rotation_vector.angle(); //c2->c1，旋转角度（弧度表示）
        if(axis(2)>0)//使轴和角有统一方向的计量(z轴)？
        {
            deltaTheta_21 *= -1;
            axis *= -1;
        }

        data_selection::imu_preIntegration imuPreIntegration_tmp;
        imuPreIntegration_tmp.P = delta_p;
        imuPreIntegration_tmp.R = delta_q;
        imuPreIntegration_tmp.V = delta_v;
        imuPreIntegration_tmp.Ba = linearized_ba;
        imuPreIntegration_tmp.Bg = linearized_bg;
        imuPreIntegration_tmp.axis = axis;
        imuPreIntegration_tmp.delta_angle = deltaTheta_21;
        imuPreIntegration_tmp.tlc = -R21 * delta_p;

        //7,将对齐后的数据传递给配对数据
        //IMU
        data_selection::sync_data sync_tmp;
        double T = camDatas[i].end_t - camDatas[i].start_t;
        sync_tmp.T = T;
        sync_tmp.P = delta_p;
        sync_tmp.R = delta_q;
        sync_tmp.V = delta_v;
        sync_tmp.Ba = linearized_ba;
        sync_tmp.Bg = linearized_bg;
        sync_tmp.jacobian_ = jacobian;
        sync_tmp.axis_imu = axis;
        sync_tmp.angle_imu = deltaTheta_21;
        sync_tmp.t21_imu = -R21 * delta_p;
        sync_tmp.q21_imu = q_21;
        sync_tmp.sum_dt = sum_dt;
        //Camera
        sync_tmp.cam_t12 = camDatas[i].t12;
        sync_tmp.scan_match_results[0] = camDatas[i].t12[0];//t(c1, c1c2)
        sync_tmp.scan_match_results[1] = camDatas[i].t12[2];
        sync_tmp.scan_match_results[2] = camDatas[i].deltaTheta_21;//c2->c1
        sync_tmp.t21_cam = -camDatas[i].R21 * camDatas[i].t12;//t(c2, c2c1)=-Rc2c1*t(c1, c1c2)
        sync_tmp.q21_cam = Eigen::Quaterniond(camDatas[i].R21);//Rc2c1的四元数表达
        sync_tmp.q12_cam= Eigen::Quaterniond(camDatas[i].R12);//Rc1c2的四元数表达
        sync_tmp.angle = camDatas[i].deltaTheta_21; // should be cl  theta(c2->c1)
        sync_tmp.axis = camDatas[i].axis_21;
        sync_tmp.startTime = camDatas[i].start_t;
        sync_tmp.endTime = camDatas[i].end_t;
        sync_tmp.Rwc1_cam = camDatas[i].Rwc1;
        sync_tmp.twc1_cam = camDatas[i].twc1;
        sync_tmp.Rwc2_cam = camDatas[i].Rwc2;
        sync_tmp.twc2_cam = camDatas[i].twc2;
        camDatas_tmp.push_back(camDatas[i]);
        preIntegrationDatas.push_back(imuPreIntegration_tmp);
        preIntegrationDatas_tmp.push_back(imuPreIntegration_tmp);
        sync_result.push_back(sync_tmp);
        id_imu--;
    }//for image loop
    camDatas.swap(camDatas_tmp);///交换两个容器内容(camDatas内容有删减)
    return;
}

//以时间最新的数据基准，得到数据头对齐的两组数据缓存,超出的部分删除
void data_selection::startPosAlign(imuPtr& imuDatas, CamPtr& camDatas)
{
    _imuDatas.resize(imuDatas.size());
    std::copy(imuDatas.begin(),imuDatas.end(),_imuDatas.begin());
    double t0_cam = camDatas[0].start_t;
    double t0_imu = imuDatas[0].time;
    if(t0_cam > t0_imu)//如果相机时间大，以相机数据为基准，距离相机数据头最近的imu数据段之前都被删除
    {
        int imu_start = 0;
        while (imuDatas[imu_start].time < t0_cam)
            imu_start++;
        int imu_aligned = imu_start-1;
        for (int i = 0; i < imu_aligned; ++i)
            imuDatas.erase(imuDatas.begin());

        std::cout << std::fixed << "aligned start position1: "
                    << camDatas[0].start_t << "  "
                    << imuDatas[0].time <<std::endl;
        return;
    }
    else if(t0_imu > t0_cam)//如果imu时间大，以imu数据为基准，距离imu数据头最近的相机数据段之前都被删除
    {

        int cam_start = 0;
        while (camDatas[cam_start].start_t < t0_imu)
            cam_start++;
        int cam_aligned = cam_start + 1;
        for (int i = 0; i < cam_aligned; ++i)
            camDatas.erase(camDatas.begin());

        int imu_start = 0;
        while (imuDatas[imu_start].time < t0_cam)
            imu_start++;
        int imu_aligned = imu_start + 1;
        for (int i = 0; i < imu_aligned; ++i)
            imuDatas.erase(imuDatas.begin());

        std::cout << std::fixed << "aligned start position2 : "
                << camDatas[0].start_t << "  "
                << imuDatas[0].time <<std::endl;
        return;
    }
    else
    {
        return;
    }

}

//for test
void data_selection::startPosAlign_test(std::vector<odo_data_test>& imuDatas,
                                        CamPtr& camDatas)
{
//    _imuDatas.resize(imuDatas.size());
//    std::copy(imuDatas.begin(),imuDatas.end(),_imuDatas.begin());
    double t0_cam = camDatas[0].start_t;
    double t0_imu = imuDatas[0].time;
    if(t0_cam > t0_imu)//如果相机时间大，以相机数据为基准，距离相机数据头最近的imu数据段之前都被删除
    {
        int imu_start = 0;
        while (imuDatas[imu_start].time < t0_cam)
            imu_start++;
        int imu_aligned = imu_start - 1;
        for (int i = 0; i < imu_aligned; ++i)
            imuDatas.erase(imuDatas.begin());

        std::cout << std::fixed << "aligned start position1: "
                  << camDatas[0].start_t << "  "
                  << imuDatas[0].time <<std::endl;
        return;
    }
    else if(t0_imu > t0_cam)//如果imu时间大，以imu数据为基准，距离imu数据头最近的相机数据段之前都被删除
    {

        int cam_start = 0;
        while (camDatas[cam_start].start_t < t0_imu)
            cam_start++;
        int cam_aligned = cam_start + 1;
        for (int i = 0; i < cam_aligned; ++i)
            camDatas.erase(camDatas.begin());

        int imu_start = 0;
        while (imuDatas[imu_start].time < t0_cam)
            imu_start++;
        int imu_aligned = imu_start + 1;
        for (int i = 0; i < imu_aligned; ++i)
            imuDatas.erase(imuDatas.begin());

        std::cout << std::fixed << "aligned start position2 : "
                  << camDatas[0].start_t << "  "
                  << imuDatas[0].time <<std::endl;
        return;
    }
    else
    {
        return;
    }

}

void data_selection::camOdoAlign_test(std::vector<odo_data_test> &odoDatas,
                                      std::vector<cam_data> &camDatas,
                                      std::vector<data_selection::sync_data_test> &sync_result)
{
    int id_imu = 0;
    imuPtr imuDatas_tmp;
    CamPtr camDatas_tmp;

    //1.以相机帧为计量单位，配对数据为从第四帧到倒数第２帧段循环
    int last_end_id = 1;
    for (int i = 0; unsigned(i) < camDatas.size() - 3; ++i)
    {
        //2.数据粗筛
        if( std::isnan(camDatas[i].deltaTheta_21) )
            continue;

//        id_imu = last_end_id - 1;
        //3. 找到距离相机i帧start_time最近的imu时间戳imu_start
        while (odoDatas[id_imu].time < camDatas[i].start_t)
            id_imu++;
        int imu_start_id = id_imu;
        while (odoDatas[id_imu].time < camDatas[i].end_t)
            id_imu++;
        int imu_end_id = id_imu;
//        std::cout<<"start_id: "<<imu_start_id<<std::endl;
//        std::cout<<"end_id: "<<imu_end_id<<std::endl;
//        std::cout<<"-------------\n";

//        last_end_id = imu_end_id;

        //5.排除无用数据
        if(imu_end_id - imu_start_id <= 1)
            continue;
        //6,
        //start
        double dt1 = odoDatas[imu_start_id+1].time - camDatas[i].start_t;
        double T = odoDatas[imu_start_id+1].time - odoDatas[imu_start_id].time;
        double omega = dt1 / T;
        Eigen::Quaterniond tmp_qi = odoDatas[imu_start_id].q;
        Eigen::Vector3d tmp_ti = odoDatas[imu_start_id].tvec;

        Eigen::Quaterniond tmp_qj = odoDatas[imu_start_id+1].q;
        Eigen::Vector3d tmp_tj = odoDatas[imu_start_id+1].tvec;

        Eigen::Quaterniond slerp_quat1 = tmp_qi.slerp(omega, tmp_qj);
        Eigen::Vector3d interpTrans1;
        interpTrans1(0) = tmp_tj(0) * (1-omega) + tmp_ti(0) * omega;
        interpTrans1(1) = tmp_tj(1) * (1-omega) + tmp_ti(1) * omega;
        interpTrans1(2) = tmp_tj(2) * (1-omega) + tmp_ti(2) * omega;


        //end
        double dt2 = camDatas[i].end_t - odoDatas[imu_end_id-1].time;
        T = odoDatas[imu_end_id].time - odoDatas[imu_end_id-1].time;
        omega = dt2 / T;
//        std::cout<<"omega: "<<omega<<std::endl;
        tmp_qi = odoDatas[imu_end_id-1].q;
        tmp_ti = odoDatas[imu_end_id-1].tvec;
        tmp_qj = odoDatas[imu_end_id].q;
        tmp_tj = odoDatas[imu_start_id+1].tvec;
//        std::cout<<"ti: "<<tmp_ti.transpose()<<", qi: "
//            <<tmp_qi.coeffs().transpose()<<std::endl;
//        std::cout<<"tj: "<<tmp_tj.transpose()<<", qj: "
//                 <<tmp_qj.coeffs().transpose()<<std::endl;

        Eigen::Quaterniond slerp_quat2 = tmp_qi.slerp(omega, tmp_qj);
        Eigen::Vector3d interpTrans2;
        interpTrans2(0) = tmp_tj(0) * omega + tmp_ti(0) * (1-omega);
        interpTrans2(1) = tmp_tj(1) * omega + tmp_ti(1) * (1-omega);
        interpTrans2(2) = tmp_tj(2) * omega + tmp_ti(2) * (1-omega);
//        std::cout<<"t1: "<<interpTrans2.transpose()<<", q1: "
//                 <<slerp_quat2.coeffs().transpose()<<std::endl;


        Eigen::Quaterniond qwc1 = slerp_quat1;
        Eigen::Quaterniond qwc2 = slerp_quat2;
        Eigen::Vector3d t1 = interpTrans1;
        Eigen::Vector3d t2 = interpTrans2;
//        Eigen::Quaterniond qwc1 = odoDatas[imu_start_id].q;
//        Eigen::Quaterniond qwc2 = odoDatas[imu_end_id].q;
//        Eigen::Vector3d t1 = odoDatas[imu_start_id].tvec;
//        Eigen::Vector3d t2 = odoDatas[imu_end_id].tvec;
        Eigen::Matrix3d R12 = qwc1.toRotationMatrix().inverse() * qwc2.toRotationMatrix();
        Eigen::Matrix3d R21 = qwc2.toRotationMatrix().inverse() * qwc1.toRotationMatrix();

        Eigen::Vector3d t12 = qwc1.toRotationMatrix().inverse() * (t2 - t1);
        Eigen::Vector3d t21 = qwc2.toRotationMatrix().inverse() * (t1 - t2);

        Eigen::Quaterniond q12(R12);
        Eigen::Quaterniond q21(R21);

        Eigen::AngleAxisd rotation_vector(q21);//3:旋转向量表示

        Eigen::Vector3d axis_odo = rotation_vector.axis();//旋转轴
        double deltaTheta_odo = rotation_vector.angle(); //c2->c1，旋转角度（弧度表示）
        if(axis_odo(2)>0)//使轴和角有统一方向的计量(y轴)？
        {
            deltaTheta_odo *= -1;
            axis_odo *= -1;
        }
        std::cout<<"odo_axis: "<<axis_odo.transpose()<<std::endl;
        //7,将对齐后的数据传递给配对数据
        //odo
        data_selection::sync_data_test sync_tmp;
        sync_tmp.R12_odo = R12;
        sync_tmp.R21_odo = R21;
        sync_tmp.t12_odo = t12;
        sync_tmp.t21_odo = t21;
        sync_tmp.q12_odo = q12;
        sync_tmp.q21_odo = q21;
        sync_tmp.angle_odo = deltaTheta_odo;
        sync_tmp.axis_odo= axis_odo;

        //Camera
        sync_tmp.cam_t12 = camDatas[i].t12;
        sync_tmp.scan_match_results[0] = camDatas[i].t12[0];//t(c1, c1c2)
        sync_tmp.scan_match_results[1] = camDatas[i].t12[2];
        sync_tmp.scan_match_results[2] = camDatas[i].deltaTheta_21;//c2->c1
        //all base c2
        sync_tmp.t21_cam = -camDatas[i].R21 * camDatas[i].t12;//t(c2, c2c1)=-Rc2c1*t(c1, c1c2)
        sync_tmp.q21_cam = Eigen::Quaterniond(camDatas[i].R21);//Rc2c1的四元数表达
        sync_tmp.q12_cam= Eigen::Quaterniond(camDatas[i].R12);//Rc1c2的四元数表达
        sync_tmp.angle = camDatas[i].deltaTheta_21; // should be cl  theta(c2->c1)
        sync_tmp.axis = camDatas[i].axis_21;
        sync_tmp.startTime = camDatas[i].start_t;
        sync_tmp.endTime = camDatas[i].end_t;
        camDatas_tmp.push_back(camDatas[i]);
        sync_result.push_back(sync_tmp);
        id_imu--;
    }//for image loop
    camDatas.swap(camDatas_tmp);///交换两个容器内容(camDatas内容有删减)
    return;
}

void data_selection::selectData_test(std::vector<odo_data_test> &odoDatas,
                                     std::vector<cam_data> &camDatas,
                                     std::vector<data_selection::sync_data_test> &sync_result) {
    //1.数据切齐
    startPosAlign_test(odoDatas,camDatas);//以时间最新的数据基准，得到数据头对齐的两组数据缓存,超出的部分删除
    //get rid of cam data ( deltaTheta = nan, tlc_length<1e-4,axis(1)<0.96), and align cam and odo data
    //2.数据配对
    camOdoAlign_test(odoDatas,camDatas,sync_result);

    //get rid of the odo data whose distance is less than 1e-4 and whose theta sign isn't same
    std::vector<data_selection::sync_data_test> vec_sync_tmp;
    imuPreInegrationPtr imu_matches;///
    CamPtr cam_matches;
    int size = std::min(camDatas.size(), odoDatas.size());

    //3.筛选数据
    int nFlagDiff = 0;
    for (int i = 0; i < size; ++i)
    {
//        if(fabs(odoDatas[i].v_left) < 1e-3 || fabs(odoDatas[i].v_right) < 1e-3)//以里程计平均速度为衡量，将近似静止的帧去除
//            continue;
        //the sign is opposite
//        if((odoDatas[i].v_right - odoDatas[i].v_left) * camDatas[i].deltaTheta < 0.0)// rL == rR  去除相机和里程计旋转测量值方向相反的帧
//            {nFlagDiff++; continue; }
        cam_matches.push_back(camDatas[i]);
        vec_sync_tmp.push_back(sync_result[i]);
    }
    std::cout << "nFlagDiff = " << nFlagDiff <<std::endl;
    sync_result.swap(vec_sync_tmp);
    camDatas.swap(cam_matches);
}
