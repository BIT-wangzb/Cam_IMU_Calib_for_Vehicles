/*******************************************************
 * Copyright (C) 2019, SLAM Group, Megvii-R
 *******************************************************/

#include <stdio.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <fstream>//zdf 
#include <math.h>
#include <chrono>

//#include "camera_models/include/Camera.h"
//#include "camera_models/include/CameraFactory.h"
//#include "calc_cam_pose/calcCamPose.h"

#include "solveQyx.h"

#define DEG2RAD 0.01745329
#define RAD2DEG 57.29578

using namespace std;

std::mutex m_buf;
bool hasImg = false ;//zdf
std::vector<data_selection::imu_data> imuDatas;
std::vector<data_selection::cam_data> camDatas;
std::vector<data_selection::imu_data> imuDatas2;
std::vector<data_selection::cam_data> camDatas2;
std::vector<data_selection::odo_data_test> groundDatas;

//record the first frame calculated successfully
bool fisrt_frame = true;
Eigen::Matrix3d Rwc0;
Eigen::Vector3d twc0;
//decide if the frequent is decreased
bool halfFreq = false;
int frame_index = 0;

std::vector< std::vector<cv::Point3d> > vec_cam_p3ds;

void calc_process2(Eigen::Matrix3d &Rci)
{
    std::cout << "============ calibrating pcb... ===============" << std::endl;
    data_selection ds2;
    std::vector<data_selection::sync_data> sync_result2;
    //数据同步处理
    ds2.selectData(imuDatas2,camDatas2,sync_result2);//将筛选好对的数据存放在odoDatas\camDatas\sync_result，下标一一对应

    cSolver cSolver;
    cSolver.solveGyroscopeBias(sync_result2,Rci);
    ds2.reprogate(sync_result2);
    std::vector<std::vector<data_selection::sync_data> > tmp_calib_data;
    tmp_calib_data.push_back(sync_result2);
    Eigen::Vector3d gc;
    //
    cSolver.solvePcb(tmp_calib_data,Rci,gc);
    cout<<"gc: "<<gc.transpose()<<endl;
    cout<<"gc norm: "<<gc.norm()<<endl;
    cSolver.RefinePcb(tmp_calib_data,Rci,gc);
    return;
}

void calc_process(Eigen::Matrix3d &Rci)
{
    //1.构建外参解算算法对象
    SolveQyx cSolveQyx;
    std::cout << "============ calibrating... ===============" << std::endl;
    //2.构建标定数据准备算法对象
    data_selection ds;
    std::vector<data_selection::sync_data> sync_result;
    //3.数据同步处理
    ds.selectData(imuDatas,camDatas,sync_result);//将筛选好对的数据存放在odoDatas\camDatas\sync_result，下标一一对应

    Eigen::Matrix3d Ryx,Ryx_imu;
    cSolveQyx.estimateRyx_cam(sync_result,Ryx);
    cout<<"cam Ryx:\n"<<Ryx<<endl;
    cSolveQyx.estimateRyx_imu(sync_result,Ryx_imu);
    cout<<"imu Ryx:\n"<<Ryx_imu<<endl;

    cSolver cSolver;
    std::vector<std::vector<data_selection::sync_data> > tmp_calib_data;
    tmp_calib_data.push_back(sync_result);
    Rci.setIdentity();
    //gc,yaw
    cSolver.solveRcb(tmp_calib_data,Ryx,Ryx_imu,Rci);
    return;
}

//斜坡运动
bool readCamPose2(string path, double timeShift)
{
    cv::FileStorage fs("../config.yaml",cv::FileStorage::READ);
    int n = fs["interval2"];
    fs.release();

    ifstream pose_txt;
    pose_txt.open(path,ios::in);
    if (!pose_txt.is_open())
    {
        cout << "读取cam文件失败" << endl;
        return false;
    }
    string str;
    bool first_line = true;
    bool first_frame = true;
    Eigen::Matrix3d Rwl;
    Eigen::Vector3d twl;
    double last_time;
    while (getline(pose_txt,str))
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }
        for (int i = 0; i < n; ++i) {
        getline(pose_txt,str);//间隔读取

        }
        if (str.size() == 0)
            break;
        stringstream line_info(str);
        string temp_str;
        vector<string> temp_vec;
        while (line_info >> temp_str)
            temp_vec.push_back(temp_str);

        double curr_time = stod(temp_vec[0]) + timeShift;
        Eigen::Vector3d twc;
        twc(0) = stod(temp_vec[1]);
        twc(1) = stod(temp_vec[2]);
        twc(2) = stod(temp_vec[3]);
        Eigen::Quaterniond q_wc;
        q_wc.x() = stod(temp_vec[4]);
        q_wc.y() = stod(temp_vec[5]);
        q_wc.z() = stod(temp_vec[6]);
        q_wc.w() = stod(temp_vec[7]);

        Eigen::Matrix3d Rwc;
        Rwc = q_wc.toRotationMatrix();
        if (first_frame)
        {
            Rwc0 = Rwc;
            twc0 = twc;
            Rwl = Rwc;
            twl = twc;
            last_time = curr_time;
            first_frame = false;
            continue;
        }

        Eigen::Matrix3d R12 = Rwl.inverse() * Rwc;
        Eigen::Quaterniond q_12(R12);//2:四元数表示
        Eigen::Matrix3d R21 = R12.inverse();//1:欧拉旋转矩阵表示
        Eigen::Quaterniond q_21(R21);//2:四元数表示

        Eigen::AngleAxisd rotation_vector(q_21);//3:旋转向量表示
        Eigen::Vector3d axis_21 = rotation_vector.axis();//旋转轴
        double deltaTheta_21 = rotation_vector.angle(); //c2->c1，旋转角度（弧度表示）
        if(axis_21(1)>0)//使轴和角有统一方向的计量(y轴)？
        {
            deltaTheta_21 *= -1;
            axis_21 *= -1;
        }
        //获得两个相机状态间的相对位置
        Eigen::Vector3d t12 = -Rwl.inverse() * (twl - twc);//?

        //保存为相机数据
        data_selection::cam_data cam_tmp;
        cam_tmp.start_t = last_time; //前一时刻相机时间
        cam_tmp.end_t = curr_time; //当前采集时间
        //cam_tmp.theta_y = theta_y;
        cam_tmp.deltaTheta_21 = deltaTheta_21; //
        cam_tmp.axis_21 = axis_21; //相机相对旋转（轴 Rc2c1）
//        cout<<"axis(1): "<<axis(1)<<endl;
        cam_tmp.R21 =  R21;//相机相对旋转(前帧为参考 Rc2c1)
        cam_tmp.R12 = R12;
        cam_tmp.t12 =  t12;//相机相对位移(前帧为参考 t(c1, c1c2))
        cam_tmp.Rwc1 = Rwl;
        cam_tmp.twc1 = twl;
        cam_tmp.Rwc2 = Rwc;
        cam_tmp.twc2 = twc;
        camDatas2.push_back(cam_tmp);

        Rwl = Rwc;
        twl = twc;
        last_time = curr_time;
    }
    pose_txt.close();

    return true;
}
bool readIMUData2(string path)
{
    ifstream imu_txt;
    imu_txt.open(path,ios::in);
    if (!imu_txt.is_open())
    {
        cout << "读取imu文件失败" << endl;
        return false;
    }
    string str;
    bool first_line = true;
    Eigen::Matrix3d Rwl;
    Eigen::Vector3d twl;
    while (getline(imu_txt,str)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        stringstream line_info(str);
        string temp_str;
        vector<string> temp_vec;
        while (line_info >> temp_str)
            temp_vec.push_back(temp_str);

        data_selection::imu_data imuData;
        imuData.time = stod(temp_vec[0]);
        imuData.acc_x = stod(temp_vec[1]);
        imuData.acc_y = stod(temp_vec[2]);
        imuData.acc_z = stod(temp_vec[3]);
        imuData.angular_x = stod(temp_vec[4]);
        imuData.angular_y = stod(temp_vec[5]);
        imuData.angular_z = stod(temp_vec[6]);
        imuDatas2.push_back(imuData);
    }
    imu_txt.close();
    return true;
}

//平面运动
bool readCamPose(string path, double timeShift)
{
    cv::FileStorage fs("../config.yaml",cv::FileStorage::READ);
    int n = fs["interval1"];
    ifstream pose_txt;
    pose_txt.open(path,ios::in);
    if (!pose_txt.is_open())
    {
        cout << "读取cam文件失败" << endl;
        return false;
    }
    string str;
    bool first_line = true;
    bool first_frame = true;
    Eigen::Matrix3d Rwl;
    Eigen::Vector3d twl;
    double last_time;
    
    while (getline(pose_txt,str))
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }
        for (int i = 0; i < n; ++i) {
            getline(pose_txt,str);//间隔读取
        }
        if (str.size() == 0)
            break;
        stringstream line_info(str);
        string temp_str;
        vector<string> temp_vec;
        while (line_info >> temp_str)
            temp_vec.push_back(temp_str);

        double curr_time = stod(temp_vec[0]) + timeShift;
        Eigen::Vector3d twc;
        twc(0) = stod(temp_vec[1]);
        twc(1) = stod(temp_vec[2]);
        twc(2) = stod(temp_vec[3]);
        Eigen::Quaterniond q_wc;
        q_wc.x() = stod(temp_vec[4]);
        q_wc.y() = stod(temp_vec[5]);
        q_wc.z() = stod(temp_vec[6]);
        q_wc.w() = stod(temp_vec[7]);

        Eigen::Matrix3d Rwc;
        Rwc = q_wc.toRotationMatrix();
        if (first_frame)
        {
            Rwc0 = Rwc;
            twc0 = twc;
            Rwl = Rwc;
            twl = twc;
            last_time = curr_time;
            first_frame = false;
            continue;
        }

        Eigen::Matrix3d R12 = Rwl.inverse() * Rwc;
        Eigen::Quaterniond q_12(R12);//2:四元数表示
        Eigen::Matrix3d R21 = R12.inverse();//1:欧拉旋转矩阵表示
        Eigen::Quaterniond q_21(R21);//2:四元数表示

        Eigen::AngleAxisd rotation_vector(q_21);//3:旋转向量表示
        Eigen::Vector3d axis_21 = rotation_vector.axis();//旋转轴
        double deltaTheta_21 = rotation_vector.angle(); //c2->c1，旋转角度（弧度表示）
        if(axis_21(1)>0)//使轴和角有统一方向的计量(y轴)？
        {
            deltaTheta_21 *= -1;
            axis_21 *= -1;
        }
        //获得两个相机状态间的相对位置
        Eigen::Vector3d t12 = -Rwl.inverse() * (twl - twc);//?

        //保存为相机数据
        data_selection::cam_data cam_tmp;
        cam_tmp.start_t = last_time; //前一时刻相机时间
        cam_tmp.end_t = curr_time; //当前采集时间
        //cam_tmp.theta_y = theta_y;
        cam_tmp.deltaTheta_21 = deltaTheta_21; //
        cam_tmp.axis_21 = axis_21; //相机相对旋转（轴 Rc2c1）
//        cout<<"axis(1): "<<axis(1)<<endl;
        cam_tmp.R21 =  R21;//相机相对旋转(前帧为参考 Rc2c1)
        cam_tmp.R12 = R12;
        cam_tmp.t12 =  t12;//相机相对位移(前帧为参考 t(c1, c1c2))
        cam_tmp.Rwc1 = Rwl;
        cam_tmp.twc1 = twl;
        cam_tmp.Rwc2 = Rwc;
        cam_tmp.twc2 = twc;
        camDatas.push_back(cam_tmp);
        

        Rwl = Rwc;
        twl = twc;
        last_time = curr_time;    
    }
    pose_txt.close();

    return true;
}

bool readIMUData(string path)
{
    ifstream imu_txt;
    imu_txt.open(path,ios::in);
    if (!imu_txt.is_open())
    {
        cout << "读取imu文件失败" << endl;
        return false;
    }
    string str;
    bool first_line = true;
    Eigen::Matrix3d Rwl;
    Eigen::Vector3d twl;
    while (getline(imu_txt,str)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        stringstream line_info(str);
        string temp_str;
        vector<string> temp_vec;
        while (line_info >> temp_str)
            temp_vec.push_back(temp_str);

        data_selection::imu_data imuData;
        imuData.time = stod(temp_vec[0]);
        imuData.acc_x = stod(temp_vec[1]);
        imuData.acc_y = stod(temp_vec[2]);
        imuData.acc_z = stod(temp_vec[3]);
        imuData.angular_x = stod(temp_vec[4]);
        imuData.angular_y = stod(temp_vec[5]);
        imuData.angular_z = stod(temp_vec[6]);
        imuDatas.push_back(imuData);
    }
    imu_txt.close();
    return true;
}

//for test
bool readOdomData(string path)
{
    ifstream ground_txt;
    ground_txt.open(path,ios::in);
    if (!ground_txt.is_open())
    {
        cout << "读取odom文件失败" << endl;
        return 0;
    }
    string str;
    bool first_line = true;
    while (getline(ground_txt,str))
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }
        stringstream line_info(str);
        string temp_str;
        vector<string> temp_vec;
        while (line_info >> temp_str)
            temp_vec.push_back(temp_str);

        data_selection::odo_data_test odo_tmp;
        odo_tmp.time = stod(temp_vec[0]);
        odo_tmp.tvec(0) = stod(temp_vec[1]);
        odo_tmp.tvec(1) = stod(temp_vec[2]);
        odo_tmp.tvec(2) = stod(temp_vec[3]);
        odo_tmp.q.x() = stod(temp_vec[4]);
        odo_tmp.q.y() = stod(temp_vec[5]);
        odo_tmp.q.z() = stod(temp_vec[6]);
        odo_tmp.q.w() = stod(temp_vec[7]);

        groundDatas.push_back(odo_tmp);
    }
    ground_txt.close();

    return 1;
}

int main(int argc, char **argv)
{
    //0,初始化
    cv::FileStorage fsSettings("../config.yaml", cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    //t_imu = t_cam + timeShift
    double timeShift = 0.0;
    string cam_pose_path, imu_data_path, odo_data_path;
    fsSettings["time_shift"] >> timeShift;
    fsSettings["cam_pose_path"] >> cam_pose_path;
    fsSettings["imu_data_path"] >> imu_data_path;
    fsSettings["odo_data_path"] >> odo_data_path;

    //1.1,读取平面camera pose数据
    bool success = readCamPose(cam_pose_path, timeShift);
    if (!success)
        return 0;
    //1.2,读取IMU数据
    success = readIMUData(imu_data_path);
    if (!success)
        return 0;
    
    if (!success)
        return 0;
    //1.3，开始计算RCB
    Eigen::Matrix3d Rci;
    calc_process(Rci);
    cout<<"Ric result: "<<Rci.inverse().eulerAngles(2,1,0).transpose()*RAD2DEG<<endl;

    //2.1,读取平面camera pose数据
    string cam_pose_path2, imu_data_path2;
    fsSettings["cam_pose_path2"] >> cam_pose_path2;
    fsSettings["imu_data_path2"] >> imu_data_path2;
    fsSettings.release();
    success = readCamPose2(cam_pose_path2, timeShift);
    if (!success)
        return 0;
    success = readIMUData2(imu_data_path2);
    if (!success)
        return 0;

    //2.2 计算PCB
    calc_process2(Rci);
    return 0;
}
