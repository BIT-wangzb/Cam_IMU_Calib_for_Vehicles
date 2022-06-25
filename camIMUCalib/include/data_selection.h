#ifndef DATA_SELECTION_H
#define DATA_SELECTION_H

#endif

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include <fstream>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

class data_selection
{
public:
    struct odo_data_test
    {
        double time;
        Eigen::Vector3d tvec;
        Eigen::Quaterniond q;
    };
    struct imu_data
    {
        double time;
        double acc_x;
        double acc_y;
        double acc_z;
        double angular_x;
        double angular_y;
        double angular_z;
    };
    struct cam_data
    {
        double start_t;
        double end_t;
        //double theta_y;
        double deltaTheta_21;
        Eigen::Vector3d axis_21;
        Eigen::Matrix3d R21;
        Eigen::Matrix3d R12;
        Eigen::Vector3d t12;
        Eigen::Matrix3d Rwc1;
        Eigen::Vector3d twc1;
        Eigen::Matrix3d Rwc2;
        Eigen::Vector3d twc2;
    };
    struct imu_preIntegration
    {
        Eigen::Vector3d P;
        Eigen::Vector3d V;
        Eigen::Matrix3d R;
        Eigen::Vector3d Ba;
        Eigen::Vector3d Bg;
        Eigen::Vector3d axis;
        double delta_angle;
        Eigen::Vector3d tlc;
    };
    /*
   * \brief The sync_data struct. Used for
   * storing synchronized data.
   */
  struct sync_data {
    // Period
    //imu
    double T;//同步时间片段
    Eigen::Vector3d P;
    Eigen::Vector3d V;
    Eigen::Matrix3d R;
    Eigen::Vector3d Ba;
    Eigen::Vector3d Bg;
    Eigen::Vector3d axis_imu;
    Eigen::Vector3d t21_imu;
    double angle_imu;
    Eigen::Quaterniond q21_imu;
    double sum_dt;
    Eigen::Matrix<double,15,15> jacobian_;

    Eigen::Vector3d cam_t12;
    double scan_match_results[3];

    Eigen::Vector3d t21_cam;
    Eigen::Quaterniond q21_cam;
    Eigen::Quaterniond q12_cam;
    Eigen::Matrix3d Rwc1_cam;
    Eigen::Vector3d twc1_cam;
    Eigen::Matrix3d Rwc2_cam;
    Eigen::Vector3d twc2_cam;
    double angle;
    Eigen::Vector3d axis;
    double startTime;
    double endTime;

  };

  void Init();
  void startPosAlign(std::vector<imu_data>& imuDatas, std::vector<cam_data>& camDatas);
  void selectData(std::vector<imu_data> &imuDatas, std::vector<cam_data> &camDatas,
        std::vector<data_selection::sync_data> &sync_result);

  void camOdoAlign(std::vector<imu_data> &imuDatas,
                     std::vector<cam_data> &camDatas,
                     std::vector<imu_preIntegration> &preIntegrationDatas,
                     std::vector<sync_data> &sync_result);
  void midPointIntegration(double T, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                             Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian);
  void propagate(double T, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1);
  void reprogate(std::vector<sync_data> &tmp_sync_data);
  void processIMU(double _dt, const Eigen::Vector3d &linear_acceleration,
                    const Eigen::Vector3d &angular_velocity);
  void ResetState(double time, Eigen::Vector3d imu_acc, Eigen::Vector3d imu_vel,
                  Eigen::Vector3d Ba, Eigen::Vector3d Bg);



  data_selection();

  //for test
  struct sync_data_test {
      //odo
      double angle_odo;
      Eigen::Vector3d axis_odo;
      Eigen::Matrix3d R12_odo;
      Eigen::Matrix3d R21_odo;
      Eigen::Vector3d t21_odo;
      Eigen::Vector3d t12_odo;
      Eigen::Quaterniond q21_odo;
      Eigen::Quaterniond q12_odo;

      //camera data : x y yaw , x  y from tlc (not tcl)
      Eigen::Vector3d cam_t12;
      double scan_match_results[3];// correct lx ly by R_x
      // Estimated rototranslation based on odometry params.
      double o[3];
      // Estimated disagreement  sm - est_sm
      double est_sm[3];
      double err_sm[3]; //  s  - (-) l (+) o (+) l
      // Other way to estimate disagreement:   l (+) s  - o (+) l
      double err[3];
      int mark_as_outlier;
      //tcl_cam and qcl_cam are original data(not correted by R_x)
      Eigen::Vector3d t21_cam;//06/06
      Eigen::Quaterniond q21_cam;
      Eigen::Quaterniond q12_cam;
      double angle;
      Eigen::Vector3d axis;
      double startTime;
      double endTime;

  };
  void startPosAlign_test(std::vector<odo_data_test>& odoDatas,
                          std::vector<cam_data>& camDatas);

  void camOdoAlign_test(std::vector<odo_data_test> &odoDatas,
                        std::vector<cam_data>& camDatas,
                        std::vector<data_selection::sync_data_test> &sync_result);

  void selectData_test(std::vector<odo_data_test> &imuDatas, std::vector<cam_data> &camDatas,
                    std::vector<data_selection::sync_data_test> &sync_result);

    
private:
    Eigen::Vector3d g;
    Eigen::Vector3d prev_acc, prev_gyr;
    double prev_time;
//    Eigen::Vector3d acc_1, gyr_1;
    //预积分值
    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;
    Eigen::Vector3d linearized_ba, linearized_bg;
    Eigen::Vector3d axis;
    double deltaTheta_21;
    int frame_count;

    std::vector<imu_data> _imuDatas;

public:
    Eigen::Matrix<double,15,15> jacobian, covariance;
    Eigen::Matrix<double,18,18> noise;
    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyr_bias;
};
