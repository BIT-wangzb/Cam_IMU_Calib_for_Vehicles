#ifndef CALIB_SOLVER_H
#define CALIB_SOLVER_H

#include "utils.h"
#include "csm/csm_all.h"
#include "data_selection.h"

class cSolver{ 

public:

  cSolver();

  struct solver_params {
    int mode;

    double max_cond_number;

    int outliers_iterations;
    double outliers_percentage;
  };

  struct calib_result {
    double radius_l, radius_r;
    double axle;

    /** externel paras lx ly theta between Cam and odo */
    double l[3];
  };

  bool solve(const std::vector<data_selection::sync_data> &calib_data, int mode, double max_cond_number, struct calib_result& res);

  void calib(std::vector<data_selection::sync_data> &calib_data, int outliers_iterations,calib_result &res);
  //void calib(std::vector<data_selection::sync_data> &calib_data, int outliers_iterations);

public:

  Eigen::VectorXd full_calibration_min(const Eigen::MatrixXd &M);

  Eigen::VectorXd numeric_calibration(const Eigen::MatrixXd &H);

  double calculate_error(const Eigen::VectorXd &x, const Eigen::MatrixXd &M);

  Eigen::VectorXd x_given_lambda(const Eigen::MatrixXd &M, const double &lambda, const Eigen::MatrixXd &W);

  void compute_disagreement(data_selection::sync_data &calib_data, const struct calib_result &res);

  void estimate_noise(std::vector<data_selection::sync_data> &calib_data, const struct calib_result &res,
                      double &std_x, double &std_y, double &std_th);

  double calculate_sd(const double array[], const int s, const int e);

  Eigen::MatrixXd compute_fim(const std::vector<data_selection::sync_data> &calib_data, const struct calib_result &res,
                   const Eigen::Matrix3d &inf_sm);


  void solveGyroscopeBias(std::vector<data_selection::sync_data> &calib_data,
                          Eigen::Matrix3d Rcb);

  void solveOtherResult(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                        Eigen::Matrix3d Ryx_cam,
                        Eigen::Matrix3d Ryx_imu,
                        Eigen::Matrix3d &Rci);
  void solveOtherResult2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                          Eigen::Matrix3d Ryx_cam,
                          Eigen::Matrix3d Ryx_imu,
                          Eigen::Matrix3d &Rci);
  void solveOtherResult_test2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                           const Eigen::Matrix3d Rcb,
                           Eigen::Vector3d &gc);
  void solveOtherResult_test3(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                                const Eigen::Matrix3d Rcb,
                                Eigen::Vector3d &gc);
  void solveOtherResult_test4(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                                const Eigen::Matrix3d Rcb,
                                Eigen::Vector3d &gc);
  void RefineResult(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                    Eigen::Matrix3d Rcb,
                    Eigen::Vector3d &gc);

  void RefineResult_test1(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                      Eigen::Matrix3d Rcb,
                      Eigen::Vector3d &gc);

  void RefineResult_test2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                            Eigen::Matrix3d Rcb,
                            Eigen::Vector3d &gc);
  void RefineResult_test3(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                            Eigen::Matrix3d Rcb,
                            Eigen::Vector3d &gc);

  void solveOtherResult_gc(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                          Eigen::Matrix3d Ryx_cam,
                          Eigen::Matrix3d Ryx_imu);
  void solveOtherResult1(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                          Eigen::Matrix3d Ryx_cam,
                          Eigen::Matrix3d Ryx_imu,
                          Eigen::Matrix3d &Rci);
  void solveGravity(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                        Eigen::Matrix3d Rci,
                        Eigen::Vector3d &gc);

  void RefineRotation(std::vector<data_selection::sync_data> &calib_data,
                      Eigen::Matrix3d &_Rcb);

  //for bit test
  void solveOtherResult_test(std::vector<data_selection::sync_data_test> &calib_data,
                          Eigen::Matrix3d Ryx_cam,
                          Eigen::Matrix3d Ryx_odo);

  //tool
  double GetVariance(std::vector<double> &vector_);

public:
    int K1,K2;
    int ws;//window size

};

#endif
