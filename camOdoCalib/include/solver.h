#ifndef CALIB_SOLVER_H
#define CALIB_SOLVER_H

//#include "utils.h"
//#include "csm/csm_all.h"
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

public:

  void solveGyroscopeBias(std::vector<data_selection::sync_data> &calib_data,
                          Eigen::Matrix3d Rcb);

  void solveRcb(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                        Eigen::Matrix3d Ryx_cam,
                        Eigen::Matrix3d Ryx_imu,
                        Eigen::Matrix3d &Rci);

  //求解Pcb
  void solvePcb(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                           const Eigen::Matrix3d Rcb,
                           Eigen::Vector3d &gc);
  void solvePcb_test2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                        Eigen::Matrix3d Ryx_cam,
                        Eigen::Matrix3d Ryx_imu,
                        Eigen::Matrix3d &Rci);
  void solvePcb_test3(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                                const Eigen::Matrix3d Rcb,
                                Eigen::Vector3d &gc);
  //优化Pcb
  void RefinePcb(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                    Eigen::Matrix3d Rcb,
                    Eigen::Vector3d &gc);

  void RefinePcb_test1(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                      Eigen::Matrix3d Rcb,
                      Eigen::Vector3d &gc);

  void RefinePcb_test2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                            Eigen::Matrix3d Rcb,
                            Eigen::Vector3d &gc);
  void RefinePcb_test3(std::vector<std::vector<data_selection::sync_data> > &calib_data,
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
