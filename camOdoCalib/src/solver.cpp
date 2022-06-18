#include <iostream>
#include "solver.h"
#include "solveQyx.h"
#include "utils.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <sophus/so3.hpp>

#include "ErrorFunction.h"
#include "utility.h"
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

void cSolver::calib(std::vector<data_selection::sync_data> &calib_data,
                    int outliers_iterations,calib_result &res)
{
  std::cout << std::endl << "there are " << calib_data.size() << " datas for calibrating!" << std::endl;
  std::vector<data_selection::sync_data> calib_history[outliers_iterations + 1];  
  //calib_result res;
  std::vector<data_selection::sync_data> outliers_data;

  for (int iteration = 0; iteration <= outliers_iterations; iteration++)
  {
    calib_history[iteration] = calib_data;

    // Calibration
    if (!solve(calib_data, 0, 75, res))
    {
      std::cout << colouredString("Failed calibration.", RED, BOLD) << std::endl;
      continue;
    }
    else
    {
      /*std::cout << '\n' << "-------Calibration Results-------" << '\n' << "Axle between wheels: " << res.axle << '\n' << "cam-odom x: " << res.l[0] << '\n'
                << "cam-odom y: " << res.l[1] << '\n' << "cam-odom yaw: " << res.l[2] << '\n' << "Left wheel radius: " << res.radius_l << '\n'
                << "Right wheel radius: " << res.radius_r << std::endl;*/
    }

    // Compute residuals
    for (unsigned int i = 0; i < calib_data.size(); i++)
    {
      //compute o[3] err[3] err_sm[3] in calib_data
      compute_disagreement(calib_data[i], res);
    }

    // Sort residuals and compute thresholds
    std::vector<double> err_theta;
    for (unsigned int i = 0; i < calib_data.size(); i++)
    {
      err_theta.push_back(fabs(calib_data[i].err[2]));
    }

    std::vector<double> err_xy;
    for (unsigned int i = 0; i < calib_data.size(); i++)
    {
      double x = calib_data[i].err[0];
      double y = calib_data[i].err[1];
      err_xy.push_back(sqrt(x*x + y*y));
    }

    std::vector<double> err_theta_sorted(err_theta);
    std::sort(err_theta_sorted.begin(), err_theta_sorted.end());

    std::vector<double> err_xy_sorted(err_xy);
    std::sort(err_xy_sorted.begin(), err_xy_sorted.end());

    int threshold_index = (int) std::round ((0.90) * calib_data.size());
    double threshold_theta = err_theta_sorted[threshold_index];
    double threshold_xy = err_xy_sorted[threshold_index];

    int noutliers = 0;
    int noutliers_theta = 0;
    int noutliers_xy = 0;
    int noutliers_both = 0;

    for (unsigned int i = 0; i < calib_data.size(); i++)
    {
      int xy = err_xy[i] > threshold_xy;
      int theta = err_theta[i] > threshold_theta;

      calib_data[i].mark_as_outlier = xy | theta;

      if(xy) noutliers_xy++;
      if(theta) noutliers_theta++;
      if(xy && theta) noutliers_both ++;
      if(xy || theta) noutliers ++;
    }

    std::vector<data_selection::sync_data> n;
    for (unsigned int i = 0; i < calib_data.size(); i++)
    {
      if (!calib_data[i].mark_as_outlier) n.push_back(calib_data[i]);
      else outliers_data.push_back(calib_data[i]); //zdf 2019.06.10
    }

    calib_data = n;
  }

  double laser_std_x, laser_std_y, laser_std_th;
  int estimate_with = 1;
  estimate_noise(calib_history[estimate_with], res, laser_std_x, laser_std_y, laser_std_th);

    /* Now compute the FIM */
    // 论文公式 9 误差的协方差
//  std::cout <<'\n' << "Noise: " << '\n' << laser_std_x << ' ' << laser_std_y
//            << ' ' << laser_std_th << std::endl;

  Eigen::Matrix3d laser_fim = Eigen::Matrix3d::Zero();
  laser_fim(0,0) = (float)1 / (laser_std_x * laser_std_x);
  laser_fim(1,1) = (float)1 / (laser_std_y * laser_std_y);
  laser_fim(2,2) = (float)1 / (laser_std_th * laser_std_th);

  //Eigen::Matrix3d laser_cov = laser_fim.inverse();

  std::cout << '\n' << "-------Calibration Results-------" << '\n' << "Axle between wheels: " << res.axle << '\n' << "cam-odom x: " << res.l[0] << '\n'
                << "cam-odom y: " << res.l[1] << '\n' << "cam-odom yaw: " << res.l[2] << '\n' << "Left wheel radius: " << res.radius_l << '\n'
                << "Right wheel radius: " << res.radius_r << std::endl;

  return;

}

bool cSolver::solve(const std::vector<data_selection::sync_data> &calib_data,
                    int mode, double max_cond_number, calib_result &res)
{
/*!<!--####################		FIRST STEP: estimate J21 and J22  	#################-->*/
  double J21, J22;

  Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
  Eigen::Vector2d g = Eigen::Vector2d::Zero();
  Eigen::Vector2d L_i = Eigen::Vector2d::Zero();

//  std::cout << "A: " << A << ' ' << "g: " << g << std::endl;
//  std::cout << "orz.." << std::endl;

  //double th_i;
  int n = (int)calib_data.size();

  for (int i = 0; i < n; i++) {
    const data_selection::sync_data &t = calib_data[i];
    L_i(0) = t.T * t.velocity_left;
    L_i(1) = t.T * t.velocity_right;
//    std::cout << (L_i * L_i.transpose()) << '\n' << std::endl;
    A = A + (L_i * L_i.transpose());           // A = A + L_i'*L_i;  A symmetric
//    std::cout << (t.scan_match_results[2] * L_i) << std::endl;
    g = (t.scan_match_results[2] * L_i) + g;   // g = g + L_i'*y_i;  sm :{x , y, theta}
//    std::cout << "A: " << A << ' ' << "g: " << g << std::endl;
  }


  // Verify that A isn't singular
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
  //std::cout << "cond "  << cond << std::endl;
  if (cond > max_cond_number)
  {
    std::cout << colouredString("Matrix A is almost singular.", RED, BOLD) << std::endl;
    return 0;
  }

  // Ay = g --> y = inv(A)g; A square matrix;
  Eigen::Vector2d y = Eigen::Vector2d::Zero();
  y = A.colPivHouseholderQr().solve(g);

  std::cout << "J21 = " << y(0) << " , J22 = " << y(1) << std::endl;

  J21 = y(0);
  J22 = y(1);

  if (std::isnan(J21) || std::isnan(J22)) {
    std::cout << colouredString("Can't find J21, J22", RED, BOLD) << std::endl;
    return 0;
  }

/*!<!--############## 		SECOND STEP: estimate the remaining parameters  		########-->*/
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(5, 5);
  Eigen::MatrixXd M2 = Eigen::MatrixXd::Zero(6, 6);
  Eigen::MatrixXd L_k = Eigen::MatrixXd::Zero(2, 5);
  Eigen::MatrixXd L_2k = Eigen::MatrixXd::Zero(2, 6);

  double  cx, cy, cx1, cx2, cy1, cy2, t1, t2;
  double o_theta;
  //double sm[3];

  //int nused = 0;
  for (int k = 0; k < n; k++) {
    const data_selection::sync_data & t = calib_data[k];
    o_theta = t.T * (J21 * t.velocity_left + J22 * t.velocity_right);
    //double w0 = o_theta / t.T;

    if (fabs(o_theta) > 1e-12) {
      t1 = sin(o_theta) / o_theta;
      t2 = (1 - cos(o_theta)) / o_theta;
      /*t1 = cos(o_theta);
        t2 = sin(o_theta);*/
    }
    else {
      t1 = 1;
      t2 = 0;
    }

    cx1 = 0.5 * t.T * (-J21 * t.velocity_left) * t1;
    cx2 = 0.5 * t.T * (J22 * t.velocity_right) * t1;
    cy1 = 0.5 * t.T * (-J21 * t.velocity_left) * t2;
    cy2 = 0.5 * t.T * (J22 * t.velocity_right) * t2;

    if ((mode == 0) || (mode == 1)) {
      cx = cx1 + cx2;
      cy = cy1 + cy2;
      L_k << -cx, (1 - cos(o_theta)), sin(o_theta), t.scan_match_results[0], -t.scan_match_results[1],
          -cy, -sin(o_theta), (1 - cos(o_theta)), t.scan_match_results[1], t.scan_match_results[0];
      // M = M + L_k' * L_k; M is symmetric
      M = M + L_k.transpose() * L_k;
    }
    else {
      L_2k << -cx1, -cx2, (1 - cos(o_theta)), sin(o_theta), t.scan_match_results[0], -t.scan_match_results[1],
          -cy1, -cy2, -sin(o_theta), (1 - cos(o_theta)), t.scan_match_results[1], t.scan_match_results[0];
      M2 = M2 + L_2k.transpose() * L_2k;
    }
  }

  double est_b=0.0, est_d_l=0.0, est_d_r=0.0, laser_x=0.0, laser_y=0.0, laser_th=0.0;
  Eigen::VectorXd x;
  switch(mode)
  {
  case 0:
  {
    x = full_calibration_min(M);

    est_b = x(0);
    est_d_l = 2 * (-est_b * J21);
    est_d_r = 2 * (est_b * J22);
    laser_x = x(1);
    laser_y = x(2);
    laser_th = atan2(x(4), x(3));
    break;
  }
  default:
    break;
  }
  res.axle = est_b;
  res.radius_l = est_d_l/2;
  res.radius_r = est_d_r/2;
  res.l[0] = laser_x;
  res.l[1] = laser_y;
  res.l[2] = laser_th;

  return 1;
}

Eigen::VectorXd cSolver::full_calibration_min(const Eigen::MatrixXd &M)
{
  double m11 = M(0, 0);
  double m13 = M(0, 2);
  double m14 = M(0, 3);
  double m15 = M(0, 4);
  double m22 = M(1, 1);
  // double m25 = M(1, 4);
  double m34 = M(2, 3);
  double m35 = M(2, 4);
  double m44 = M(3, 3);
  // double m55 = M(4, 4);
  double a, b, c;

  a = m11 * pow(m22,2) - m22 * pow(m13,2);
  b = 2 * m13 * m22 * m35 * m15 - pow(m22,2) * pow(m15,2) - 2 * m11 * m22 * pow(m35, 2)
      + 2 * m13 * m22 * m34 * m14 - 2 * m22 * pow(m13,2) * m44 - pow(m22,2) * pow(m14,2)
      + 2 * m11 * pow(m22,2) * m44 + pow(m13,2) * pow(m35,2) - 2 * m11 * m22 * pow(m34,2)
      + pow(m13,2) * pow(m34,2);
  c = -2 * m13 * pow(m35, 3) * m15 - m22 * pow(m13,2) * pow(m44,2) + m11 * pow(m22,2) * pow(m44,2)
      + pow(m13,2) * pow(m35,2) * m44 + 2 * m13 * m22 * m34 * m14 * m44
      + pow(m13,2) * pow(m34,2) * m44 - 2 * m11 * m22 * pow(m34,2) * m44
      - 2 * m13 * pow(m34,3) * m14 - 2 * m11 * m22 * pow(m35,2) * m44
      + 2 * m11 * pow(m35,2) * pow(m34,2) + m22 * pow(m14,2) * pow(m35,2)
      - 2 * m13 * pow(m35,2) * m34 * m14 - 2 * m13 * pow(m34, 2) * m35 * m15
      + m11 * pow(m34,4) + m22 * pow(m15,2) * pow(m34,2)
      + m22 * pow(m35,2) * pow(m15,2) + m11 * pow(m35,4)
      - pow(m22,2) * pow(m14,2) * m44 + 2 * m13 * m22 * m35 * m15 * m44
      + m22 * pow(m34,2) * pow(m14,2) - pow(m22,2) * pow(m15,2) * m44;

    /* 	Calcolo radice del polinomio 	*/
  if ((pow(b,2) - 4 * a * c) >= 0)
  {
    double r0 = (-b - sqrt(pow(b,2) - 4 * a * c)) / (2 * a);
    double r1 = (-b + sqrt(pow(b,2) - 4 * a * c)) / (2 * a);

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(5, 5);
    W(3,3) = 1;
    W(4,4) = 1;
    Eigen::VectorXd x0 = x_given_lambda(M, r0, W);//  求出来的状态量
    Eigen::VectorXd x1 = x_given_lambda(M, r1, W);

    double e0 = calculate_error(x0, M);
    double e1 = calculate_error(x1, M);

    return e0 < e1 ? x0 : x1;
  }
  else {
    std::cout << colouredString("Imaginary solution!", RED, BOLD) << std::endl;
    return Eigen::VectorXd(5);
  }
}

double cSolver::calculate_error(const Eigen::VectorXd &x, const Eigen::MatrixXd &M)
{
  double error;
  Eigen::VectorXd tmp = Eigen::VectorXd::Zero(x.rows());
  tmp = M * x;
  error = x.transpose() * tmp;

  return error;
}

Eigen::VectorXd cSolver::x_given_lambda(const Eigen::MatrixXd &M, const double &lambda, const Eigen::MatrixXd &W)
{
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(5,5);
  Eigen::MatrixXd ZZ = Eigen::MatrixXd::Zero(5,5);

  Z = M + lambda * W;

  ZZ = Z.transpose() * Z;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(ZZ);

//  Eigen::EigenSolver<Eigen::MatrixXd> es;
//  es.compute(ZZ);

//  Eigen::VectorXd eigenvalues = es.pseudoEigenvalueMatrix();
//  Eigen::MatrixXd eigenvectors = es.pseudoEigenvectors();
  Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
  Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors();
  //int colnum = eigenvalues.minCoeff();
  Eigen::VectorXd v0 = eigenvectors.col(eigenvalues.minCoeff());// min value
  Eigen::Vector2d tmp_v = Eigen::Vector2d::Zero(2);
  tmp_v(0) = v0(3);
  tmp_v(1) = v0(4);

  double norm = tmp_v.norm();
  double coeff = (v0(0) >= 0 ? 1 : -1) / norm;
  v0 = coeff * v0;
  return v0;
}

void cSolver::compute_disagreement(data_selection::sync_data &calib_data, const calib_result &res)
{
  double J11 = res.radius_l / 2;
  double J12 = res.radius_r / 2;
  double J21 = - res.radius_l / res.axle;
  double J22 = res.radius_r / res.axle;

  double speed = J11 * calib_data.velocity_left + J12 * calib_data.velocity_right;
  double omega = J21 * calib_data.velocity_left + J22 * calib_data.velocity_right;

  double o_theta = calib_data.T * omega;

  double t1, t2;
  if (fabs(o_theta) > 1e-12)
  {
    t1 = sin(o_theta) / o_theta;
    t2 = (1 - cos(o_theta)) / o_theta;
  }
  else
  {
    t1 = 1;
    t2 = 0;
  }

  calib_data.o[0] = t1 * speed * calib_data.T;
  calib_data.o[1] = t2 * speed * calib_data.T;
  calib_data.o[2] = o_theta;

  double l_plus_s[3];
  double o_plus_l[3];

  oplus_d(res.l, calib_data.scan_match_results, l_plus_s);
  oplus_d(calib_data.o, res.l, o_plus_l);

  for (unsigned int i = 0; i < 3; i++)
  {
    calib_data.err[i] = l_plus_s[i] - o_plus_l[i]; //err = (l+s) - (r+l), here o:odom, s:cam, l:externel paras
  }

  pose_diff_d(o_plus_l, res.l, calib_data.est_sm);

  // est_sm 就是公式 9 ， 从轮速计估计的增量 通过外参数估计的 激光坐标系两时刻之间的增量
  for (unsigned int i = 0; i < 3; i++)
  {
    // err_sm = s = -l + r + l;
    calib_data.err_sm[i] = calib_data.est_sm[i] - calib_data.scan_match_results[i];
  }

}

void cSolver::estimate_noise(std::vector<data_selection::sync_data> &calib_data,
                             const calib_result &res, double &std_x, double &std_y,
                             double &std_th)
{
  int n = calib_data.size();
  double err_sm[3][n];
  for(unsigned int i = 0; i < calib_data.size(); i++)
  {
    compute_disagreement(calib_data[i], res);
    err_sm[0][i] = calib_data[i].err_sm[0];
    err_sm[1][i] = calib_data[i].err_sm[1];
    err_sm[2][i] = calib_data[i].err_sm[2];
  }

  std_x = calculate_sd(err_sm[0], 0, n);
  std_y = calculate_sd(err_sm[1], 0, n);
  std_th = calculate_sd(err_sm[2], 0, n);
}

double cSolver::calculate_sd(const double array[], const int s, const int e)
{
  double sum = 0;
  double mean = 0;
  double sd = 0;
  for (int i = s; i < e; i++)
  {
    sum += array[i];
  }
  mean = sum / (float)e;

  for (int i = s; i < e; i++)
  {
    sd += pow((array[i] - mean), 2);
  }

  sd = sqrt(sd / (float)e);

  return sd;
}

Eigen::MatrixXd cSolver::compute_fim(const std::vector<data_selection::sync_data> &calib_data,
                          const calib_result &res, const Eigen::Matrix3d &inf_sm)
{

  Eigen::MatrixXd fim = Eigen::MatrixXd::Zero(6,6);

  /* Compute the derivative using the 5-point rule (x-h, x-h/2, x,
     x+h/2, x+h). Note that the central point is not used. */
  const double eps = 1e-3;
  for (size_t i = 0; i < calib_data.size(); ++i) {

    Eigen::Matrix<double, 3, 6> num_jacobian;

    // jacobian_radius_l
    {
      data_selection::sync_data data0= calib_data[i],data1= calib_data[i],data2= calib_data[i],data3 = calib_data[i];
      calib_result res0 = res, res1 = res, res2 = res, res3 = res;
      res0.radius_l -= eps;
      res1.radius_l += eps;
      res2.radius_l -= eps/2.;
      res3.radius_l += eps/2.;
      compute_disagreement(data0,res0);
      compute_disagreement(data1,res1);
      compute_disagreement(data2,res2);
      compute_disagreement(data3,res3);
      for (int k = 0; k < 3; ++k) {
        double r3 = 0.5 * (data1.est_sm[k] - data0.est_sm[k]);
        double r5 = (4.0 / 3.0) * (data3.est_sm[k] - data2.est_sm[k]) - (1.0 / 3.0) * r3;
        num_jacobian(k,0) = r5 / eps ;
      }
    }

    // jacobian_radius_r
    {
      data_selection::sync_data data0= calib_data[i],data1= calib_data[i],data2= calib_data[i],data3 = calib_data[i];
      calib_result res0 = res, res1 = res, res2 = res, res3 = res;
      res0.radius_r -= eps;
      res1.radius_r += eps;
      res2.radius_r -= eps/2.;
      res3.radius_r += eps/2.;
      compute_disagreement(data0,res0);
      compute_disagreement(data1,res1);
      compute_disagreement(data2,res2);
      compute_disagreement(data3,res3);
      for (int k = 0; k < 3; ++k) {
        double r3 = 0.5 * (data1.est_sm[k] - data0.est_sm[k]);
        double r5 = (4.0 / 3.0) * (data3.est_sm[k] - data2.est_sm[k]) - (1.0 / 3.0) * r3;
        num_jacobian(k,1) = r5 / eps ;
      }
    }
    // jacobian_axle
    {
      data_selection::sync_data data0= calib_data[i],data1= calib_data[i],data2= calib_data[i],data3 = calib_data[i];
      calib_result res0 = res, res1 = res, res2 = res, res3 = res;
      res0.axle -= eps;
      res1.axle += eps;
      res2.axle -= eps/2.;
      res3.axle += eps/2.;
      compute_disagreement(data0,res0);
      compute_disagreement(data1,res1);
      compute_disagreement(data2,res2);
      compute_disagreement(data3,res3);
      for (int k = 0; k < 3; ++k) {
        double r3 = 0.5 * (data1.est_sm[k] - data0.est_sm[k]);
        double r5 = (4.0 / 3.0) * (data3.est_sm[k] - data2.est_sm[k]) - (1.0 / 3.0) * r3;
        num_jacobian(k,2) = r5 / eps ;
      }
    }
    // jacobian_l[0]
    {
      data_selection::sync_data data0= calib_data[i],data1= calib_data[i],data2= calib_data[i],data3 = calib_data[i];
      calib_result res0 = res, res1 = res, res2 = res, res3 = res;
      res0.l[0] -= eps;
      res1.l[0] += eps;
      res2.l[0] -= eps/2.;
      res3.l[0] += eps/2.;
      compute_disagreement(data0,res0);
      compute_disagreement(data1,res1);
      compute_disagreement(data2,res2);
      compute_disagreement(data3,res3);
      for (int k = 0; k < 3; ++k) {
        double r3 = 0.5 * (data1.est_sm[k] - data0.est_sm[k]);
        double r5 = (4.0 / 3.0) * (data3.est_sm[k] - data2.est_sm[k]) - (1.0 / 3.0) * r3;
        num_jacobian(k,3) = r5 / eps ;
      }
    }
    // jacobian_l[1]
    {
      data_selection::sync_data data0= calib_data[i],data1= calib_data[i],data2= calib_data[i],data3 = calib_data[i];
      calib_result res0 = res, res1 = res, res2 = res, res3 = res;
      res0.l[1] -= eps;
      res1.l[1] += eps;
      res2.l[1] -= eps/2.;
      res3.l[1] += eps/2.;
      compute_disagreement(data0,res0);
      compute_disagreement(data1,res1);
      compute_disagreement(data2,res2);
      compute_disagreement(data3,res3);
      for (int k = 0; k < 3; ++k) {
        double r3 = 0.5 * (data1.est_sm[k] - data0.est_sm[k]);
        double r5 = (4.0 / 3.0) * (data3.est_sm[k] - data2.est_sm[k]) - (1.0 / 3.0) * r3;
        num_jacobian(k,4) = r5 / eps ;
      }
    }
    // jacobian_l[2]
    {
      data_selection::sync_data data0= calib_data[i],data1= calib_data[i],data2= calib_data[i],data3 = calib_data[i];
      calib_result res0 = res, res1 = res, res2 = res, res3 = res;
      res0.l[2] -= eps;
      res1.l[2] += eps;
      res2.l[2] -= eps/2.;
      res3.l[2] += eps/2.;
      compute_disagreement(data0,res0);
      compute_disagreement(data1,res1);
      compute_disagreement(data2,res2);
      compute_disagreement(data3,res3);
      for (int k = 0; k < 3; ++k) {
        double r3 = 0.5 * (data1.est_sm[k] - data0.est_sm[k]);
        double r5 = (4.0 / 3.0) * (data3.est_sm[k] - data2.est_sm[k]) - (1.0 / 3.0) * r3;
        num_jacobian(k,5) = r5 / eps ;
      }
    }


    fim += num_jacobian.transpose() * inf_sm * num_jacobian;

  }

  return fim;
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

//求解yaw，gc两个分量和pcb3个分量
//测试该方法，估计pcb两个平移分量，该方法无法成功估计
void cSolver::solveOtherResult1(std::vector<std::vector<data_selection::sync_data> > &calib_data,
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
void cSolver::solveOtherResult_test3(std::vector<std::vector<data_selection::sync_data> > &calib_data,
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

//设Rcb已知，求解gc和pcb 滑窗
void cSolver::solveOtherResult_test4(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                                     const Eigen::Matrix3d Rcb,
                                     Eigen::Vector3d &gc)
{
    static ofstream f_pcb;
    f_pcb.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/tcb_estimation1.txt");
    f_pcb<<std::fixed<<std::setprecision(6);
    f_pcb<<"sliding window: "<<ws<<endl;
    f_pcb<<"solveOtherResult_test4 zed_tci = [0.0224, -0.0112, -0.0165]\n";
    f_pcb<<"solveOtherResult_test4 oxts_tci = [-0.081, 1.03, -1.45]\n";
    f_pcb <<"gcx, gcy, gcz, pcb_x, pcb_y, pcb_z\n";

    double err_min = std::numeric_limits<double>::max();
    double best_x1, best_x2, best_x3, best_x4, best_x5, best_x6;;

    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }
    Eigen::Vector3d gc_init = Eigen::Vector3d::Zero();
    Eigen::Vector3d pcb_init = Eigen::Vector3d::Zero();

    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        f_pcb <<"---------\n";
        for (int motionId = 0; motionId < calib_data[segmentId].size()-ws-1; ++motionId)
        {
            Eigen::MatrixXd tmp_G(ws*3,6);
            tmp_G.setZero();
            Eigen::MatrixXd tmp_b(ws*3,1);
            tmp_b.setZero();

            int valid_num = 0;
            for (int k = 0; k < ws; ++k)
            {
                double axis1 = calib_data[0][motionId+k].axis(0);
                double axis2 = calib_data[0][motionId+k].axis(1);
                double axis3 = calib_data[0][motionId+k].axis(2);
                if(fabs(axis1) > 0.3 && fabs(axis2) > 0.3 && fabs(axis3) > 0.3)
                    valid_num++;
                Eigen::Vector3d pc1 = calib_data[segmentId][motionId+k].twc1_cam;
                Eigen::Vector3d pc2 = calib_data[segmentId][motionId+k].twc2_cam;
                Eigen::Vector3d pc3 = calib_data[segmentId][motionId+k+1].twc2_cam;

                Eigen::Vector3d t12_imu = calib_data[segmentId][motionId+k].P;
                Eigen::Vector3d t23_imu = calib_data[segmentId][motionId+k+1].P;
                Eigen::Vector3d v12_imu = calib_data[segmentId][motionId+k].V;
                Eigen::Vector3d t12_cam = calib_data[segmentId][motionId+k].cam_t12;
                Eigen::Vector3d t23_cam = calib_data[segmentId][motionId+k+1].cam_t12;

                Eigen::Matrix3d Rwc1_cam = calib_data[segmentId][motionId+k].Rwc1_cam;
                Eigen::Matrix3d Rwc2_cam = calib_data[segmentId][motionId+k].Rwc2_cam;
                Eigen::Matrix3d Rwc3_cam = calib_data[segmentId][motionId+k+1].Rwc2_cam;
                double dT12 = calib_data[segmentId][motionId+k].T;
                double dT23 = calib_data[segmentId][motionId+k+1].T;
                if (dT12 != calib_data[segmentId][motionId+k].sum_dt)
                    cout<<"sum_dt: "<<calib_data[segmentId][motionId+k].sum_dt<<", dT12: "<<dT12<<endl;
                if (dT23 != calib_data[segmentId][motionId+k+1].sum_dt)
                    cout<<"sum_dt: "<<calib_data[segmentId][motionId+k+1].sum_dt<<", dT23: "<<dT23<<endl;
//            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId+k].q21_cam.toRotationMatrix();
//            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+k+1].q21_cam.toRotationMatrix();

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

                tmp_G.block<3,3>(k*3,0) = beta;
                tmp_G.block<3,3>(k*3,3) = phi;
                tmp_b.block<3,1>(k*3,0) = test_b;
            }
            Eigen::VectorXd x;
            x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);

            f_pcb << valid_num<<" ";
            f_pcb << x(0)<<" ";
            f_pcb << x(1)<<" ";
            f_pcb << x(2)<<" ";
            f_pcb << x(3)<<" ";
            f_pcb << x(4)<<" ";
            f_pcb << x(5)<<endl;

            //计算误差
            double err = 0.0;
            gc_init(0) = x(0);
            gc_init(1) = x(1);
            gc_init(2) = x(2);
            pcb_init(0) = x(3);
            pcb_init(1) = x(4);
            pcb_init(2) = x(5);
            valid_num = 0;
            for (int k = 0; k < ws; ++k)
            {
                double axis1 = calib_data[0][motionId+k].axis(0);
                double axis2 = calib_data[0][motionId+k].axis(1);
                double axis3 = calib_data[0][motionId+k].axis(2);
                if(fabs(axis1) > 0.3 && fabs(axis2) > 0.3 && fabs(axis3) > 0.3)
                    valid_num++;
                Eigen::Vector3d pc1 = calib_data[0][motionId+k].twc1_cam;
                Eigen::Vector3d pc2 = calib_data[0][motionId+k].twc2_cam;
                Eigen::Vector3d pc3 = calib_data[0][motionId+k+1].twc2_cam;

                Eigen::Vector3d t12_imu = calib_data[0][motionId+k].P;
                Eigen::Vector3d t23_imu = calib_data[0][motionId+k+1].P;
                Eigen::Vector3d v12_imu = calib_data[0][motionId+k].V;
                Eigen::Vector3d t12_cam = calib_data[0][motionId+k].cam_t12;
                Eigen::Vector3d t23_cam = calib_data[0][motionId+k+1].cam_t12;

                Eigen::Matrix3d Rwc1_cam = calib_data[0][motionId+k].Rwc1_cam;
                Eigen::Matrix3d Rwc2_cam = calib_data[0][motionId+k].Rwc2_cam;
                Eigen::Matrix3d Rwc3_cam = calib_data[0][motionId+k+1].Rwc2_cam;
                double dT12 = calib_data[0][motionId+k].T;
                double dT23 = calib_data[0][motionId+k+1].T;
                Eigen::Matrix3d R21_cam = calib_data[0][motionId+k].q21_cam.toRotationMatrix();
                Eigen::Matrix3d R32_cam = calib_data[0][motionId+k+1].q21_cam.toRotationMatrix();

                Eigen::Matrix3d Jpba12 = calib_data[0][motionId+k].jacobian_.block<3,3>(0,9);
                Eigen::Matrix3d Jpba23 = calib_data[0][motionId+k+1].jacobian_.block<3,3>(0,9);
                Eigen::Matrix3d Jvba12 = calib_data[0][motionId+k].jacobian_.block<3,3>(6,9);


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
                Eigen::Vector3d pred = beta * gc_init + phi * pcb_init;
                Eigen::Vector3d e = pred - test_b;
                err += e.norm();
            }
            if (valid_num > 0.2*ws)
            {
                if (err < err_min)
                {
                    err_min = err;
                    best_x1 = x(0);
                    best_x2 = x(1);
                    best_x3 = x(2);
                    best_x4 = x(3);
                    best_x5 = x(4);
                    best_x6 = x(5);
                    gc(0) = x(0);
                    gc(1) = x(1);
                    gc(2) = x(2);
                }
                f_pcb << err <<" ";
                f_pcb << valid_num<<" ";
                f_pcb << x(0)<<" ";
                f_pcb << x(1)<<" ";
                f_pcb << x(2)<<" ";
                f_pcb << x(3)<<" ";
                f_pcb << x(4)<<" ";
                f_pcb << x(5)<<endl;
            }
        }
    }
    f_pcb <<"min error: "<<err_min<<", best tcb: "<<best_x1<<", "
           <<best_x2<<", "<<best_x3<<", "
           <<best_x4<<", "<<best_x5<<", "
           <<best_x6<<endl;
    f_pcb.close();

}

//设Rcb已知，求解gc和pcb 非滑窗
void cSolver::solveOtherResult_test2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
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
//求解bias，优化重力，平移,滑窗
void cSolver::RefineResult_test1(std::vector<std::vector<data_selection::sync_data>> &calib_data,
                           Eigen::Matrix3d Rcb,
                           Eigen::Vector3d &gc)
{
    double s;
    Eigen::Vector2d theta_xy_last;
    Eigen::Vector3d bias_last, pcb_last;
    theta_xy_last.setZero();
    bias_last.setZero();
    pcb_last.setZero();
    double err_min = std::numeric_limits<double>::max();
    double best_x1, best_x2, best_x3;

    static ofstream f_pcb3;
    f_pcb3.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/refine_tcb1.txt");
    f_pcb3<<std::fixed<<std::setprecision(6);
    f_pcb3<<"sliding window: "<<ws<<endl;
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
    for (int i = 0; i < calib_data[0].size()-1; ++i)
    {
        Eigen::MatrixXd A(3*ws,8);
        Eigen::MatrixXd b(3*ws,1);
        int valid_num = 0;
        for (int k = 0; k < ws; ++k)
        {
            double axis1 = calib_data[0][i+k].axis(0);
            double axis2 = calib_data[0][i+k].axis(1);
            double axis3 = calib_data[0][i+k].axis(2);
            if(fabs(axis1) > 0.3 && fabs(axis2) > 0.3 && fabs(axis3) > 0.3)
                valid_num++;

            Eigen::Vector3d pc1 = calib_data[0][i+k].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[0][i+k].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[0][i+k+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[0][i+k].P;
            Eigen::Vector3d t23_imu = calib_data[0][i+k+1].P;
            Eigen::Vector3d v12_imu = calib_data[0][i+k].V;
            Eigen::Vector3d t12_cam = calib_data[0][i+k].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[0][i+k+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[0][i+k].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[0][i+k].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[0][i+k+1].Rwc2_cam;
            double dT12 = calib_data[0][i+k].T;
            double dT23 = calib_data[0][i+k+1].T;
            Eigen::Matrix3d R21_cam = calib_data[0][i+k].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[0][i+k+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d Jpba12 = calib_data[0][i+k].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jpba23 = calib_data[0][i+k+1].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jvba12 = calib_data[0][i+k].jacobian_.block<3,3>(6,9);


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
            Eigen::Vector3d test_b = -lambda + gamma;

            A.block<3,2>(3*k,0) = beta.block<3,2>(0,0);
            A.block<3,3>(3*k,2) = bias;
//            A.block<3,3>(3*k,2) = phi;
            A.block<3,3>(3*k,5) = phi;
            b.block<3,1>(3*k,0) = test_b;
        }

        Eigen::VectorXd x;
        x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        theta_xy_last = x.head(2);
        bias_last = x.block<3,1>(2,0);
        pcb_last = x.tail(3);

        //计算误差
        double err = 0.0;
        theta_xy_last = x.head(2);
        bias_last = x.block<3,1>(2,0);
        pcb_last = x.tail(3);
        Eigen::Vector3d dtheta;
        dtheta(0) = x(0); dtheta(1) = x(1); dtheta(2) = 0.0;
        Eigen::Matrix3d Rwi_ = RWI*Sophus::SO3d::exp(dtheta).matrix();
        Eigen::Vector3d gc_refined = Rwi_*GI;
        valid_num = 0;
        for (int k = 0; k < ws; ++k)
        {
            double axis1 = calib_data[0][i+k].axis(0);
            double axis2 = calib_data[0][i+k].axis(1);
            double axis3 = calib_data[0][i+k].axis(2);
            if(fabs(axis1) > 0.3 && fabs(axis2) > 0.3 && fabs(axis3) > 0.3)
                valid_num++;
            Eigen::Vector3d pc1 = calib_data[0][i+k].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[0][i+k].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[0][i+k+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[0][i+k].P;
            Eigen::Vector3d t23_imu = calib_data[0][i+k+1].P;
            Eigen::Vector3d v12_imu = calib_data[0][i+k].V;
            Eigen::Vector3d t12_cam = calib_data[0][i+k].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[0][i+k+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[0][i+k].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[0][i+k].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[0][i+k+1].Rwc2_cam;
            double dT12 = calib_data[0][i+k].T;
            double dT23 = calib_data[0][i+k+1].T;
            Eigen::Matrix3d R21_cam = calib_data[0][i+k].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[0][i+k+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d Jpba12 = calib_data[0][i+k].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jpba23 = calib_data[0][i+k+1].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jvba12 = calib_data[0][i+k].jacobian_.block<3,3>(6,9);


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
            Eigen::Vector3d test_b = -lambda + gamma;
            Eigen::Vector3d pcb_init;
            Eigen::Vector3d pred = beta.block<3,2>(0,0) * theta_xy_last
                                    + bias * bias_last + phi * pcb_last;
            Eigen::Vector3d e = pred - test_b;
            err += e.norm();
        }

        if (valid_num > 0.2*ws)
        {
            if (err < err_min)
            {
                err_min = err;
                best_x1 = x(5);
                best_x2 = x(6);
                best_x3 = x(7);
            }
            f_pcb3 << err <<" ";
            f_pcb3 << valid_num<<" ";
            f_pcb3 << x(0)<<" ";
            f_pcb3 << x(1)<<" ";
            f_pcb3 << x(2)<<" ";
            f_pcb3 << x(3)<<" ";
            f_pcb3 << x(4)<<" ";
            f_pcb3 << x(5)<<" ";
            f_pcb3 << x(6)<<" ";
            f_pcb3 << x(7)<<" ";
            f_pcb3 <<"gc = ["<<gc_refined(0);
            f_pcb3 <<", "<<gc_refined(1);
            f_pcb3 <<", "<<gc_refined(2)<<"]"<<endl;
        }
    }
    f_pcb3 <<"min error: "<<err_min<<", best tcb: "<<best_x1<<", "
           <<best_x2<<", "<<best_x3<<endl;
    f_pcb3.close();

}

//优化重力，平移,滑窗
void cSolver::RefineResult_test2(std::vector<std::vector<data_selection::sync_data>> &calib_data,
                                 Eigen::Matrix3d Rcb,
                                 Eigen::Vector3d &gc)
{
    Eigen::Vector2d theta_xy_last;
    theta_xy_last.setZero();
    double err_min = std::numeric_limits<double>::max();
    double best_x1, best_x2, best_x3;

    static ofstream f_pcb3;
    f_pcb3.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/refine_tcb2.txt");
    f_pcb3<<std::fixed<<std::setprecision(6);
    f_pcb3<<"sliding window: "<<ws<<endl;
    f_pcb3<<"RefineResult_test2 right_zed_tci = [-0.0975,-0.0112,-0.0166]\n";
    f_pcb3<<"RefineResult_test2 left_zed_tci = [0.0224, -0.0112, -0.0165]\n";
    f_pcb3<<"RefineResult_test2 oxts_tci = [-0.081, 1.03, -1.45]\n";
    f_pcb3<<"# Num theta_x theta_y pcb_x pcb_y pcb_z\n";

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

    for (int i = 0; i < calib_data[0].size()-ws-1; ++i)
    {
        Eigen::MatrixXd A(3*ws,5);
        Eigen::MatrixXd b(3*ws,1);
        int valid_num = 0;
        for (int k = 0; k < ws; ++k)
        {
            double axis1 = calib_data[0][i+k].axis(0);
            double axis2 = calib_data[0][i+k].axis(1);
            double axis3 = calib_data[0][i+k].axis(2);
            if(fabs(axis1) > 0.3 && fabs(axis2) > 0.3 && fabs(axis3) > 0.3)
                valid_num++;
            Eigen::Vector3d pc1 = calib_data[0][i+k].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[0][i+k].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[0][i+k+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[0][i+k].P;
            Eigen::Vector3d t23_imu = calib_data[0][i+k+1].P;
            Eigen::Vector3d v12_imu = calib_data[0][i+k].V;
            Eigen::Vector3d t12_cam = calib_data[0][i+k].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[0][i+k+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[0][i+k].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[0][i+k].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[0][i+k+1].Rwc2_cam;
            double dT12 = calib_data[0][i+k].T;
            double dT23 = calib_data[0][i+k+1].T;
            Eigen::Matrix3d R21_cam = calib_data[0][i+k].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[0][i+k+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d Jpba12 = calib_data[0][i+k].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jpba23 = calib_data[0][i+k+1].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jvba12 = calib_data[0][i+k].jacobian_.block<3,3>(6,9);


            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = -0.5*(dT12*dT12*dT23 + dT12*dT23*dT23)*RWI*skewGI;//gc
            Eigen::Vector3d Ba;
            cv::FileStorage fs("../config.yaml",cv::FileStorage::READ);
            Ba(0) = fs["acc_bias_x"];
            Ba(1) = fs["acc_bias_y"];
            Ba(2) = fs["acc_bias_z"];
            fs.release();
//            Ba(0) = -0.05; Ba(1) = -0.02; Ba(2) = -0.02;
            Eigen::Matrix3d bias = Rwc2_cam*Rcb*Jpba23*dT12 -
                                   Rwc1_cam*Rcb*Jpba12*dT23 +
                                   Rwc1_cam*Rcb*Jvba12*dT12*dT23;//bias
            Eigen::Vector3d v_Ba = bias * Ba;
            Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*dT23;//pcb

            //gamma
            Eigen::Vector3d gamma_1 =  Rwc1_cam * Rcb * t12_imu * dT23;
            Eigen::Vector3d gamma_2 =  -Rwc2_cam * Rcb * t23_imu * dT12;
            Eigen::Vector3d gamma_3 = -Rwc1_cam * Rcb * v12_imu * dT12 * dT23;
            Eigen::Vector3d gamma_4 = -0.5*RWI*GI*(dT12*dT12*dT23 + dT12*dT23*dT23);
            Eigen::Vector3d gamma = gamma_1 + gamma_2 + gamma_3 + gamma_4;
            Eigen::Vector3d test_b = -lambda + gamma - v_Ba;
            Eigen::Vector3d pcb_init;
            pcb_init(0) = -0.081; pcb_init(1) = 1.03; pcb_init(2) = -1.45;
//            pcb_init(0) = 0.0224; pcb_init(1) = -0.0112; pcb_init(2) = -0.0165;
            Eigen::Vector3d pred = beta * gc + phi * pcb_init;
            Eigen::Vector3d e = pred - test_b;
            double wi = exp(-e.norm()*2);
//            cout<<"wi: "<<wi<<", norm: "<<e.norm()<<endl;
            if (e.norm() > 0.15)
                wi = 0.0;
            beta *= wi;
            phi *= wi;
            test_b *= wi;
            A.block<3,2>(3*k,0) = beta.block<3,2>(0,0);
//            A.block<3,3>(3*k,2) = bias;
//            A.block<3,3>(3*k,2) = phi;
            A.block<3,3>(3*k,2) = phi;
            b.block<3,1>(3*k,0) = test_b;
        }
        Eigen::VectorXd x;
        x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        //计算误差
        double err = 0.0;
        Eigen::Vector3d pcb_refined;
        pcb_refined(0) = x(2);
        pcb_refined(1) = x(3);
        pcb_refined(2) = x(4);
        theta_xy_last = x.head(2);
        Eigen::Vector3d dtheta;
        dtheta(0) = x(0); dtheta(1) = x(1); dtheta(2) = 0.0;
        Eigen::Matrix3d Rwi_ = RWI*Sophus::SO3d::exp(dtheta).matrix();
        Eigen::Vector3d gc_refined = Rwi_*GI;

        f_pcb3 << valid_num<<" ";
        f_pcb3 << x(0)<<" ";
        f_pcb3 << x(1)<<" ";
        f_pcb3 << x(2)<<" ";
        f_pcb3 << x(3)<<" ";
        f_pcb3 << x(4)<<" ";
        f_pcb3 <<"gc = ["<<gc_refined(0);
        f_pcb3 <<", "<<gc_refined(1);
        f_pcb3 <<", "<<gc_refined(2)<<"]"<<endl;
        valid_num = 0;
        for (int k = 0; k < ws; ++k)
        {
            double axis1 = calib_data[0][i+k].axis(0);
            double axis2 = calib_data[0][i+k].axis(1);
            double axis3 = calib_data[0][i+k].axis(2);
            if(fabs(axis1) > 0.3 && fabs(axis2) > 0.3 && fabs(axis3) > 0.3)
            {
                valid_num++;
            }
            Eigen::Vector3d pc1 = calib_data[0][i+k].twc1_cam;
            Eigen::Vector3d pc2 = calib_data[0][i+k].twc2_cam;
            Eigen::Vector3d pc3 = calib_data[0][i+k+1].twc2_cam;

            Eigen::Vector3d t12_imu = calib_data[0][i+k].P;
            Eigen::Vector3d t23_imu = calib_data[0][i+k+1].P;
            Eigen::Vector3d v12_imu = calib_data[0][i+k].V;
            Eigen::Vector3d t12_cam = calib_data[0][i+k].cam_t12;
            Eigen::Vector3d t23_cam = calib_data[0][i+k+1].cam_t12;

            Eigen::Matrix3d Rwc1_cam = calib_data[0][i+k].Rwc1_cam;
            Eigen::Matrix3d Rwc2_cam = calib_data[0][i+k].Rwc2_cam;
            Eigen::Matrix3d Rwc3_cam = calib_data[0][i+k+1].Rwc2_cam;
            double dT12 = calib_data[0][i+k].T;
            double dT23 = calib_data[0][i+k+1].T;
            Eigen::Matrix3d R21_cam = calib_data[0][i+k].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[0][i+k+1].q21_cam.toRotationMatrix();

            Eigen::Matrix3d Jpba12 = calib_data[0][i+k].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jpba23 = calib_data[0][i+k+1].jacobian_.block<3,3>(0,9);
            Eigen::Matrix3d Jvba12 = calib_data[0][i+k].jacobian_.block<3,3>(6,9);


            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = -0.5*(dT12*dT12*dT23 + dT12*dT23*dT23)*RWI*skewGI;//gc
            Eigen::Vector3d Ba;
            cv::FileStorage fs("../config.yaml",cv::FileStorage::READ);
            Ba(0) = fs["acc_bias_x"];
            Ba(1) = fs["acc_bias_y"];
            Ba(2) = fs["acc_bias_z"];
            fs.release();
//            Ba(0) = -0.05; Ba(1) = -0.02; Ba(2) = -0.02;
            Eigen::Matrix3d bias = Rwc2_cam*Rcb*Jpba23*dT12 -
                                   Rwc1_cam*Rcb*Jpba12*dT23 +
                                   Rwc1_cam*Rcb*Jvba12*dT12*dT23;//bias
            Eigen::Vector3d v_Ba = bias * Ba;
            Eigen::Matrix3d phi = (Rwc2_cam - Rwc3_cam)*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*dT23;//pcb

            //gamma
            Eigen::Vector3d gamma_1 =  Rwc1_cam * Rcb * t12_imu * dT23;
            Eigen::Vector3d gamma_2 =  -Rwc2_cam * Rcb * t23_imu * dT12;
            Eigen::Vector3d gamma_3 = -Rwc1_cam * Rcb * v12_imu * dT12 * dT23;
            Eigen::Vector3d gamma_4 = -0.5*RWI*GI*(dT12*dT12*dT23 + dT12*dT23*dT23);
            Eigen::Vector3d gamma = gamma_1 + gamma_2 + gamma_3 + gamma_4;
            Eigen::Vector3d test_b = -lambda + gamma - v_Ba;
            Eigen::Vector3d pcb_init;
            Eigen::Vector3d pred = beta.block<3,2>(0,0) * theta_xy_last
                                    + phi * pcb_refined;
            Eigen::Vector3d e = pred - test_b;
            err += e.norm();
        }

        if (valid_num > 0.2*ws)
        {
            if (err < err_min)
            {
                err_min = err;
                best_x1 = x(2);
                best_x2 = x(3);
                best_x3 = x(4);
            }
            f_pcb3 << err <<" ";
            f_pcb3 << valid_num<<" ";
            f_pcb3 << x(0)<<" ";
            f_pcb3 << x(1)<<" ";
            f_pcb3 << x(2)<<" ";
            f_pcb3 << x(3)<<" ";
            f_pcb3 << x(4)<<" ";
            f_pcb3 <<"gc = ["<<gc_refined(0);
            f_pcb3 <<", "<<gc_refined(1);
            f_pcb3 <<", "<<gc_refined(2)<<"]"<<endl;
        }

    }
    f_pcb3 <<"min error: "<<err_min<<", best tcb: "<<best_x1<<", "
            <<best_x2<<", "<<best_x3<<endl;
    f_pcb3.close();
}

//求解bias，优化重力，平移,非滑窗
void cSolver::RefineResult_test3(std::vector<std::vector<data_selection::sync_data>> &calib_data,
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
void cSolver::RefineResult(std::vector<std::vector<data_selection::sync_data>> &calib_data,
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
void cSolver::solveOtherResult(std::vector<std::vector<data_selection::sync_data> > &calib_data,
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
//        tcb(0) = -0.081;
//        tcb(1) = 1.03;
//        tcb(2) = -1.45;
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

//求解Rcb
//yaw的判断应该使用收敛条件，这里使用的误差最小来判断，不太合适
//判断yaw和gc的两个分量的标准差
void cSolver::solveOtherResult2(std::vector<std::vector<data_selection::sync_data> > &calib_data,
                                Eigen::Matrix3d Ryx_cam,
                                Eigen::Matrix3d Ryx_imu,Eigen::Matrix3d &Rci)
{
    static ofstream f_gamma3;
    f_gamma3.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/Rcb_estimate2.txt");
    f_gamma3<<std::fixed<<std::setprecision(6);
    f_gamma3 <<"SolveOtherResult2\n";

    bool first_solve = true;
    Eigen::Vector3d gc_last = Eigen::Vector3d::Zero();
    Eigen::Vector2d gamma_last = Eigen::Vector2d::Zero();
    int segmentCount = calib_data.size();
    int motionCount = 0;
    for (int i = 0; i < segmentCount; ++i)
    {
        motionCount += calib_data[i].size();
    }
    Eigen::Matrix3d Ryx_imu_inv = Ryx_imu.transpose();
    Eigen::Matrix3d Ryx_cam_inv = Ryx_cam.transpose();

    //x = [gc,cos,sin]
    for (int segmentId =0 ; segmentId < segmentCount; ++segmentId)
    {
        Eigen::Vector3d tcb;
//        tcb.setZero();
        tcb(0) = 0.02241856;
        tcb(1) = -0.01121906;
        tcb(2) = -0.01653902;
        Eigen::Vector3d gc;
//        gc(0) = 0.11943;
//        gc(1) = -9.7944;
//        gc(2) = -0.54;
        gc(0) = 0.033;
        gc(1) = 9.7975;
        gc(2) = 0.4928;
//        cout<<"segmentCount: "<<segmentCount<<endl;
        int num = calib_data[segmentId].size()-1;
//        cout<<"num: "<<num<<endl;
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
            Eigen::Matrix3d R21_cam = calib_data[segmentId][motionId].q21_cam.toRotationMatrix();
            Eigen::Matrix3d R32_cam = calib_data[segmentId][motionId+1].q21_cam.toRotationMatrix();


            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            Eigen::Vector3d lambda = (pc2-pc1)*dT23 + (pc2-pc3)*dT12;//s
            Eigen::Matrix3d beta = 0.5*I*(dT12*dT12*dT23 + dT12*dT23*dT23);//gc
//            Eigen::Vector3d Gc_test = beta*gc;
            Eigen::Vector3d phi = (Rwc2_cam - Rwc3_cam)*tcb*dT12 -
                                  (Rwc1_cam - Rwc2_cam)*tcb*dT23;//pcb

            //gamma1
            Eigen::Vector3d pi_12 = Ryx_imu * t12_imu;
            Eigen::Matrix3d tmp_R12;
            tmp_R12<< pi_12(0), -pi_12(1),0,
                    pi_12(1),pi_12(0),0,
                    0,0,pi_12(2);
            Eigen::Matrix3d gamma_1 = -Rwc1_cam * Ryx_cam_inv * tmp_R12 * dT23;
            //gamma2
            Eigen::Vector3d pi_23 = Ryx_imu * t23_imu;
            Eigen::Matrix3d tmp_R23;
            tmp_R23<< pi_23(0), -pi_23(1),0,
                    pi_23(1),pi_23(0),0,
                    0,0,pi_23(2);
            Eigen::Matrix3d gamma_2 = Rwc2_cam * Ryx_cam_inv * tmp_R23 * dT12;
            //gamma3
            Eigen::Vector3d vi_12 = Ryx_imu * v12_imu;
            Eigen::Matrix3d tmp_v12;
            tmp_v12<< vi_12(0), -vi_12(1),0,
                    vi_12(1),vi_12(0),0,
                    0,0,vi_12(2);
            Eigen::Matrix3d gamma_3 = Rwc1_cam * Ryx_cam_inv * tmp_v12 * dT12 * dT23;
            Eigen::Matrix3d gamma = gamma_1 + gamma_2 + gamma_3;

            //b

//            计算权重
            if (!first_solve)
            {
                Eigen::Vector3d pred = beta*gc_last +
                        gamma.block<3,2>(0,0) * gamma_last;
//                Eigen::Vector3d pred = gamma.block<3,2>(0,0) * gamma_last;
                Eigen::Vector3d e = pred + lambda + phi + gamma.block<3,1>(0,2);
                double wi = exp(-e.norm()*200.0);
//                cout<<"e norm: "<<e.norm()<<endl;
                if (e.norm() > 0.01)
                    wi = 0.0;
                lambda *=wi;
                beta *=wi;
                phi *= wi;
                gamma *=wi;
//                test *= wi;
                f_gamma3 <<"e.norm="<<e.norm()<<" ";
                f_gamma3 <<"wi="<<wi<<" ";
            }

            Eigen::Vector3d test = -lambda-phi-gamma.block<3,1>(0,2);
            tmp_G.block<3,3>(motionId*3,0) = beta;
            tmp_G.block<3,2>(motionId*3,3) = gamma.block<3,2>(0,0);
            tmp_b.block<3,1>(motionId*3,0) = test;

            if (motionId > 15)
            {
                //求解
                Eigen::VectorXd x;
                x = tmp_G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp_b);
                double alpha = atan2(x(4), x(3));
                f_gamma3 << motionId <<" ";
                f_gamma3 << x(0)<<" ";
                f_gamma3 << x(1)<<" ";
                f_gamma3 << x(2)<<" ";
//                f_gamma3 << x(1)<<" ";
//                f_gamma3 << x(1)<<" ";
                f_gamma3 << alpha*RAD2DEG<<" ";
                Eigen::Matrix3d Rz;
                Rz << cos(alpha), -sin(alpha), 0,
                        sin(alpha), cos(alpha),0,
                        0,0,1;
                Eigen::Matrix3d Rcb = Ryx_cam_inv * Rz * Ryx_imu;
                Eigen::Vector3d euler = Rcb.inverse().eulerAngles(2,1,0);
                f_gamma3 << euler(0)*RAD2DEG<<" ";
                f_gamma3 << euler(1)*RAD2DEG<<" ";
                f_gamma3 << euler(2)*RAD2DEG<<endl;
                if (first_solve)
                    first_solve = false;

                gc_last = x.head(3);
                gamma_last = x.tail(2);
            }
        }
    }
    f_gamma3.close();

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

void cSolver::RefineRotation(std::vector<data_selection::sync_data> &calib_data,
                             Eigen::Matrix3d &_Rcb)
{
    static ofstream f_imu;
    f_imu.open("/home/wangzb/Documents/01_Learning/02_Paper/03_cam-IMU/CamOdomCalibraTool/imu_axis.txt");
    f_imu<<std::fixed<<std::setprecision(6);
    f_imu <<"x y z\n";
//    Eigen::Matrix3d Rci;
//    Rci << -0.095409, -0.995027, -0.028602,
//            -0.125496,  0.040527, -0.991266,
//            0.987496, -0.090986, -0.128738;

    int N=0;
    for (int i = 0; i < calib_data.size(); ++i)
    {
        double axis_x, axis_y, axis_z;
        axis_x = calib_data[i].axis_imu(0);
        axis_y = calib_data[i].axis_imu(1);
        axis_z = calib_data[i].axis_imu(2);
//        if (calib_data[i].axis_imu(2)<-0.96)
        if(fabs(axis_x)<0.22 && fabs(axis_y)<0.22 && fabs(axis_z)<0.22)//
            continue;
        N++;
    }
//    int frame_count = 100;
//    int N = (int)floor(calib_data.size() / frame_count);
    int num = 0;
    Eigen::MatrixXd A(N*4,4);
    Eigen::MatrixXd b(N*4,1);
    for (int i = 0; i < N; ++i)
    {
        double axis_x, axis_y, axis_z;
        axis_x = calib_data[i].axis_imu(0);
        axis_y = calib_data[i].axis_imu(1);
        axis_z = calib_data[i].axis_imu(2);
        f_imu << axis_x<<" "<<axis_y<<" "<<axis_z<<endl;
//        if (calib_data[i].axis_imu(2)<-0.96)
//            continue;
        if(fabs(axis_x)<0.22 && fabs(axis_y)<0.22 && fabs(axis_z)<0.22)//
            continue;
        num++;
        Eigen::Quaterniond qi = calib_data[i].q21_imu.conjugate();
        Eigen::Quaterniond qc = calib_data[i].q12_cam;

        Quaterniond r2(_Rcb.inverse()*qi*_Rcb);
        double angular_distance = RAD2DEG * qc.angularDistance(r2);
//        cout<<"distance: "<<angular_distance<<endl;
//        double huber = angular_distance > 1.5 ? 1.5/angular_distance : 1.0;
        double huber = 1.0;

        //该形式为JPL形式，对应求解的x=[x,y,z,w]
        Matrix4d L, R;
        double w = qc.w();
        Vector3d q = qc.vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;
        w = qi.w();
        q = qi.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        //该形式为Hamilton形式，对应求解的x=[w,x,y,z]
        Eigen::Matrix4d RQ;
        RQ << qi.w(), -qi.x(), -qi.y(), -qi.z(),
                qi.x(), qi.w(), qi.z(), -qi.y(),
                qi.y(), -qi.z(), qi.w(), qi.x(),
                qi.z(), qi.y(), -qi.x(), qi.w();
        Eigen::Matrix4d LQ;
        LQ << qc.w(), -qc.x(), -qc.y(), -qc.z(),
                qc.x(), qc.w(), -qc.z(), qc.y(),
                qc.y(), qc.z(), qc.w(), -qc.x(),
                qc.z(), -qc.y(), qc.x(), qc.w();
        Eigen::Matrix4d RL = LQ-RQ;
        A.block<4,4>(i*4,0) = huber*(L-R);
//        A.block<4,4>(i*4,0) = huber*RL;
    }
    Eigen::JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Eigen::Vector4d x = svd.matrixV().col(3);
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    cout<<"ric_cov: "<<ric_cov.transpose()<<endl;
    Eigen::Quaterniond ref_qcb(x);
//    Eigen::Quaterniond ref_qcb;
//    ref_qcb.w() = x(0);
//    ref_qcb.x() = x(1);
//    ref_qcb.y() = x(2);
//    ref_qcb.z() = x(3);

        Eigen::Matrix3d ref_Rbc = ref_qcb.toRotationMatrix().inverse();
        Eigen::Vector3d euler_bc = ref_Rbc.eulerAngles(2,1,0);
        Eigen::Matrix3d R_truth;
        R_truth << 0.003160590246273326, -0.9999791454805219, -0.005631986624785923,
            -0.002336363271321251, 0.0056246151765369234, -0.9999814523833838,
            0.9999922760081504, 0.0031736899915518757, -0.0023185374437218464;
        Eigen::Matrix3d dR = R_truth - ref_qcb.toRotationMatrix();
        cout<<"euler: "<<euler_bc.transpose()*RAD2DEG<<endl;
        cout<<"dR: "<<dR.norm()<<endl;
        cout<<"num: "<<num<<endl;


    f_imu.close();
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