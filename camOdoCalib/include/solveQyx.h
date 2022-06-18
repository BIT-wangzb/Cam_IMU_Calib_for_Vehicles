#ifndef SOLVEQXY_H
#define SOLVEQXY_H

#include "solver.h"
#include <ceres/ceres.h>

template<typename T>
Eigen::Matrix< T,4,4 > QuaternionMultMatLeft(const Eigen::Quaternion< T >& q )
{
    return (Eigen::Matrix< T, 4,4>() << q.w(), -q.z(), q.y(), q.x(),
                                                                          q.z(), q.w(), -q.x(), q.y(),
                                                                          -q.y(), q.x(), q.w(), q.z(),
                                                                          -q.x(), -q.y(), -q.z(), q.w()).finished();
}

template< typename T>
Eigen::Matrix< T ,4,4 > QuaternionMultMatRight(const Eigen::Quaternion< T >& q )
{
  return (Eigen::Matrix<T ,4,4>() <<q.w(), q.z(), -q.y(), q.x(),
                                                                      -q.z(), q.w(), q.x(), q.y(),
                                                                      q.y(), -q.x(), q.w(), q.z(),
                                                                      -q.x(), -q.y(), -q.z(), q.w()).finished();

}

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

    bool estimateRyx(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx);
    bool estimateRyx_imu(std::vector<data_selection::sync_data> sync_result, Eigen::Matrix3d &Ryx);

    void correctCamera(std::vector<data_selection::sync_data> &sync_result,
      std::vector<data_selection::cam_data> &camDatas, Eigen::Matrix3d Ryx);

    void refineExPara(std::vector<data_selection::sync_data> sync_result,
        cSolver::calib_result &internelPara,Eigen::Matrix3d Ryx);
    //bool esExParaByCeres(const std::vector<data_selection::sync_data> &calib_data,cSolver::calib_result &res);

    void q2Euler_zyx(Eigen::Quaterniond q, Eigen::Vector3d &res);

    //for bit test
    bool estimateRyx_cam_test(std::vector<data_selection::sync_data_test> sync_result,
                     Eigen::Matrix3d &Ryx);
    bool estimateRyx_odo_test(std::vector<data_selection::sync_data_test> sync_result,
                          Eigen::Matrix3d &Ryx);
    void toEulerAngle(Eigen::Quaterniond q, Eigen::Vector3d &euler);

private:
    bool estimateRyx2(const std::vector<Eigen::Quaterniond > &quats_odo,
      const std::vector<Eigen::Vector3d > &tvecs_odo,
      const std::vector<Eigen::Quaterniond> &quats_cam,
      const std::vector<Eigen::Vector3d > &tvecs_cam,
      Eigen::Matrix3d &R_yx);

    void refineEstimate(Eigen::Matrix4d &Trc, double scale,
        const std::vector<Eigen::Quaterniond > &quats_odo,
        const std::vector<Eigen::Vector3d> &tvecs_odo,
        const std::vector<Eigen::Quaterniond> &quats_cam,
        const std::vector<Eigen::Vector3d> &tvecs_cam);

    bool SolveConstraintqyx(const Eigen::Vector4d t1, const Eigen::Vector4d t2, double& x1, double& x2);
    bool SolveConstraintqyx_test(const Eigen::Vector4d t1,
                                 const Eigen::Vector4d t2,
                                 double& x1, double& x2);

};
class CameraOdomErr
  {
  private:
    Eigen::Quaterniond m_q1, m_q2;
    Eigen::Vector3d m_t1,m_t2;

  public:
  //odom cam
    CameraOdomErr(Eigen::Quaterniond q1, Eigen::Vector3d t1,    
      Eigen::Quaterniond q2, Eigen::Vector3d t2)   
    : m_q1(q1),m_q2(q2), m_t1(t1),m_t2(t2)
    { }

  template<typename T>
    bool operator() (const T* const q4x1 , const T* const t3x1, T* residuals) const
    { 
      Eigen::Quaternion<T> qrc( q4x1[0], q4x1[1],q4x1[2],q4x1[3]);
      Eigen::Matrix<T,3,1> trc;
      trc<<t3x1[0],t3x1[1], T(0);

      Eigen::Quaternion<T> q_odo = m_q1.cast<T>();
      Eigen::Matrix<T,3,1>  t_odo =  m_t1.cast<T>();
      Eigen::Quaternion<T> q_cc = m_q2.cast<T>();
      Eigen::Matrix<T, 3,1> t_cc = m_t2.cast<T>();

      Eigen::Matrix<T,3,3> R_odo = q_odo.toRotationMatrix();
      Eigen::Matrix<T,3,3> Rrc = qrc.toRotationMatrix();

      Eigen::Matrix<T, 3,1> t_err = (R_odo - Eigen::Matrix<T,3,3>::Identity() ) * trc - (Rrc * t_cc) + t_odo;

    //  q is unit quaternion,   q.inv() = q.conjugate();
    //  q_odo * q_oc = q_oc * q_cc  -->   q_oc.conjugate() * q_odo * q_oc * q_cc.conjugate() = 0;
      Eigen::Quaternion<T> q_err = qrc.conjugate() * q_odo * qrc * q_cc.conjugate();
      Eigen::Matrix<T,3,3> R_err = q_err.toRotationMatrix();

      T roll, pitch, yaw;
      mat2RPY(R_err,roll, pitch,yaw);

      residuals[0] = t_err[0];
      residuals[1] = t_err[1];
      residuals[2] = t_err[2];
      residuals[3] = roll;
      residuals[4] = pitch;
      residuals[5] = yaw;
      
      return true;
    }
  };

#endif