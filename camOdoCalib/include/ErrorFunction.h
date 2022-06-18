//
// Created by wangzb on 2022/5/17.
//

#ifndef CAM_ODO_CAL_ERRORFUNCTION_H
#define CAM_ODO_CAL_ERRORFUNCTION_H

#include <iostream>
#include <ceres/ceres.h>
#include "tool.h"

using namespace std;
class ErrorFunction
{
private:
    double obs_q12_i_w, obs_q12_i_x, obs_q12_i_y, obs_q12_i_z;
    double obs_q12_c_w, obs_q12_c_x, obs_q12_c_y, obs_q12_c_z;
public:
    ErrorFunction(double q12_i_w, double q12_i_x, double q12_i_y, double q12_i_z,
                  double q12_c_w, double q12_c_x, double q12_c_y, double q12_c_z)
            : obs_q12_i_w(q12_i_w), obs_q12_i_x(q12_i_x),
              obs_q12_i_y(q12_i_y), obs_q12_i_z(q12_i_z),
              obs_q12_c_w(q12_c_w), obs_q12_c_x(q12_c_x),
              obs_q12_c_y(q12_c_y), obs_q12_c_z(q12_c_z){}

    template<typename T>
    bool operator()(const T* const qcb,
                    T* residuals)const{
        // qcb[w,x,y,z]
        T res[4];
        T q12_i[4];
        T q12_c[4];
        q12_i[0] = T(obs_q12_i_w);q12_i[1] = T(obs_q12_i_x);
        q12_i[2] = T(obs_q12_i_y);q12_i[3] = T(obs_q12_i_z);
        q12_c[0] = T(obs_q12_c_w);q12_c[1] = T(obs_q12_c_x);
        q12_c[2] = T(obs_q12_c_y);q12_c[3] = T(obs_q12_c_z);
        computeError(q12_i,q12_c,qcb,res);
        residuals[0] = res[0];
        residuals[1] = res[1];
        residuals[2] = res[2];
        residuals[3] = res[3];

        return true;
    }

    static ceres::CostFunction* Create(const double q12_i_w, double q12_i_x,
                                       double q12_i_y, double q12_i_z,
                                        double q12_c_w, double q12_c_x,
                                        double q12_c_y, double q12_c_z)
    {
        return (new ceres::AutoDiffCostFunction<ErrorFunction,4,4>(
                new ErrorFunction(q12_i_w,q12_i_x,q12_i_y,q12_i_z,
                                  q12_c_w,q12_c_x,q12_c_y,q12_c_z)));
    }


};

#endif //CAM_ODO_CAL_ERRORFUNCTION_H
