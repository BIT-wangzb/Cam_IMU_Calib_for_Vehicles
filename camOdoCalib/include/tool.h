//
// Created by wangzb on 2022/5/17.
//

#ifndef CAM_ODO_CAL_TOOL_H
#define CAM_ODO_CAL_TOOL_H

template<typename T>
inline bool computeError(const T* q12_i, const T* q12_c,
                         const T* qcb, T* residual)
{
    //q: [w,x,y,z]
    T qw = q12_i[0];
    T qx = q12_i[1];
    T qy = q12_i[2];
    T qz = q12_i[3];
    T RightQ[16];
    RightQ[0] = qw; RightQ[1] = -qx; RightQ[2] = -qy; RightQ[3] = qz;
    RightQ[4] = qx; RightQ[5] = qw; RightQ[6] = qz; RightQ[7] = -qy;
    RightQ[8] = qy; RightQ[9] = -qz; RightQ[10] = qw; RightQ[11] = qx;
    RightQ[12] = qz; RightQ[13] = qy; RightQ[14] = -qx; RightQ[15] = qw;
    T LeftQ[16];
    qw = q12_c[0];
    qx = q12_c[1];
    qy = q12_c[2];
    qz = q12_c[3];
    LeftQ[0] = qw; LeftQ[1] = -qx; LeftQ[2] = -qy; LeftQ[3] = -qz;
    LeftQ[4] = qx; LeftQ[5] = qw; LeftQ[6] = -qz; LeftQ[7] = qy;
    LeftQ[8] = qy; LeftQ[9] = qz; LeftQ[10] = qw; LeftQ[11] = -qx;
    LeftQ[12] = qz; LeftQ[13] = -qy; LeftQ[14] = qx; LeftQ[15] = qw;

    T LQ_RQ[16];
    LQ_RQ[0] = LeftQ[0]-RightQ[0]; LQ_RQ[1] = LeftQ[1]-RightQ[1];
    LQ_RQ[2] = LeftQ[2]-RightQ[2]; LQ_RQ[3] = LeftQ[3]-RightQ[3];
    LQ_RQ[4] = LeftQ[4]-RightQ[4]; LQ_RQ[5] = LeftQ[5]-RightQ[5];
    LQ_RQ[6] = LeftQ[6]-RightQ[6]; LQ_RQ[7] = LeftQ[7]-RightQ[7];
    LQ_RQ[8] = LeftQ[8]-RightQ[8]; LQ_RQ[9] = LeftQ[9]-RightQ[9];
    LQ_RQ[10] = LeftQ[10]-RightQ[10]; LQ_RQ[11] = LeftQ[11]-RightQ[11];
    LQ_RQ[12] = LeftQ[12]-RightQ[12]; LQ_RQ[13] = LeftQ[13]-RightQ[13];
    LQ_RQ[14] = LeftQ[14]-RightQ[14]; LQ_RQ[15] = LeftQ[15]-RightQ[15];

    residual[0] = LQ_RQ[0]*qcb[0] + LQ_RQ[1]*qcb[1] + LQ_RQ[2]*qcb[2] + LQ_RQ[3]*qcb[3];
    residual[1] = LQ_RQ[4]*qcb[0] + LQ_RQ[5]*qcb[1] + LQ_RQ[6]*qcb[2] + LQ_RQ[7]*qcb[3];
    residual[2] = LQ_RQ[8]*qcb[0] + LQ_RQ[9]*qcb[1] + LQ_RQ[10]*qcb[2] + LQ_RQ[11]*qcb[3];
    residual[3] = LQ_RQ[12]*qcb[0] + LQ_RQ[13]*qcb[1] + LQ_RQ[14]*qcb[2] + LQ_RQ[15]*qcb[3];
//    std::cout<<"error: "<<res[0]<<", "<<res[1]<<", "<<res[2]<<", "<<res[3]<<std::endl;
//    std::cout<<"error: "<<residual[0]<<std::endl;

    return true;
}

#endif //CAM_ODO_CAL_TOOL_H
