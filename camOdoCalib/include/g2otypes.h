#ifndef G2OTYPES_H
#define G2OTYPES_H

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <sophus/so3.hpp>

//#include "so3.h"
//#include "NavState.h"
//#include "IMUPreintegrator.h"
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
//#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
using namespace Eigen;
namespace g2o
{

/**
 * @brief The VertexGyrBias class
 * For gyroscope bias compuation in Visual-Inertial initialization
 */
class VertexGyrBias : public BaseVertex<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexGyrBias();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update_);
};

/**
 * @brief The EdgeGyrBias class
 * For gyroscope bias compuation in Visual-Inertial initialization
 */
class EdgeGyrBias : public BaseUnaryEdge<3, Vector3d, VertexGyrBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeGyrBias();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    Matrix3d dRbij;
    Matrix3d J_dR_bg;
    Matrix3d Rwbi;
    Matrix3d Rwbj;

    void computeError();

    virtual void linearizeOplus();
};

class VertexExtrinsicRotation : public BaseVertex<4,Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexExtrinsicRotation();
    bool read(std::istream& is);
    bool write(std::ostream& os) const;
    virtual void setToOriginImpl() {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update_);

};

}

#endif // G2OTYPES_H
