#include "g2otypes.h"

namespace g2o
{
/**
 * @brief VertexGyrBias::VertexGyrBias
 */
VertexGyrBias::VertexGyrBias() : BaseVertex<3, Vector3d>()
{
}

bool VertexGyrBias::read(std::istream& is)
{
    Vector3d est;
    for (int i=0; i<3; i++)
        is  >> est[i];
    setEstimate(est);
    return true;
}

bool VertexGyrBias::write(std::ostream& os) const
{
    Vector3d est(estimate());
    for (int i=0; i<3; i++)
        os << est[i] << " ";
    return os.good();
}

void VertexGyrBias::oplusImpl(const double* update_)  {
    Eigen::Map<const Vector3d> update(update_);
    _estimate += update;
    // Debug log
    //std::cout<<"updated bias estimate: "<<_estimate.transpose()<<", gyr bias update: "<<update.transpose()<<std::endl;
}

/**
 * @brief EdgeGyrBias::EdgeGyrBias
 */
EdgeGyrBias::EdgeGyrBias() : BaseUnaryEdge<3, Vector3d, VertexGyrBias>()
{
}


bool EdgeGyrBias::read(std::istream& is)
{
    return true;
}

bool EdgeGyrBias::write(std::ostream& os) const
{
    return true;
}

void EdgeGyrBias::computeError()
{
    const VertexGyrBias* v = static_cast<const VertexGyrBias*>(_vertices[0]);
    Vector3d bg = v->estimate();
    Matrix3d dRbg = Sophus::SO3d::exp(J_dR_bg * bg).matrix();
    Sophus::SO3d errR ( ( dRbij * dRbg ).transpose() * Rwbi.transpose() * Rwbj ); // dRij^T * Riw * Rwj
    _error = errR.log();
    // Debug log
    //std::cout<<"dRbg: "<<std::endl<<dRbg<<std::endl;
    //std::cout<<"error: "<<_error.transpose()<<std::endl;
    //std::cout<<"chi2: "<<_error.dot(information()*_error)<<std::endl;
}


void EdgeGyrBias::linearizeOplus()
{
    Sophus::SO3d errR ( dRbij.transpose() * Rwbi.transpose() * Rwbj ); // dRij^T * Riw * Rwj
//    Matrix3d Jlinv = Sophus::SO3d::Jaco(errR.log());

//    _jacobianOplusXi = - Jlinv * J_dR_bg;

    // Debug log
    //std::cout<<"jacobian to bg:"<<std::endl<<_jacobianOplusXi<<std::endl;
    //std::cout<<"Jlinv: "<<Jlinv<<std::endl<<"J_dR_bg: "<<J_dR_bg<<std::endl;
}

//--------外参旋转顶点--------
VertexExtrinsicRotation::VertexExtrinsicRotation()
        :BaseVertex<4, Vector4d>()
{
}
bool VertexExtrinsicRotation::read(std::istream& is)
    {
        return true;
    }
bool VertexExtrinsicRotation::write(std::ostream& os) const
{
    return true;
}

void VertexExtrinsicRotation::oplusImpl(const double* update_)
{
    Eigen::Map<const Vector4d> update(update_);
    Eigen::Quaterniond dq;
    dq.x() = update(0);
    dq.y() = update(1);
    dq.z() = update(2);
    dq.w() = update(3);
    Eigen::Quaterniond q;
    q.x() = _estimate(0);
    q.y() = _estimate(1);
    q.z() = _estimate(2);
    q.w() = _estimate(3);
    Eigen::Matrix3d dR = dq.toRotationMatrix();
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Quaterniond qR = Eigen::Quaterniond(R*dR);
    _estimate(0) = qR.x();
    _estimate(1) = qR.y();
    _estimate(2) = qR.z();
    _estimate(3) = qR.w();
    // Debug log
    //std::cout<<"updated bias estimate: "<<_estimate.transpose()<<", gyr bias update: "<<update.transpose()<<std::endl;
}

}
