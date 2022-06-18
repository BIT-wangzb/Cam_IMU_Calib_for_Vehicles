#include "calcCamPose.h"

using namespace cv;
using namespace std;
//子函数，被calcCamPose调用
//使用AprilTag可以在一幅图像上获取特征点的3d物理坐标，便于使用Pnp求取VO
//寻找AprilTag2维像素坐标点并与实际3维物理坐标对应起来
void FindTargetCorner(cv::Mat &img_raw, const PatternType &pt,
                      std::vector<cv::Point3f> &p3ds,
                      std::vector<cv::Point2f> &p2ds)
{
  const int col = 9;
  const int row = 6;
  if (CHESS == pt)
  {
    // std::cout << "CHESSBOARD\n";
    const float square_size = 0.575; // unit:  m
    cv::Size pattern_size(col, row);
    std::vector<cv::Point2f> corners;
    if (cv::findChessboardCorners(img_raw, pattern_size, corners))
    {
      cv::cornerSubPix(
          img_raw, corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      if (corners.size() == col * row)
      {
        int count = 0;
        for (int i = 0; i < row; i++)
        {
          for (int j = 0; j < col; j++)
          {
            // Todo: change 3d-coordinate
            p3ds.emplace_back(
                cv::Point3f(j * square_size, i * square_size, 0.0));
            p2ds.emplace_back(cv::Point2f(corners[count].x, corners[count].y));
            count++;
          }
        }
      }
      else
      {
        std::cout << "Chessboard config is not correct with image\n";
      }
    }
    else
    {
      std::cout << "No chessboard detected in image\n";
    }
  }
  else if (APRIL == pt)
  {
    //const int april_rows = 6;
    const int april_cols = 6;
    const double tag_sz = 0.055;
    const double tag_spacing_sz = 0.0715; // 0.055 + 0.0165

    //1.设定AprilTag的种类
    AprilTags::TagCodes tagCodes(AprilTags::tagCodes36h11);//tagCodes36h11：0～586
    AprilTags::TagDetector detector(tagCodes);
    //2.识别图像中的标志
    std::vector<AprilTags::TagDetection> detections =
        detector.extractTags(img_raw);
    //if (detections.size() == april_rows * april_cols)
    if (detections.size() > 20)
    {
      //3. 对Tag目标进行重新排序
      std::sort(detections.begin(), detections.end(),
                AprilTags::TagDetection::sortByIdCompare);//用自定义函数对象排序（由小到大）
      
      //4.获得Tag目标的四个角点（有顺序）
      cv::Mat tag_corners(4 * detections.size(), 2, CV_32F);
      for (unsigned i = 0; i < detections.size(); i++)
      {
        for (unsigned j = 0; j < 4; j++)
        {
          tag_corners.at<float>(4 * i + j, 0) = detections[i].p[j].first;
          tag_corners.at<float>(4 * i + j, 1) = detections[i].p[j].second;
        }
      }
      //5.提取四个角点的亚像素角点坐标  tag_corners是一个行为４×Tag数量(一个Tag对应四个角点)，列为２的矩阵
      cv::cornerSubPix(
          img_raw, tag_corners, cv::Size(2, 2), cv::Size(-1, -1),
          cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      cv::cvtColor(img_raw, img_raw, CV_GRAY2BGR);

      // draw axis
      /*double center_u = tag_corners.at<float>(0, 0);
      double center_v = tag_corners.at<float>(0, 1);
      cv::Point2f center(center_u, center_v);
      double xaxis_u = tag_corners.at<float>(20, 0);
      double xaxis_v = tag_corners.at<float>(20, 1);
      cv::Point2f xaxis(xaxis_u + (xaxis_u - center_u) * 0.5,
                        xaxis_v + (xaxis_v - center_v) * 0.5);
      double yaxis_u = tag_corners.at<float>(120, 0);
      double yaxis_v = tag_corners.at<float>(120, 1);
      cv::Point2f yaxis(yaxis_u + (yaxis_u - center_u) * 0.5,
                        yaxis_v + (yaxis_v - center_v) * 0.5);
      cv::Point2f zaxis(center_u - (yaxis_u - center_u) * 0.5,
                        center_v - (xaxis_v - center_v) * 0.5);
      cv::line(img_raw, center, xaxis, cv::Scalar(0, 0, 255), 2);
      cv::putText(img_raw, "X", xaxis, CV_FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 255));
      cv::line(img_raw, center, yaxis, cv::Scalar(0, 255, 0), 2);
      cv::putText(img_raw, "Y", yaxis, CV_FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 255, 0));
      cv::line(img_raw, center, zaxis, cv::Scalar(255, 0, 0), 2);
      cv::putText(img_raw, "Z", zaxis, CV_FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 0, 0));*/

      //将Tag矩阵中四个角点的物理坐标与像素坐标对应起来
      int index = 0;
      for (auto &tag : detections)
      {
        unsigned int id = tag.id;
        unsigned int tag_row = id / april_cols;
        unsigned int tag_col = id % april_cols;

        //在图像中标记出Ｔａｇ
        cv::circle(img_raw, cv::Point2f(tag.cxy.first, tag.cxy.second), 3,
                   cv::Scalar(0, 255, 0));
        cv::putText(img_raw, std::to_string(id),
                    cv::Point2f(tag.cxy.first, tag.cxy.second),
                    CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

        //第一个角点
        p3ds.emplace_back(cv::Point3f(tag_spacing_sz * tag_col,
                                      tag_spacing_sz * tag_row, 0.0));
        p2ds.emplace_back(cv::Point2f(tag_corners.at<float>(index, 0),
                                      tag_corners.at<float>(index, 1)));
        // cv::circle(img_raw, p2ds.back(), 3, cv::Scalar(0, 255, 0));
        // std::cout << index++ << " " << p3ds.back() << std::endl;
        // cv::imshow("apriltag_detection", img_raw);
        // cv::waitKey(0);
        //第２个角点
        ++index;
        p3ds.emplace_back(cv::Point3f(tag_spacing_sz * tag_col + tag_sz,
                                      tag_spacing_sz * tag_row, 0.0));//emplace_back比push_back效率高
        p2ds.emplace_back(cv::Point2f(tag_corners.at<float>(index, 0),
                                      tag_corners.at<float>(index, 1)));
        // cv::circle(img_raw, p2ds.back(), 3, cv::Scalar(0, 255, 0));
        // std::cout << index++ << " " << p3ds.back() << std::endl;
        // cv::imshow("apriltag_detection", img_raw);
        // cv::waitKey(0);
        //第3个角点
        ++index;
        p3ds.emplace_back(cv::Point3f(tag_spacing_sz * tag_col + tag_sz,
                                      tag_spacing_sz * tag_row + tag_sz, 0.0));
        p2ds.emplace_back(cv::Point2f(tag_corners.at<float>(index, 0),
                                      tag_corners.at<float>(index, 1)));
        // cv::circle(img_raw, p2ds.back(), 3, cv::Scalar(0, 255, 0));
        // std::cout << index++ << " " << p3ds.back() << std::endl;
        // cv::imshow("apriltag_detection", img_raw);
        // cv::waitKey(0);
        //第4个角点
        ++index;
        p3ds.emplace_back(cv::Point3f(tag_spacing_sz * tag_col,
                                      tag_spacing_sz * tag_row + tag_sz, 0.0));
        p2ds.emplace_back(cv::Point2f(tag_corners.at<float>(index, 0),
                                      tag_corners.at<float>(index, 1)));
        // cv::circle(img_raw, p2ds.back(), 3, cv::Scalar(0, 255, 0));
        // std::cout << index++ << " " << p3ds.back() << std::endl;
        // cv::imshow("apriltag_detection", img_raw);
        // cv::waitKey(0);
        ++index;
      }
    }
  }
  else
  {
    std::cout << "Pattern type not supported yet.\n";
  }
}

void FindStereoTargetCorner(const cv::Mat &img0_raw, const cv::Mat &img1_raw,
                            const PatternType& pt,
                            std::vector<cv::Point2f> &img0_p2ds,
                            std::vector<cv::Point2f> &img1_p2ds)
{
    if(APRIL ==pt)
    {
        const int april_cols = 6;
        const double tag_sz = 0.036;
        const double tag_spacing_sz = 0.3;

        //1,set apriltags codes
        AprilTags::TagCodes tagCodes(AprilTags::tagCodes36h11);
        AprilTags::TagDetector detector0(tagCodes);
        AprilTags::TagDetector detector1(tagCodes);

        //2, detect apriltag corners
        std::vector<AprilTags::TagDetection> detections0 =
                detector0.extractTags(img0_raw);
        std::vector<AprilTags::TagDetection> detections1 =
                detector1.extractTags(img1_raw);

        //
        if (detections0.size() > 10 && detections1.size() > 10)
        {
            //3, sort the corners by id
            std::sort(detections0.begin(), detections0.end(),
                      AprilTags::TagDetection::sortByIdCompare);
            std::sort(detections1.begin(), detections1.end(),
                      AprilTags::TagDetection::sortByIdCompare);
            std::vector<cv::Point2f> obj_p2ds;//apriltag's physical coordinates
            for (int i = 0; i < detections0.size(); ++i)
            {
                int match_flag = false;
                for (int j = 0; j < detections1.size(); ++j)
                {
                    if (detections0[i].id == detections1[j].id)
                    {
                        for (int k = 0; k < 4; ++k)
                        {
                            img0_p2ds.emplace_back( cv::Point2f(detections0[i].p[k].first, detections0[i].p[k].second) );
                            img1_p2ds.emplace_back( cv::Point2f(detections0[j].p[k].first, detections0[j].p[k].second) );
                        }
                        //
                        unsigned int id = detections0[i].id;

                        cv::circle(img_raw0, cv::Point2f(detections0[i].cxy.first, detections0[i].cxy.second), 3,
                                   cv::Scalar(0,255,255));
                        cv::putText(img_raw0, std::to_string(id),
                                cv::Point2f(detections0[i].cxy.first, detections0[i].cxy.second),
                                CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
                        cv::circle(img_raw1, cv::Point2f(detections1[i].cxy.first, detections1[i].cxy.second), 3,
                                   cv::Scalar(0,255,255));
                        cv::putText(img_raw1, std::to_string(id),
                                    cv::Point2f(detections1[i].cxy.first, detections1[i].cxy.second),
                                    CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));

                        unsigned int tag_row = id / april_cols;
                        unsigned int tag_col = id % april_cols;

                        //find the AprilTags four corners
                        obj_p2ds.emplace_back( cv::Point2f(tag_spacing_sz * tag_col,
                                                           tag_spacing_sz * tag_row) );//the first
                        obj_p2ds.emplace_back( cv::Point2f(tag_spacing_sz * tag_col + tag_sz,
                                                           tag_spacing_sz * tag_row) );//the second
                        obj_p2ds.emplace_back( cv::Point2f(tag_spacing_sz * tag_col + tag_sz,
                                                           tag_spacing_sz * tag_row + tag_sz) );//the third
                        obj_p2ds.emplace_back( cv::Point2f(tag_spacing_sz * tag_col,
                                                           tag_spacing_sz * tag_row + tag_sz) );// the fourth
                        break;
                    }
                }
            }

            //5.提取四个角点的亚像素角点坐标
            cv::cornerSubPix(img0_raw, img0_p2ds, cv::Size(2,2),cv::Size(-1, -1),
                            cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            cv::cornerSubPix(img1_raw, img1_p2ds, cv::Size(2,2),cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

        }
        else
        {
            std::cout << "Pattern type not supported yet.\n";
        }
    }
}

//sub function,used by calcCamPose
//估计该帧imag_raw所对应的相机位姿
bool EstimatePose(const std::vector<cv::Point3f> &p3ds,
                  const std::vector<cv::Point2f> &p2ds, const double &fx,
                  const double &cx, const double &fy, const double &cy,
                  Eigen::Matrix4d &Twc, cv::Mat &img_raw,const CameraPtr &cam)
{
  //１.检查pnp条件（用于计算的点对必须大于３）
  if (p3ds.size() != p2ds.size() || p3ds.size() < 4)
  {
    return false;
  }

  cv::Mat_<float> K =
      (cv::Mat_<float>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
  cv::Mat_<float> dist = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
  cv::Mat cv_r, cv_t;
  cv::Mat inliers;
  //2.计算得到姿态cv_r和位置cv_t(世界坐标相对相机的 Rcw)
  cv::solvePnP(p3ds, p2ds, K, dist, cv_r, cv_t, false, cv::SOLVEPNP_P3P);
  cv::Mat rotation;
  cv::Rodrigues(cv_r, rotation);
  Eigen::Matrix3d Rcw;
  cv::cv2eigen(rotation, Rcw);
  Eigen::Vector3d tcw;
  cv::cv2eigen(cv_t, tcw);
  Twc.block<3, 3>(0, 0) = Rcw.inverse();
  Twc.block<3, 1>(0, 3) = -Rcw.inverse() * tcw;

  //3.为了检验正确性，将世界坐标画在相机图像中
  std::vector<Eigen::Vector3d> axis;//将世界坐标系的原点，三个轴转换到相机坐标系中
  axis.push_back(Rcw * Eigen::Vector3d(0, 0, 0) + tcw);
  axis.push_back(Rcw * Eigen::Vector3d(0.5, 0, 0) + tcw);
  axis.push_back(Rcw * Eigen::Vector3d(0, 0.5, 0) + tcw);
  axis.push_back(Rcw * Eigen::Vector3d(0, 0, 0.5) + tcw);
  std::vector<Eigen::Vector2d> imgpts(4);
  for (int i = 0; i < 4; ++i)
  {
    cam->spaceToPlane(axis[i], imgpts[i]);//考虑畸变，将一个相机参考系中的三维坐标点转为图像像素坐标
  }
  // cv::projectPoints(axis, cv_r, cv_t, K, dist, imgpts);
  cv::line(img_raw, cv::Point2f(imgpts[0](0), imgpts[0](1)), cv::Point2f(imgpts[1](0), imgpts[1](1)), cv::Scalar(255, 0, 0), 2);//BGR
  cv::line(img_raw, cv::Point2f(imgpts[0](0), imgpts[0](1)), cv::Point2f(imgpts[2](0), imgpts[2](1)), cv::Scalar(0, 255, 0), 2);
  cv::line(img_raw, cv::Point2f(imgpts[0](0), imgpts[0](1)), cv::Point2f(imgpts[3](0), imgpts[3](1)), cv::Scalar(0, 0, 255), 2);
  cv::putText(img_raw, "X", cv::Point2f(imgpts[1](0), imgpts[1](1)), CV_FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(255, 0, 0));
  cv::putText(img_raw, "Y", cv::Point2f(imgpts[2](0), imgpts[2](1)), CV_FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 255, 0));
  cv::putText(img_raw, "Z", cv::Point2f(imgpts[3](0), imgpts[3](1)), CV_FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 0, 255));

  cv::putText(
      img_raw, "t_wc: (m) " + std::to_string(Twc(0, 3)) + " " + std::to_string(Twc(1, 3)) + " " + std::to_string(Twc(2, 3)),
      cv::Point2f(50, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));

  return true;
}

bool calcCamPose(const double &timestamps, const cv::Mat &image,
                 const CameraPtr &cam, Eigen::Matrix4d &Twc)
{
  cv::Mat img_raw = image.clone();
  //1.将彩色图像转为黑白图像
  if (img_raw.channels() == 3)
  {
    cv::cvtColor(img_raw, img_raw, CV_BGR2GRAY);
  }

  //2.寻找图像中的特征
  std::vector<cv::Point3f> p3ds;
  std::vector<cv::Point2f> p2ds;
  // FindTargetCorner(img_raw, CHESS, p3ds, p2ds);
  FindTargetCorner(img_raw, APRIL, p3ds, p2ds);


  //3.对特征进行矫正（使用相机模型）,得到矫正像素坐标un_pts
  //get the coordinates in the normalized camera coordinate system
  std::vector<double> p = cam->getK();//得到相机参数
  // std::cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;
  std::vector<cv::Point2f> un_pts;
  for (int i = 0, iend = (int)p2ds.size(); i < iend; ++i)
  {
    Eigen::Vector2d a(p2ds[i].x, p2ds[i].y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);//根据内参矫正为归一化坐标

    //????
    un_pts.push_back(
        cv::Point2f(p[0] * b.x() / b.z() + p[2], p[1] * b.y() / b.z() + p[3]));//再转换为像素坐标
    // std::cout << "p2ds: " << p2ds[i] << std::endl;
    // std::cout << "un_pts: " << un_pts[i] << std::endl;
  }

  //4.估计该帧所对应的相机位姿(Twc)
  if (EstimatePose(p3ds, un_pts, p[0], p[2], p[1], p[3], Twc, img_raw,cam))
  {
     cv::imshow("apriltag_detection & camPose_calculation", img_raw);
     cv::waitKey(1);
    return true;
  }
  else
  {
    return false;
  }
}

//calculate stereo camera VO
bool calcStereoCamPose(const double &timestamps,
        const cv::Mat& image0, const cv::Mat& image1,
        cv::Mat cam0_remap[2], cv::Mat cam1_remap[2],
        const cv::Mat &Q, std::vector<cv::Point3d> &cam_p3ds)
{
    if (image0.channels() == 3 && image1.channels() == 3)
    {
        cv::cvtColor(image0, image0, CV_BGR2GRAY);
        cv::cvtColor(image1, image1, CV_BGR2GRAY);
    }
    cv::Mat img_raw0;
    cv::Mat img_raw1;
    cv::remap(image0, img_raw0,
            cam0_remap[0], cam0_remap[1], cv::INTER_LINEAR);
    cv::remap(image1, img_raw1,
              cam1_remap[0], cam1_remap[1], cv::INTER_LINEAR);

    std::vector<cv::Point2f> img0_p2ds;
    std::vector<cv::Point2f> img1_p2ds;
    //find corners
    FindStereoTargetCorner(img_raw0, img_raw1, APRIL,img0_p2ds, img1_p2ds);
    double disp;
    std::vector<cv::Point3d> pixel_p3ds;
    for (int i = 0; i < img0_p2ds.size(); ++i)
    {
        disp = img0_p2ds[i].x - img1_p2ds[i].x;
        pixel_p3ds.emplace_back(cv::Point3d(img0_p2ds[i].x
                                            img0_p2ds[i].y
                                            disp));
    }
    cv::perspectiveTransform(pixel_p3ds, cam_p3ds, Q);
}

void calculate_VO (std::vector< std::vector<cv::Point3d> > cam_p3ds_queue,
                    Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
    if (cam_p3ds_queue.size() < 2)
        return;
    Point3f p1, p2;
    int N = cam_p3ds_queue[0].size();
    for (int i = 0; i < N; ++i)
    {
        p1 += cam_p3ds_queue[0][i];
        p2 += cam_p3ds_queue[1][i];
    }
    p1 /= N;
    p2 /= N;
    vector<Point3f> q1(N), q2(N);//remove the center
    for (int i = 0; i < N; ++i)
    {
        q1[i] = cam_p3ds_queue[0][i] - p1;
        q2[i] = cam_p3ds_queue[1][i] - p2;
    }
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) *
                Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    //SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU |
                                        Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    R = U *(V.transpose());
    t = Eigen::Vector3d(p1.x, p1.y, p1.z) -
                        R * Eigen::Vector3d(p2.x, p2.y, p2.z);
}
