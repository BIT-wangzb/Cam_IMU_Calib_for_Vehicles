# Cam_IMU_Calib_for_Vehicles
## 在自动驾驶车辆中，标定相机和IMU的外参
自动驾驶车辆的平面运动，会使标定问题退化，所以该代码将时间偏移标定，旋转外参标定和平移标定分为三个独立的过程。
### ---时间偏移标定借助Kalibr运行；
使用本代码提供的IccSensors.py替换Kalibr中的IccSensors.py
### ---旋转外参可以从车辆的平面运动中进行精确标定；
### ---平移外参可以从车辆的斜坡运动中进行标定。

## 1. 依赖项
### 1.1 **Ubuntu**
在Ubuntu20.04上测试
### 1.2. **Ceres Solver**
### 1.3. **其他**
OpenCV 3.4 , Eigen3

## 2. 编译

```
    mkdir build && cd build
    cmake .. && make -j6
```
## 3. 注意

### 3.1 录制数据
录制相机和IMU数据时，车辆平面运动以'8'字在近似水平面的路面上行驶；车辆的斜坡运动尽量以蛇形运动行驶，坡度倾角越大越好。
### 3.2 通过UcoSLAM或其他方法获取带尺度的相机里程计

## 4. 执行标定
修改配置文件，选择正确的路径
```
    ./main_node 
```
## 参考论文
Guo, C. X., Mirzaei, F. M., & Roumeliotis, S. I. (2012, May). An analytical least-squares solution to the odometer-camera extrinsic calibration problem. In 2012 IEEE International Conference on Robotics and Automation (pp. 3962-3968). IEEE.

## 参考代码
```
    https://github.com/MegviiRobot/CamOdomCalibraTool
```
