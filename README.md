# Cam_IMU_Calib_for_Vehicles
## 在自动驾驶车辆中，标定相机和IMU的外参

## 1. Prerequisites
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
录制相机和IMU数据一定要充分
### 3.2 通过UcoSLAM获取相机里程计


## 4. 执行标定
修改配置文件，选择正确的路径
```
    ./main_node
    
```

