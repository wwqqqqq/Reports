1. 视觉行人检测
   * 基于人脸识别的跟踪
     * 人脸识别的速度和正确率均已达到一个很高的层次，但在实际的激动机器人跟随场景中，人不是一直面对移动机器人。
   * 基于模板匹配的跟踪
   * 基于轮廓信息的跟踪
2. 激光行人检测
   * 使用几何特征识别目标
   * 基于运动检测识别目标


跟踪算法
1. Particle Filter
   * 首先对跟踪目标进行建模，并定义一种相似度度量确定粒子与目标的匹配程度。在目标搜索的过程中，它会先按照一定的分布（比如均匀分布或高斯分布）在全局撒一些粒子，统计这些粒子与目标的相似度，确定目标可能的位置。在可能性较高的位置上，下一帧加入更多新的粒子，确保在更大概率上跟踪上目标。
   * 跟踪速度很快，而且能解决目标的部分遮挡问题。
2. MeanShift
   * 均值漂移算法，沿着概率密度的梯度方向进行迭代移动，最终达到密度分布的极大值位置。本质是梯度下降法找出局部概率密度最大值。
   * 此方法可以通过较少的迭代次数快速找到与目标最相似的位置效果较好。缺点是不能解决目标的遮挡问题，且不能适应运动目标的形状和大小变化等。
3. CamShift
   * MeanShift算法的改进，可以适应运动目标的大小形状的改变，具有较好的跟踪效果，但当背景色和目标颜色接近时，容易使目标的区域变大，最终有可能导致目标跟踪丢失
4. MHT 
5. Kalman Filter
   * 该方法认为物体的运动模型服从高斯模型，从而对目标的运动状态进行预测，然后通过与观察模型进行对比，根据误差来更新运动目标的状态。
   * 精度不算很高，限定目标运动服从线性高斯分布。
6. 相关滤波 correlation filter
   * 一种基于循环矩阵的核跟踪方法，利用傅里叶变换快速实现了检测的过程。在训练分类器时，一般认为离目标位置较近的是正样本，而离目标较远的认为是负样本。
   * 利用傅里叶变换加速计算，可以达到100帧/秒以上的跟踪效果。可以高效的融合多种特征，如HOG、deep feature等。
7. 深度学习
   * MDNet
   * TCNN
   * SiamFC
   * GOTURN

8.  基于特征匹配的目标跟踪
   * 通过前后帧之间的特征匹配实现目标的定位。目标跟踪中用到的特征主要有几何形状，子空间特征，外形轮廓和特征点。
   * 贝叶斯跟踪：目标的运动往往是随机的，这样的运动过程可以采用随机过程来描述。
   * 核方法：对相似度概率密度函数或者后验概率密度函数采用直接的连续估计。
11. 基于运动检测的目标跟踪
   * 根据目标运动和背景运动之间的差异实现目标的检测和追踪。
   * 帧间图像差分法、背景估计法、能力累积法、运动场估计法、光流算法等。

一般把目标跟踪分为两个部分：特征提取 + 目标跟踪算法
目标特征大致可以分为以下几种：
1. 以目标区域的颜色直方图作为特征。颜色特征具有旋转不变性，且不受目标物大小和形状的变化影响，在颜色空间中分布大致相同
2. 目标的轮廓特征。算法速度较快，且在目标有小部分遮挡的情况下同样有良好的效果
3. 目标的纹理特征。较轮廓特征跟踪效果会有所改善。

难点：外观变形、光照变化、快速运动、运动模糊、背景相似干扰、平面外旋转、平面内旋转、尺度变化、遮挡、出视野

摄像头运动时无法通过背景相减法获得目标的具体位置和大小描述，常用方法有
1. 质心跟踪算法
2. 边缘跟踪算法
3. 场景锁定跟踪算法
4. 组合跟踪算法


# Approaches
1. Particle filters | multiple hypothesis tracking (MHT) | Kalman filters to estimate the current position of people, starting from their last known positions and the current sensor information.
   1. use the information provided by a laser scanner to detect people's legs
      * [Multiple hypothesis tracking of clusters of people](http://persoal.citius.usc.es/manuel.mucientes/pubs/Mucientes06_iros.pdf)
      * [Efficient people tracking in laser range data using a multi-hypothesis leg-tracker with adaptive occlusion probabilities](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4543447)
      * [Laser based people following behaviour in an emergency environment](https://link.springer.com/content/pdf/10.1007%2F978-3-642-10817-4.pdf)
   2. combine leg and face detection to obtain the position and track a person
      * [Person tracking with a mobile robot based on multi-modal anchoring](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5194&rep=rep1&type=pdf)
      * [Multisensor-based human detection and tracking for mobile service robots](http://eprints.lincoln.ac.uk/2096/1/Bellotto2009.pdf)
2. Frame differencing + Particle filters + EM algorithm [People Tracking and Following with Mobile Robot Using an Omnidirectional Camera and a Laser](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1641769)
3. uses the laser to extract the 3D relative position of blobs that might have originated from a person and uses these as measurements to a probabilistic data associate filter (PDAF) [People Tracking and Following with Mobile Robot Using an Omnidirectional Camera and a Laser](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1641769)
4. HOG + SVM (Entire body, color and texture)
   * [Histograms of Oriented Gradients for Human Detection](https://hal.inria.fr/file/index/docid/548512/filename/hog_cvpr2005.pdf) 
   * [Feature Analysis for Human Recognition and Discrimination: Application to a Person-Following Behavior in a Mobile Robot](https://www.sciencedirect.com/science/article/pii/S092188901200070X)
     * sensor fusion (Vision + Laser)
5. remove the ground -> 3D clustering -> Euclidean CLustering algorithm -> sub-clustering to detect persons -> HOG-based people detection to the resulting clusters -> online AdaBoost
   * [A Software Architecture for RGB-D People Tracking Based on ROS Framework for a Mobile Robot](https://link.springer.com/content/pdf/10.1007%2F978-3-642-35485-4.pdf)
6. Partially Observable Monte-Carlo Planning (POMCP)
   * [Continuous Real Time POMCP to Find-and-Follow People by a Humanoid Service Robot](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7041445)
7. a pair wise cluster tracker is empleyed to localize the person. A positive and negative classifier is then utilized to verify the tracker's result and to update the tracking model. In addition, a detector pre=trained by a CNN is used to further improve the result of tracking.
   * [A Classification-Lock Tracking Strategy Allowing a Person-Following Robot to Operate in a Complicated Indoor Environment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6263996/pdf/sensors-18-03903.pdf)
8. Person re-indentication
   * [User Recognition for Guiding and Following People with a Mobile Robot in a Clinical Environment](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7353880)
9. Vision + RFID | Particle filtering
   * [Vision and RFID data fusion for tracking peole in crowds by a mobile robot](https://www.sciencedirect.com/science/article/pii/S1077314210000317)


# Features:
1. height and gait [Identification of a Specific Person Using Color, Height, and Gait Features for a Person Following Robot](https://www.sciencedirect.com/science/article/pii/S0921889015303225)
   * Color features can easily be extracted from an image and are effective for identifying a person by thier clothing color. Texture and shape features, such as HOG and SIFT, are also used for a more robust person identification. However, all such appearance features are weak under severe illumination environments, we thus use only a color feature as an appearance feature for simplicity.
   * Color histogram is one of the most popular representations for color modeling. We use a hue-saturation histogram to reduce the effect of light intensity changes.
   * Even if there are multiple persons with similar heights, the height is useful for reducing the number of candidates for the target person. To calculate the height of a person, we first determine the topmost position in the image (i.e. sinciput of the head region) and then estimate the height using the camera geometry.
   * A saturation-intensity histogram of a hair region is computed from the hair images in advance, and then a Gaussian mixture model (GMM) is fitted to the histogram. We make two images from an input image, one representing the similarity of hair color and the other representing the magnitude of the gradient, and calculate the pixel-wise product of the images. The pixel which has the highest product value is considered as the sinciput of the person.
   * By using an RGB-D camera, we can separate the person region from the background region, and then extract gait feature.
   * When a person is walking, the legs of the person swing and stop alternately. The interval when a leg is swinging is refered to as swing phase. During stance phase, the leg which stops and supports the body of the person is referred to as a supporting leg. If we can obtain the supporting leg position (where the leg touches the ground), we can calculate gait features, such as a step length and a stance width.
2. face recognition + upper body recognition [Reliable People Tracking Approach for Mobile Robot in Indoor Environments](https://www.sciencedirect.com/science/article/pii/S0736584509000611/pdfft?md5=eafdb955f326ea7c61697f7130c167e6&pid=1-s2.0-S0736584509000611-main.pdf)
3. Each layer contains a classifier able to detect a particular body part such as head, an upper body or a leg [Multi-Part People Detection Using 2D Range Data](http://robotics.ait.kyushu-u.ac.jp/kurazume/papers/IJSR10-Oscar.pdf)
4. Color and texture (extract from the human's torso region) [Feature Analysis for Human Recognition and Discrimination: Application to a Person-Following Behavior in a Mobile Robot]

