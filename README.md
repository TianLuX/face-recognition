# face-recognition
基于lfw数据集的人脸识别<br>
分别使用vgg-16和resnet-50作为主干网络<br>
使用面部矫正器faceAligner,将人脸旋转至水平，并居中放大，存入lfw_align文件夹中<br>
并使用dataAugmentataion.py在lfw数据集中增加采样，完成数据增强<br>
运行时需要首先运行data.py文件，生成label和set的pkl文件存入dataset文件夹中(pkl文件过大没有上传，需要运行生成)<br>
train.py为训练和测试文件<br>
output文件夹中为不同数据量和主干网络的测试结果，以及结果的可视化（折线图）