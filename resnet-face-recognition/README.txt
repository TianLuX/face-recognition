代码运行步骤：
1、在resnet-face-recognition文件夹下新建dataset文件夹，用于保存pkl数据
2、运行data文件，生成训练数据和测试数据
3、运行train_test_triplet文件，训练和测试模型

-------------------------------------------------
主要文件说明：
lfw_align：处理后的人脸照片，将人脸放大后旋转居中以相同目录形式存储
contrastTrain.txt：三元对比训练数据
pairsDevTrain.txt：二元对比训练数据
parisDevTest.txt：测试数据
data：生成训练和测试所需的pkl数据
module：存放主干网络和损失函数
train_test_triplet：训练和测试模型
