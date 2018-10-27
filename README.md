
"""
	checkpoints/： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练
	data/：数据相关操作，包括数据预处理、dataset实现等
	models/：模型定义，可以有多个模型，例如上面的AlexNet和ResNet34，一个模型对应一个文件
	utils/：可能用到的工具函数，在本次实验中主要是封装了可视化工具
	config.py：配置文件，所有可配置的变量都集中在此，并提供默认值
	main.py：主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数
	requirements.txt：程序依赖的第三方库
	README.md：提供程序的必要说明


	模型的定义主要保存在models/目录下，其中BasicModule是对nn.Module的简易封装，提供快速加载和保存模型的接口。
"""
