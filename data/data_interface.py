# # 训练脚本调用 Trainer.fit(model, datamodule=DInterface(...)) →
# # │
# # ├── __init__()               ✅ 读取参数、保存配置、动态导入数据集类
# # │
# # ├── setup(stage='fit')       ✅ 实例化 train / val 数据集
# # │     └── self.instancialize(train=True / False)
# # │          └── 自动推理参数 + 创建数据集类实例（StandardData）
# # │
# # ├── train_dataloader()       ✅ 创建 PyTorch DataLoader(trainset, ...)
# # ├── val_dataloader()
# # └── test_dataloader()

# import inspect
# import importlib
# import pickle as pkl
# import lightning.pytorch as pl
# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import WeightedRandomSampler


# class DInterface(pl.LightningDataModule):

#     def __init__(self, 
#                 num_workers=8,
#                 dataset='plant_data',
#                 data_csv='./data/train.csv',         # 必传
#                 soft_labels_filename=None,           # 可选
#                 image_folder='./data/images',
#                 image_size=(224, 224),
#                 batch_size=32,
#                  **kwargs):
#         #**kwargs 表示任意数量的键值对参数（keyword arguments），以字典形式传入函数。
#         super().__init__()
#         self.num_workers = num_workers #num_workers: DataLoader 的工作线程数
#         self.dataset = dataset
#         self.kwargs = kwargs
#         self.batch_size = kwargs['batch_size']
#         self.load_data_module() #它会动态加载对应的数据集模块 + 类（比如 StandardData）

#     def setup(self, stage=None):
#         # Assign train/val datasets for use in dataloaders 这是 Lightning 生命周期钩子函数，用于构建数据集：
#         if stage == 'fit' or stage is None:
#             self.trainset = self.instancialize(train=True)
#             self.valset = self.instancialize(train=False)

#         # Assign test dataset for use in dataloader(s)
#         if stage == 'test' or stage is None:
#             self.testset = self.instancialize(train=False)

#         # # If you need to balance your data using Pytorch Sampler,
#         # # please uncomment the following lines.
    
#         # with open('./data/ref/samples_weight.pkl', 'rb') as f:
#         #     self.sample_weight = pkl.load(f)

#     # def train_dataloader(self):
#     #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
#     #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

#     def train_dataloader(self):
#         return DataLoader(
#             self.trainset, 
#             batch_size=self.batch_size, 
#             num_workers=self.num_workers, 
#             shuffle=True,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.valset, 
#             batch_size=self.batch_size, 
#             num_workers=self.num_workers, 
#             shuffle=False,
#             pin_memory=True
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.testset, 
#             batch_size=self.batch_size, 
#             num_workers=self.num_workers, 
#             shuffle=False,
#             pin_memory=True
#         )

#     def load_data_module(self):
#         name = self.dataset
#         # Change the `snake_case.py` file name to `CamelCase` class name.
#         # Please always name your model file name as `snake_case.py` and
#         # class name corresponding `CamelCase`.
#         camel_name = ''.join([i.capitalize() for i in name.split('_')]) #把蛇形命名的文件名转成驼峰命名的类名
#         try:
#             self.data_module = getattr(importlib.import_module(
#                 '.'+name, package=__package__), camel_name)
#         except:
#             raise ValueError(
#                 f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

   
#     def instancialize(self, **other_args):
#         """ Instancialize a model using the corresponding parameters
#             from self.hparams dictionary. You can also input any args
#             to overwrite the corresponding value in self.kwargs.
#         """
#          #动态参数匹配 + 自动初始化类实例
#         class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
#         inkeys = self.kwargs.keys()
#         args1 = {}
#         for arg in class_args:
#             if arg in inkeys:
#                 args1[arg] = self.kwargs[arg]
#         args1.update(other_args)
#         return self.data_module(**args1)
