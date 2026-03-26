#pretrain->sft->reward model->ppo
#ppo需要一个reward model来评估生成文本的质量，reward model可以通过人工标注或者自动化的方式来构建。下面是一个简单的PPO算法的实现示例：
#reward model 需要使用和现在大模型差不多或更好的模型训练，感觉可以用蒸馏
#还需要一个价值网络来评估当前策略的价值，价值网络可以通过监督学习的方式来训练，使用之前收集的数据来训练一个回归模型来预测当前策略的价值。
import torch
