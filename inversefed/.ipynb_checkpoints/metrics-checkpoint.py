"""This is code based on https://sudomake.ai/inception-score-explained/."""
import torch
import torchvision

from collections import defaultdict

class InceptionScore(torch.nn.Module):
    """Class that manages and returns the inception score of images.""" """管理和返回图像初始分数的类。"""

    def __init__(self, batch_size=32, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with setup and target inception batch size. 使用设置和目标初始批量大小进行初始化。"""
        super().__init__()
        self.preprocessing = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False) #上采样
        self.model = torchvision.models.inception_v3(pretrained=True).to(**setup) #模型
        self.model.eval() #eval() 1.dropout层会让所有的激活单元都通过，2.batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值
        self.batch_size = batch_size

    def forward(self, image_batch):
        """Image batch should have dimensions BCHW and should be normalized. 图像批次应该具有 BCHW 尺寸并且应该被规范化。

        B should be divisible by self.batch_size. B应该能被 self.batch_size 整除。
        """
        B, C, H, W = image_batch.shape
        batches = B // self.batch_size
        scores = []
        for batch in range(batches):
            input = self.preprocessing(image_batch[batch * self.batch_size: (batch + 1) * self.batch_size])
            scores.append(self.model(input))
        prob_yx = torch.nn.functional.softmax(torch.cat(scores, 0), dim=1) #横着拼接scores和0
        entropy = torch.where(prob_yx > 0, -prob_yx * prob_yx.log(), torch.zeros_like(prob_yx)) #zero_like: 生成和括号内变量维度维度一致的全是零的内容 where:当prob_yx>0,取值 -prob_yx * prob_yx.log()；否则，取0；
        return entropy.sum()


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR. 计算PSNR值"""
    def get_psnr(img_in, img_ref):
        print(f'factor={factor}')
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        print("this")
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def bn_regularizer(feature_maps, bn_layers):
    bn_reg = 0
    for i, layer in enumerate(bn_layers):
        # print(f'i = {i}')
        # #print(f'layer = {layer}')
        print(f'feature_maps[i] = {feature_maps[i]}')
        fm = feature_maps[i]
        print(f'fm = {fm}')
        if len(fm.shape) == 3:
            dim = [0, 2]
        elif len(fm.shape) == 4:
            dim = [0, 2, 3]
        elif len(fm.shape) == 5:
            dim = [0, 2, 3, 4]
        bn_reg += torch.norm(fm.mean(dim=dim) - layer.state_dict()["running_mean"], p=1)
        bn_reg += torch.norm(fm.var(dim=dim) - layer.state_dict()["running_var"], p=1)
    return bn_reg

def group_consistency(x, group_x):
    mean_group_x = sum(group_x) / len(group_x)
    return torch.norm(x - mean_group_x, p=2)

def canny_consistency(x, canny_x):
#     print(x)
    x = torch.Tensor(x)
#     print(f'x.shape={x.shape}')
    canny_x =  torch.Tensor(canny_x )
#     print(f'canny_x.shape={canny_x.shape}')
    return torch.norm(x - canny_x, p=2)

def activation_errors(model, x1, x2):
    """Compute activation-level error metrics for every module in the network. 计算网络中每个模块的激活级错误度量。"""
    model.eval()

    device = next(model.parameters()).device

    hooks = []
    data = defaultdict(dict)
    inputs = torch.cat((x1, x2), dim=0) #竖向拼接
    separator = x1.shape[0]

    def check_activations(self, input, output):
        module_name = str(*[name for name, mod in model.named_modules() if self is mod]) # 模型名称
        try:
            layer_inputs = input[0].detach()
            residual = (layer_inputs[:separator] - layer_inputs[separator:]).pow(2)
            se_error = residual.sum()
            mse_error = residual.mean()
            sim = torch.nn.functional.cosine_similarity(layer_inputs[:separator].flatten(),
                                                        layer_inputs[separator:].flatten(),# flatten()展平矩阵，输出tensor
                                                        dim=0, eps=1e-8).detach() #计算cosine相似度
            data['se'][module_name] = se_error.item()
            data['mse'][module_name] = mse_error.item()
            data['sim'][module_name] = sim.item()
        except (KeyboardInterrupt, SystemExit):
            raise
        except AttributeError:
            pass

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(check_activations)) # 将获取的输入输出添加到module列表中

    try:
        outputs = model(inputs.to(device))
        for hook in hooks:
            hook.remove()
    except Exception as e:
        for hook in hooks:
            hook.remove()
        raise

    return data
