import time
from keras.applications import vgg16
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imread, imsave
import numpy as np
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b

img_height = 504
img_width = 315

style_weight = 1.0
total_variation_weight = 1.0
# this will contain our generated image


result_prefix = 'result.jpg'

img_nrows = 400
img_ncols = int(img_width * img_nrows / img_height)

content_weight = 0.025

base_image_path = 'dest.jpg'
style_reference_image_path = 'src.jpg'


# 刚那个优化函数的输出是一个向量
def eval_loss_and_grads(x):
    # 把输入reshape层矩阵
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    # 激动激动，这里调用了我们刚定义的计算图！
    outs = f_outputs([x])
    loss_value = outs[0]
    # outs是一个长为2的tuple，0号位置是loss，1号位置是grad。我们把grad拍扁成矩阵
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        # 这个类别的事不干，专门保存损失值和梯度值
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        # 调用刚才写的那个函数同时得到梯度值和损失值，但只返回损失值，而将梯度值保存在成员变量self.grads_values中，这样这个函数就满足了func要求的条件
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        # 这个函数不用做任何计算，只需要把成员变量self.grads_values的值返回去就行了
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()


def preprocess_image(image_path):
    # 使用Keras内置函数读入图片，由于网络没有全连阶层，target_size可以随便设。
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    # 读入的图片用内置函数转换为numpy array格式，这两个函数都在keras.preprocessing.image里
    img = img_to_array(img)
    # ：维度扩展，这步在Keras用于图像处理中也很常见，Keras的彩色图输入shape是四阶张量，第一阶是batch_size。
    # 而裸读入的图片是3阶张量。为了匹配，需要通过维度扩展扩充为四阶，第一阶当然就是1了。
    img = np.expand_dims(img, axis=0)  # 3
    # vgg提供的预处理，主要完成（1）去均值（2）RGB转BGR（3）维度调换三个任务。
    # 去均值是vgg网络要求的，RGB转BGR是因为这个权重是在caffe上训练的，caffe的彩色维度顺序是BGR。
    # 维度调换是要根据系统设置的维度顺序th/tf将通道维调到正确的位置，如th的通道维应为第二维
    img = vgg16.preprocess_input(img)
    return img


# 可以看到，后处理的567三个步骤主要就是将#4的预处理反过来了，这是为了将处理过后的图片显示出来，resonable。
def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 读入内容和风格图，包装为Keras张量，这是一个常数的四阶张量
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# 初始化一个待优化图片的占位符，这个地方待会儿实际跑起来的时候要填一张噪声图片进来。

combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# 将三个张量串联到一起，形成一个形如（3,3,img_nrows,img_ncols）的张量
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
# 设置Gram矩阵的计算图，首先用batch_flatten将输出的featuremap压扁，然后自己跟自己做乘法，跟我们之前说过的过程一样。注意这里的输入是某一层的representation。
def gram_matrix(x):
    assert K.ndim(x) == 3

    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# 设置风格loss计算方式，以风格图片和待优化的图片的representation为输入。
# 计算他们的Gram矩阵，然后计算两个Gram矩阵的差的二范数，除以一个归一化值，公式请参考文献[1]
def style_loss(style, combination):  # 2
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# 设置内容loss计算方式，以内容图片和待优化的图片的representation为输入，计算他们差的二范数，公式参考文献[1]
def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# 施加全变差正则，全变差正则用于使生成的图片更加平滑自然。
def total_variation_loss(x):
    assert K.ndim(x) == 4

    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 这是一个张量字典，建立了层名称到层输出张量的映射，通过这个玩意我们可以通过层的名字来获取其输出张量，比较方便。当然不用也行，使用model.get_layer(layer_name).output的效果也是一样的。
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# loss的值是一个浮点数，所以我们初始化一个标量张量来保存它
loss = K.variable(0.)

# layer_features就是图片在模型的block4_conv2这层的输出了，记得我们把输入做成了(3,3,nb_rows,nb_cols)这样的张量，
# 0号位置对应内容图像的representation，1号是风格图像的，2号位置是待优化的图像的。计算内容loss取内容图像和待优化图像即可
layer_features = outputs_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
# 与上面的过程类似，只是对多个层的输出作用而已，求出各个层的风格loss，相加即可。
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

# 求全变差约束，加入总loss中
loss += total_variation_weight * total_variation_loss(combination_image)

# 通过K.grad获取反传梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
# 我们希望同时得到梯度和损失，所以这两个都应该是计算图的输出
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)
# 编译计算图。Amazing！我们写了辣么多辣么多代码，其实都在规定输入输出的计算关系，到这里才将计算图编译了。
# 这条语句以后，f_outputs就是一个可用的Keras函数，给定一个输入张量，就能获得其反传梯度了。
f_outputs = K.function([combination_image], outputs)

# 根据后端初始化一张噪声图片，做去均值
if K.image_dim_ordering() == 'th':
    x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

# 迭代10次
for i in range(10):
    print('Start of iteration', i)
    start_time = time.time()
    # 这里用了一个奇怪的函数 fmin_l_bfgs_b更新x，我们一会再看它，这里知道它的作用是更新x就好
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    # 每次迭代完成后把输出的图片后处理一下，保存起来
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)  # 4
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
