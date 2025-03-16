# 梯度下降是一种优化算法，它使用目标函数的梯度来导航搜索空间。
# 可以通过使用称为Adam的偏导数的递减平均值，将梯度下降更新为对每个输入变量使用自动自适应步长。
# 如何从头开始实施Adam优化算法并将其应用于目标函数并评估结果。
# x（t）= x（t - 1）–step * f'（x（t-1）
from math import sqrt


from numpy.random import rand,seed
# 首先，对于作为搜索一部分而被优化的每个参数，我们必须维持一个矩矢量和指数加权无穷大范数，分别称为m和v（真是希腊字母nu）。在搜索开始时将它们初始化为0.0。
# m = 0 v = 0
# 该算法在从t = 1开始的时间t内迭代执行，并且每次迭代都涉及计算一组新的参数值x，
# 例如。从x（t-1）到x（t）。如果我们专注于更新一个参数，这可能很容易理解该算法，
# 该算法概括为通过矢量运算来更新所有参数。首先，计算当前时间步长的梯度（偏导数）。g（t）= f'（x（t-1））
# 接下来，使用梯度和超参数beta1更新第一时刻。m（t）= beta1 * m（t-1）+（1 – beta1）* g（t）
# 然后，使用平方梯度和超参数beta2更新第二时刻。v（t）= beta2 * v（t-1）+（1 – beta2）* g（t）^ 2
# 由于第一和第二力矩是用零值初始化的，所以它们是有偏的。接下来，对第一力矩和第二力矩进行偏差校正，并以第一力矩为起点：
#  mhat（t）= m（t）/（1 – beta1（t））
# 然后第二个时刻： vhat（t）= v（t）/（1 – beta2（t））
# 注意，beta1（t）和beta2（t）指的是beta1和beta2超参数，它们在算法的迭代过程中按时间表衰减。可以使用静态衰减时间表，不过论文建议：
# beta1（t）= beta1 ^ t
# beta2（t）= beta2 ^
# 最后，我们可以为该迭代计算参数的值。
# x（t）= x（t-1）– alpha * mhat（t）/（sqrt（vhat（t））+ eps）(eps防止除0出事)
# 注意，可以使用对本文中列出的更新规则进行更有效的重新排序：
# alpha（t）= alpha * sqrt（1 – beta2（t））/（1 – beta1（t））(学习率)
# x（t）= x（t-1）– alpha（t）* m（t）/（sqrt（v（t））+ eps）
#        alpha：初始步长（学习率），典型值为0.001。
#        beta1：第一个动量的衰减因子，典型值为0.9。
#        beta2：无穷大范数的衰减因子，典型值为0.999。

from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 确保导入了 Axes3D
from numpy.ma.core import asarray


# objective function，x2+y2，同时保证输入的范围[-1.0,1.0]
def objective(x, y):
    return x ** 2.0 + y ** 2.0
# 实现导数部分
def derivative(x, y):
    return asarray([ 2.0*x,2.0*y ])







#
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    solution = list()
    # generate an init ial point
    # 实现梯度下降优化。首先，我们可以选择问题范围内的随机点作为搜索的起点。
    # 假定我们有一个数组，该数组定义搜索范围，每个维度一行，并且第一列定义最小值，第二列定义维度的最大值。
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # 接下来，我们需要将第一时刻和第二时刻初始化为零。
    # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    # run iterations of gradient descent
    for t in range(n_iter):
        # 使用导数计算当前的梯度
        g = derivative(x[0], x[1])
        for i in range(bounds.shape[0]):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            # 计算力距
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
            # mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1 ** (t + 1))
            # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2 ** (t + 1)) # 随着深度加深
            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)

        # 评估一下
        score = objective(x[0], x[1])
        solution.append(score)
        print('>%d f(%s) = %.5f' % (t, x, score))

    return solution



seed(100)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 60
# steps size
alpha = 0.02
# factor for average gradient
beta1 = 0.8
# factor for average squared gradient
beta2 = 0.999
solutions = adam(objective,  derivative, bounds, n_iter, alpha, beta1, beta2)
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solutions = asarray(solutions)
# 假设 solutions 是一维数组，包含所有数据
x = solutions[:len(solutions)//2]  # 前半部分作为 x 坐标
y = solutions[len(solutions)//2:]  # 后半部分作为 y 坐标
pyplot.plot(x, y, '.-', color='w')
# show the plot
pyplot.show()



