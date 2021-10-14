import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from Generator.DeblurGanBlur.generate_trajectory import Trajectory
import glob

class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None, path_to_save=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False).x
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1/50, 1/10, 1/2, 1]
        else:
            self.fraction = fraction
        self.path_to_save = path_to_save
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self, show=False, save=False):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))
        if show or save:
            self.__plot_canvas(show, save)

        return self.PSFs

    def __plot_canvas(self, show, save):
        if len(self.PSFs) == 0:
            raise Exception("Please run fit() method first.")
        else:
            plt.close()
            fig, axes = plt.subplots(1, self.PSFnumber, figsize=(10, 10))
            for i in range(self.PSFnumber):
                axes[i].imshow(self.PSFs[i], cmap='gray')
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()


def saveKernels(kernel_num=1000):
    count = [0 for i in range(50)]
    count[0] = 4
    count[1] = 4
    count[2] = 4
    id = 0
    while(sum(count) < 4 * 50):
        psf = PSF(canvas=129, path_to_save='/home/wangzerun/BlurGenerator/DeblurGanBlur/psf')
        psfs = psf.fit(show=False, save=False)
        for p in psfs:
            size = dePadPsf(p)
            if size < 50:
                if count[size] < 4:
                    count[size] += 1
                    np.save('/home/wangzerun/BlurGenerator/DeblurGanBlur/psf/{}_{}.npy'.format(id,size), p)
                    id += 1
                    print('generated size {}'.format(size))



def dePadPsf(psf):
    temp = psf.copy()
    idx = np.argwhere(np.all(temp[..., :] == 0, axis=0))
    temp = np.delete(temp, idx, axis=1)
    idx = np.argwhere(np.all(temp[:, ...] == 0, axis=1))
    temp = np.delete(temp, idx, axis=0)
    return max(temp.shape[0], temp.shape[1])

def statKernels():
    import collections
    path = '/home/wangzerun/BlurGenerator/DeblurGanBlur/psf'
    label_list = glob.glob(path + '/*.npy')
    numbers = []
    for l in label_list:
        num = int(l.split('_')[1].split('.')[0])
        numbers.append(num)
    b = collections.Counter(numbers)
    # 转换成字典的格式
    dic = {number: value for number, value in b.items()}
    plt.title = "统计数字出现的次数"
    # 取得key
    x = [i for i in dic.keys()]
    y = []
    # 取得value
    for i in dic.keys():
        y.append(dic.get(i))
    plt.hist(numbers, bins=len(dic))
    plt.show()
    count = [0 for i in range(50)]
    count[0] = 4
    count[1] = 4
    count[2] = 4


if __name__ == '__main__':
    #psf = PSF(canvas=129, path_to_save='/home/wangzerun/BlurGenerator/DeblurGanBlur/psf')
    #psf.fit(show=True, save=False)
    #saveKernels()
    statKernels()