import json

import matplotlib.pyplot as plt


def plot_result(mode_list, task1, task2, task3):
    # draw the lines
    count = 0
    for reg_name in mode_list:
        label = reg_name
        with open('./{reg_name}_acc.txt', 'r') as f:
            acc = json.load(f)
        if count == 0:
            color = 'red'
        elif count == 1:
            color = 'blue'
        else:
            color = 'purple'
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(range(len(acc[task1])), acc[task1], color, label=label)
        ax1.set_ylabel(task1)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
        plt.plot(range(len(acc[task3]), len(acc[task1])), acc[task2], color, label=label)
        ax2.set_ylabel(task2)
        ax3 = plt.subplot(3, 1, 3, sharex=ax1, sharey=ax1)
        ax3.set_ylabel(task3)
        plt.plot(range(len(acc[task2]), len(acc[task1])), acc[task3], color, label=label)
        count += 1
    plt.ylim((0.02, 1.02))
    plt.legend()
    plt.show()
    return


mode_list = ['ewc', 'mas', 'basic']
plot_result(mode_list, 'SVHN', 'MNIST', 'USPS')
