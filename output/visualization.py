import matplotlib.pylab as plt
path = "resnet-5200.txt"
with open(path) as f:
    outputs = f.readlines()
    loss = []
    acc = []
    for output in outputs:
        output = output.strip(';\n')
        if 'loss' in output:
            output = output.split(' ')
            loss.append(float(output[-1]))
        elif 'acc' in output:
            output = output.split(' ')
            acc.append(float(output[-1]))
# print(loss)
# print(acc)
# loss值的变化
fig, ax = plt.subplots()
ax.set_title("loss", fontsize=12)
ax.plot(loss, color="brown")
plt.show()

# acc值的变化
fig1, ax = plt.subplots()
ax.set_title("acc", fontsize=12)
ax.set_ylim(ymin=0, ymax=1)
ax.plot(acc, color="red")
plt.show()


