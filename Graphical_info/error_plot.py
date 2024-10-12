import matplotlib.pyplot as plt

error_list = []

with open("/home/chiranjeet/c++_maths/errortext/error.txt","r") as file:
    for line in file:
       error_list.append(float(line.strip()))

plt.plot(error_list,'r')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid()
plt.tight_layout()
plt.savefig('/home/chiranjeet/c++_maths/errortext/error_plot.png')
plt.show()