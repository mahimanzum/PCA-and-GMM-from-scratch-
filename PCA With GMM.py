import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

with open('data_online.txt') as f:
    train_data = []
    for line in f:
        train_data.append([float(x) for x in line.split()])
train_data = np.array(train_data)
cov_matrix = np.cov(train_data.T)
values, vectors = eig(cov_matrix)
vectors = vectors.T
#print(values)
indx = [i for i in range(len(values))]
sort_list = zip(values, indx)
sort_list = sorted(sort_list, key = lambda t: t[0])
pca1_idx = sort_list[-1][1]
pca2_idx = sort_list[-2][1]
pc1 = vectors[pca1_idx]
pc2 = vectors[pca2_idx]
transform = []
transform.append(pc1)
transform.append(pc2)
transform = np.array(transform)
trans_data = np.dot(train_data, transform.T)
#plt.scatter(trans_data[:, 0], trans_data[:, 1], s=5, c= (0, 0, 0), alpha=0.5)
def draw_main():
    plt.scatter(trans_data[:, 0], trans_data[:, 1], alpha=0.2)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
draw_main()
#EM starting 
n_clusters = 4
#print(len(trans_data))

P = []
for i in range(len(trans_data)):
    probs = np.random.rand(n_clusters)
    P.append(probs/np.sum(probs))
P = np.array(P)
mu = []
X = trans_data.copy()
for k in range(n_clusters):
    avgs = []
    avgs.append(np.sum(P[:, k]*X[:,0])/np.sum(P[:, k]))
    avgs.append(np.sum(P[:, k]*X[:,1])/np.sum(P[:, k]))
    mu.append(avgs)
'''
mu[0] = [-4,6]
mu[1] = [4,-6]
mu[2] = [10,6]
'''
mu[0] = [0.9,2.5]
mu[1] = [7.3,2.7]
mu[2] = [3.76,-1.61]
k = 2
mean_x = np.sum(trans_data[:,0])/len(trans_data)
mean_y = np.sum(trans_data[:,1])/len(trans_data)

#mean_x = 0
#mean_y = 0
print("meanx = ", mean_x, "mean_y = ", mean_y)
mu[0] = [mean_x+(np.random.rand()-0.5)*k,mean_y+(np.random.rand()-0.5)*k]
mu[1] = [mean_x+(np.random.rand()-0.5)*k,mean_y+(np.random.rand()-0.5)*k]
mu[2] = [mean_x+(np.random.rand()-0.5)*k,mean_y+(np.random.rand()-0.5)*k]


mu = np.array(mu)

sigma = []
for k in range(n_clusters):
    X = trans_data.copy()
    X[:] = X[:] - mu[k]
    New_x = X.copy()
    New_x[:, 0] = P[:, k]*X[:, 0]
    New_x[:, 1] = P[:, k]*X[:, 1]
    sigma.append(np.array([[1,0.1],[0.1,1.0]]))
    #sigma.append(np.dot(New_x.T,X)/np.sum(P[:, k]))
sigma = np.array(sigma)
#draw_main()
w = []
for k in range(n_clusters):
    w.append(np.sum(P[:, k])/len(trans_data))
    w[k] = 0.33
print(w)
w = np.array(w)

X = trans_data.copy()
for i in range(len(trans_data)):
    sm = 0.0
    for k in range(n_clusters):
        val = w[k]*multivariate_normal.pdf(X[i],mu[k],sigma[k])
        P[i][k] = val
        sm += val
    for k in range(n_clusters):
        P[i][k] /= sm
#print("mu = ", mu,"sigma = ", sigma, "w = ", w)

def draw():
    cl_0 = []
    cl_1 = []
    cl_2 = []
    for i in range(len(trans_data)):
        mx = -1
        cls = 0
        for k in range(n_clusters):
            if P[i][k]>mx:
                mx = P[i][k]
                cls = k
        if cls == 0:
            cl_0.append(trans_data[i])
        elif cls == 1:
            cl_1.append(trans_data[i])
        else:
            cl_2.append(trans_data[i])
    cl_0 = np.array(cl_0)
    cl_1 = np.array(cl_1)
    cl_2 = np.array(cl_2)
    if cl_0.shape[0]>0:
        plt.scatter(cl_0[:, 0],cl_0[:, 1],color='red')
    if cl_1.shape[0]>0:
        plt.scatter(cl_1[:, 0],cl_1[:, 1],color='blue')
    if cl_2.shape[0]>0:
        plt.scatter(cl_2[:, 0],cl_2[:, 1],color='green')
    #plt.draw()
    #plt.flush_events()
    plt.scatter(mu[0][0],mu[0][1], s=300 ,c='black', marker="d", alpha=0.5)
    plt.scatter(mu[1][0],mu[1][1], s=300 ,c='black', marker="d", alpha=0.5)
    plt.scatter(mu[2][0],mu[2][1], s=300 ,c='black', marker="d", alpha=0.5)
    
    for k in range(n_clusters):
        x, y = np.mgrid[-4:10:.5, -6:6:.5]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv = multivariate_normal(mu[k], sigma[k])
        if(k == 0):
            plt.contour(x, y, rv.pdf(pos), c = 'red', alpha=0.5)
        elif(k == 1):
            plt.contour(x, y, rv.pdf(pos), c = 'blue', alpha=0.5)
        else:
            plt.contour(x, y, rv.pdf(pos), c = 'green', alpha=0.5)
    
    plt.show()
draw()
prev = 0
def converged():
    prob_sum = 0.0
    prob = 0.0
    global prev
    for i in range(len(trans_data)):
        for k in range(n_clusters):
            prob_sum += w[k]*multivariate_normal.pdf(trans_data[i],mu[k],sigma[k])
        prob += np.log(prob_sum)
    #print(prob)
    if abs(prob-prev) <= 0.001:
        return True
    prev = prob
    return False
n_epochs = 2001

for epoch in range(n_epochs):
    #print("iteration = ", epoch)
    #expectation step
    if converged():
        break
    X = trans_data.copy()
    for i in range(len(trans_data)):
        sm = 0.0
        for k in range(n_clusters):
            val = w[k]*multivariate_normal.pdf(X[i],mu[k],sigma[k])
            P[i][k] = val
            sm += val
        for k in range(n_clusters):
            #print(sm)
            P[i][k] /= sm    
    #maximization step
    X = trans_data.copy()
    for k in range(n_clusters):
        avgs = []
        avgs.append(np.dot(P[:, k].T, X[:,0])/np.sum(P[:, k]))
        avgs.append(np.dot(P[:, k].T, X[:,1])/np.sum(P[:, k]))
        mu[k] = np.array(avgs)
    mu = np.array(mu)
    X = trans_data.copy()
    sg = np.array([[0.0, 0.0],[0.0, 0.0]])
    for k in range(n_clusters):
        sg = np.array([[0.0, 0.0],[0.0, 0.0]])
        for i in range(len(trans_data)):
            sg += P[i][k]*np.dot(np.transpose([X[i]-mu[k]]),[X[i]-mu[k]])
            #print(np.dot(np.transpose([X[i]-mu[k]]),[X[i]-mu[k]]))
        sigma[k] = sg/np.sum(P[:,k])
        #print(sigma[k])
    for k in range(n_clusters):
        w[k] = np.sum(P[:, k])/len(trans_data)
    #show
    if epoch%1 == 0:
        draw()
        print(mu)
draw()
for k in range(n_clusters):
    print("cluster ", k , "probability sum = ", np.sum(P[:, k]))
#draw_main()
'''
[[ 3.76971041 -1.61106158]
 [ 0.92500746  2.52709008]
 [ 7.36043924  2.75120823]]
'''
'''
for k in range(n_clusters):
    x, y = np.mgrid[-1:1.5:.01, -6:5:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal(mu[k], sigma[k])
    plt.contour(x, y, rv.pdf(pos))
    plt.show()
'''

'''
for k in range(n_clusters):
    X[:] = X[:] - mu[k]
    New_x = X.copy()
    New_x[:, 0] = P[:, k]*X[:, 0]
    New_x[:, 1] = P[:, k]*X[:, 1]
    sigma[k] = np.dot(New_x.T,X)/np.sum(P[:,k])
'''