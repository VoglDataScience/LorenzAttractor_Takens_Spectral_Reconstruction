# -*- coding: utf-8 -*-

"""
Author: Markus Vogl
Date of last change: 14.04.2022
Purpose: Demonstation for YouTube Video
Description Short: This script generates attractor reconstructions out of Lorenz system.
"""

#Import packages:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding


#Generate a Lorenz System
def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    x, y, z: Points
    s, r, b: Parameters defining the Lorenz attractor
    
    x_dot, y_dot, z_dot: Values of the Lorenz attractor's partial derivatives 
    at the point x, y, z
    '''

    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return x_dot, y_dot, z_dot


#Set other parameters
dt = 0.01
num_steps = 10000

# Initial values require one or more
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Initial values setting
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and estimate the next point
for i in range(num_steps):

    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

""""
# Plot the Strange Attractor
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5, color="navy")
ax.scatter(xs, ys, zs, lw=0.1,alpha=0.1 ,color="red")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor for s=10, r=28, b=2.667")

#plt.show()
"""


# Save the coordinates to different variables for convience
tl_0 = xs
tl_1 = ys
tl_2 = zs

# Select which variable to reconstruct
t= tl_2

# Takens Delay-Time Embedding

#Select Tau Value
k = 20
l = int(k/2)

#Comb
t_0 = t[:-k]
t_1 = t[l:-l]
t_2 = t[k:]

t_total = []

#Saving of delay-time coordinates
for i in range(0,len(t_0)):
    t = [t_0[i],t_1[i],t_2[i]]
    t_total.append(t)

t_total = np.array(t_total)



#Plot Takens Delay Time Embedding
#fig = plt.figure()
#ax = fig.add_subplot(111, projection="3d")
#c_map = plt.get_cmap("seismic")
#t_c = t

#ax.scatter(xs=t_0, ys=t_1, zs=t_2, marker="+", c= t_0, cmap=c_map, vmin=min(t), vmax=max(t), s=1.5)
#ax.plot(xs=t_0, ys=t_1, zs=t_2, linewidth=0.25, color="navy", alpha=0.85)

text_title = "Taken´s Embedding Approach with tau= " + str(k)
text_ylabel = "X(t+"+ str(k) +")"
text_zlabel = "X(t+"+ str(2*k) +")"

#plt.title(text_title)

#ax.set_xlabel("X(t)")
#ax.set_ylabel(text_ylabel)
#ax.set_zlabel(text_zlabel)

#plt.show()



# Spectral Embedding:

#Select PCA components (normally embedding dimension as minimum)
#Select number of k-nearest neighbour algorithm. My empirical experiments for PhD thesis came up with heuristic 1% len(data) + 1.5*Tau (Tau == delay-time)
#Note for Lorenz system the neighbours are not as relevant as a time-series.
n_components = 3
n_neighbors = 20
t_affinity = "nearest_neighbors"

""""
# If selected, whole attractor will be reconstructed
t_total = []

for i in range(0,len(t_0)):
    t = [tl_0[i],tl_1[i],tl_2[i]]
    t_total.append(t)

t_total = np.array(t_total)
"""

# Fit the attractor
t_se = SpectralEmbedding(n_components=n_components, affinity= t_affinity, n_neighbors=n_neighbors, n_jobs=-1)
t_se_data = t_se.fit_transform(X=t_total)

#Plot Spectral Embedding
fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection="3d")
c_map = plt.get_cmap("RdYlGn")

ax.scatter(xs=t_se_data[:, 0], ys=t_se_data[:, 1], zs=t_se_data[:,2], marker="+", c= t_0, cmap=c_map, vmin=min(t), vmax=max(t), s=4)
ax.plot(xs=t_se_data[:, 0], ys=t_se_data[:, 1], zs=t_se_data[:,2],linewidth=0.2, alpha=0.85, color="navy")

text_title = "Scatter of Spectral Embedding with " +str(n_components) + " components and " +str(n_neighbors) + " neighbors with algorithm: "+str(t_affinity)
plt.title(text_title)
ax.set_xlabel("X(t)")
ax.set_ylabel(text_ylabel)
ax.set_zlabel(text_zlabel)
#ax.set_axis_off()

plt.show()
