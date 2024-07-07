import numpy as np 
import pickle
import torch
from density_estimator import *
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

from hsbi import HSBI
from matplotlib import pyplot as plt
from tqdm import tqdm





with open("/home/ge84yes/data/posteriors/rate_e_rate_i_f_w-blow.pkl", "rb") as f:
    posterior = pickle.load(f)


print(posterior)
metrics_list = {'rate_e' : (1,50),
                       'rate_i' : (1,50),
                       'cv_isi' : (0.7, 1000),
                       'f_w-blow' : (0, 0.1),
                       'w_creep' : (0.0, 0.05),
                       'wmean_ee' : (0.0, 0.5),
                       'wmean_ie' : (0.0, 5.0),
                       'mean_fano_t' : (0.5, 2.5),
                       'mean_fano_s' : (0.5, 2.5), 
                       'auto_cov' : (0.0, 0.1),

                       'std_fr' : (0, 0.5),
                       "std_rate_spatial" : (0, 5)
                       }
hsbi = HSBI()
bounds = {'low' : torch.tensor(hsbi.prior_lower_bound), 'high' : torch.tensor(hsbi.prior_upper_bound)}
low = torch.tensor([metrics_list['rate_e'][0], metrics_list['rate_i'][0], metrics_list['f_w-blow'][0]])
print(low)
high = torch.tensor([metrics_list['rate_e'][1], metrics_list['rate_i'][1], metrics_list['f_w-blow'][1]])
print(high)
prior = BoxUniform(low=low, high=high)
total = np.empty((0,10))
batches = 200
for b in tqdm(range(batches)):
    obs = prior.sample((100,))
    b = posterior.rsample(obs, bounds).detach().numpy()
    total = np.vstack((total, b))
    


# we calculate the B values 
#limits = [(-0.1, 0.1), (-0.1, 0.1)]
limits = [(-1.0, 1.0), (-1.0, 1.0), (0.5,2.0), (0.005, 0.030), (0.005, 0.030)] * 2
limits = torch.tensor(limits)
B_EE = np.expand_dims(total[:,2] * total[:,3] - total[:,4], axis=0)
B_IE = np.expand_dims(total[:,7] * total[:,8] - total[:,9], axis=0)


samples = np.concatenate((B_EE, B_IE), axis=0)
B_EE = B_EE[0,:]
B_IE = B_IE[0,:]
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
"""
fig, axes = pairplot(
    samples=total,
    limits=limits,
    offdiag=["kde"],
    diag=["kde"],
    figsize=(5, 5),
)

plt.savefig('test.png')
plt.clf()




plt.figure(figsize=(10,6))
kdeplot = sns.kdeplot(x=B_EE, y=B_IE, cmap='viridis',fill=True)
plt.xlabel(r"$B_{EE}$")
plt.ylabel(r"$B_{IE}$")
plt.colorbar(kdeplot.collections[0], label='Density')
plt.savefig("test2.png")
"""



###########################################
#
###########################################
xmin, xmax = B_EE.min(), B_EE.max()
ymin, ymax = B_IE.min(), B_IE.max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([B_EE, B_IE])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('Conditional Density Plot')
plt.xlabel(r'$B_{EE}$')
plt.ylabel(r'$B_{IE}$')
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_b.png")
plt.clf()

sns.set_theme(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.kdeplot(total[:,0], color='blue', fill=False, label=r"$\alpha_{pre,EE}$")
sns.kdeplot(total[:,1], color='red', fill=False, label=r"$\alpha_{post,EE}$")
sns.kdeplot(total[:,5], color='yellow', fill=False, label=r"$\alpha_{pre,IE}$")
sns.kdeplot(total[:,6], color='green', fill=False, label=r"$\alpha_{post,IE}$")

plt.legend()
plt.savefig("marginal_alpha.png")

plt.clf()



###########################################
#
###########################################
xmin, xmax = total[:,0].min(), total[:,0].max()
ymin, ymax = total[:,1].min(), total[:,1].max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,0], total[:,1]])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Pre  / EE-Post')
plt.xlabel(r"$\alpha_{pre,EE}$")
plt.ylabel(r"$\alpha_{post,EE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_ee_pre_ee_post.png")
plt.clf()



###########################################
#
###########################################
xmin, xmax = total[:,0].min(), total[:,0].max()
ymin, ymax = total[:,5].min(), total[:,5].max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,0], total[:,5]])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Pre  / IE-Pre')
plt.xlabel(r"$\alpha_{pre,EE}$")
plt.ylabel(r"$\alpha_{pre,IE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_ee_pre_ie_pre.png")
plt.clf()




###########################################
#
###########################################
xmin, xmax = total[:,0].min(), total[:,0].max()
ymin, ymax = total[:,6].min(), total[:,6].max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,0], total[:,6]])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Pre / IE-Post')
plt.xlabel(r"$\alpha_{pre,EE}$")
plt.ylabel(r"$\alpha_{post,IE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_ee_pre__ie_post.png")
plt.clf()




###########################################
#
###########################################
xmin, xmax = total[:,1].min(), total[:,1].max()
ymin, ymax = total[:,5].min(), total[:,5].max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,1], total[:,5]])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Post / IE-Pre')
plt.xlabel(r"$\alpha_{post,EE}$")
plt.ylabel(r"$\alpha_{pre,IE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_ee_post_ie_pre.png")
plt.clf()




###########################################
#
###########################################
xmin, xmax = total[:,1].min(), total[:,1].max()
ymin, ymax = total[:,6].min(), total[:,6].max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,1], total[:,6]])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Post / IE-Post')
plt.xlabel(r"$\alpha_{post,EE}$")
plt.ylabel(r"$\alpha_{post,IE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_ee_post_ie_post.png")
plt.clf()



###########################################
#
###########################################
xmin, xmax = total[:,5].min(), total[:,5].max()
ymin, ymax = total[:,6].min(), total[:,6].max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,5], total[:,6]])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('IE-Pre  / IE-Post')
plt.xlabel(r"$\alpha_{pre,IE}$")
plt.ylabel(r"$\alpha_{post,IE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_ie_pre_ie_post.png")
plt.clf()




###########################################
#
###########################################
xmin, xmax = total[:,0].min() + total[:,1].min(), total[:,0].max() + total[:,1].max()
ymin, ymax = B_EE.min(), B_EE.max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,0] + total[:,1], B_EE])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Pre + EE-Post  / B_EE')
plt.xlabel(r"$\alpha_{pre,EE} + \alpha_{post,EE}$")
plt.ylabel(r"$B_{EE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_b_alpha_ee_combined.png")
plt.clf()




###########################################
#
###########################################
xmin, xmax = total[:,0].min(), total[:,0].max() 
ymin, ymax = B_EE.min(), B_EE.max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,0], B_EE])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Pre  / B_EE')
plt.xlabel(r"$\alpha_{pre,EE}$")
plt.ylabel(r"$B_{EE}$")
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_b_alpha_ee_pre.png")
plt.clf()




###########################################
#
###########################################
xmin, xmax = total[:,1].min(), total[:,1].max() 
ymin, ymax = B_EE.min(), B_EE.max()
grid_points=200
x, y = np.mgrid[xmin:xmax:complex(grid_points), ymin:ymax:complex(grid_points)]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([total[:,1], B_EE])

# Calculate the KDE
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, x.shape)

# Normalize the KDE to sum to 1
z /= z.sum()

# Create the 2D KDE plot
plt.figure(figsize=(10, 6))
kdeplot = plt.contourf(x, y, z, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(kdeplot)
# Add titles and labels
plt.title('EE-Post  / B_EE')
plt.xlabel(r"$\alpha_{post,EE}$")
plt.ylabel(r"$B_{EE}$")
# Add vertical and horizontal lines to represent the axes
plt.axvline(x=0, color='white', linestyle='--', linewidth=2)  # Vertical line at x=0
plt.axhline(y=0, color='white', linestyle='--', linewidth=2)  # Horizontal line at y=0
plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)
# Show the plot
plt.savefig("conditional_b_alpha_ee_post.png")
plt.clf()
