import numpy as np 
import pickle
import torch
from density_estimator import *
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

from hsbi import HSBI
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
import matplotlib.patches as patches
with open("/home/ge84yes/data/run_1/posteriors/rate_e_rate_i_f_w-blow.pkl", "rb") as f:
    posterior = pickle.load(f)

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


total_thetas = torch.empty((0,10))
run = 0

obs = prior.sample((500,))
try:
    thetas = posterior.rsample(obs, bounds).detach().numpy()
    total_thetas = torch.concat((total_thetas, thetas))
except Exception:
    run += 1



def kernel(Aplus, t_plus, t_minus, Aminus=-1.0, tp=np.linspace(start=0, stop=1, num=10000)):
    v_p = Aplus * np.exp(- tp / t_plus) + Aminus * np.exp(-tp / t_minus)
    tp_n = - tp 
    v_n = Aplus * np.exp(tp_n / t_plus) + Aminus * np.exp(tp_n / t_minus)

   
    #return np.concatenate((tp_n, tp)), np.concatenate((v_n, v_p))
    return (tp_n, v_n), (tp, v_p)
"""
print("EE")
print((thetas[0,2], thetas[0,3], thetas[0,4]))
(xn1, yn1), (xp1, yp1) = kernel(thetas[0,2], thetas[0,3], thetas[0,4])
#print(simpson(y=yp1, x=xp1))

B_EE = (thetas[0,2] * (thetas[0,3])) - (thetas[0,4])
#print(B_EE)
# Second set of data
print("IE")
print((thetas[0,7], thetas[0,8], thetas[0,9]))
(xn2, yn2), (xp2, yp2) = kernel(thetas[0,7], thetas[0,8], thetas[0,9])
#print(simpson(y=yn2, x=xn2 / 1000))
B_IE = (thetas[0,7] * (thetas[0,8])) - (thetas[0,9])
#print(B_IE)
# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first set of data
print(yn1[0])
axes[0].plot(xn1, yn1, "r")
axes[0].plot(xp1, yp1, "r")
axes[0].set_xlabel(r"$\Delta t$ in ms")
axes[0].set_xlim([-0.25, 0.25])
axes[0].grid()
axes[0].set_title(f"Ex-Ex Kernel (B={B_EE})")

# Plot the second set of data
axes[1].plot(xn2, yn2, "r")
axes[1].plot(xp2, yp2, "r")
axes[1].set_xlabel(r"$\Delta t$ in ms")
axes[1].grid()
axes[1].set_xlim([-0.25, 0.25])
axes[1].set_title(f"Inh-Ex Kernel (B={B_IE})")

# Save the figure with both plots
plt.savefig("kernel_combined.png")

plt.clf()

plt.plot(xn1, yn1, "r")
plt.plot(xp1, yp1, "r")
plt.title("")
plt.savefig("ee.png")
plt.clf()
"""
# Lists to store coordinates and file paths
coordinates = []
file_paths = []

for i in range(1000):  # Assuming 5 sets of plots to be generated
    print("EE")
    (xn1, yn1), (xp1, yp1) = kernel(thetas[i,2], thetas[i,3], thetas[i,4])
    B_EE = (thetas[i,2] * (thetas[i,3])) - (thetas[i,4])

    print("IE")
    (xn2, yn2), (xp2, yp2) = kernel(thetas[i,7], thetas[i,8], thetas[i,9])
    B_IE = (thetas[i,7] * (thetas[i,8])) - (thetas[i,9])

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the first set of data
    axes.plot(xn1, yn1, "r", linewidth=12)
    axes.plot(xp1, yp1, "r", linewidth=12)
    #axes[0].set_xlabel(r"$\Delta t$ in ms")
    axes.set_xlim([-0.1, 0.1])
   
    axes.axis('off')
    axes.set_facecolor('none')
    #axes[0].grid()
    #axes[0].set_title(f"Ex-Ex Kernel (B={B_EE})")

    # Plot the second set of data
    axes.plot(xn2, yn2, "b", linewidth=12)
    axes.plot(xp2, yp2, "b", linewidth=12)
    #axes[1].set_xlabel(r"$\Delta t$ in ms")
    #axes[1].grid()
    #axes[1].set_xlim([-0.25, 0.25])
    #axes[1].axis('off')
    #axes[1].set_facecolor('none')
    #axes[1].set_title(f"Inh-Ex Kernel (B={B_IE})")
    fig.patch.set_alpha(0)
    # Save the figure with both plots
    file_path = f"kernel_combined_{i}.svg"
    plt.savefig(file_path, format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    # Store the coordinates and file path
    coordinates.append((B_EE, B_IE))
    file_paths.append(file_path)

# Create a larger plot to embed all individual plots
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter([coord[0] for coord in coordinates], [coord[1] for coord in coordinates], marker='o')

def check_overlap(new_coords, existing_coords, margin=0.008):
    for (x, y) in existing_coords:
        if abs(new_coords[0] - x) < margin and abs(new_coords[1] - y) < margin:
            return True
    return False

def svg_to_png(svg_file):
    from cairosvg import svg2png
    png_data = svg2png(url=svg_file)
    image = Image.open(BytesIO(png_data))
    return image

# Create a larger plot to embed all individual plots
fig, ax = plt.subplots(figsize=(10, 10))
# Color the quadrants
ax.fill_betweenx(y=[-0, 0.1], x1=0, x2=0.1, color='lightgreen', alpha=0.3)  # Top-right quadrant
ax.fill_betweenx(y=[-0.1, 0.1], x1=-0.1, x2=0, color='lightyellow', alpha=0.3)  # Top-left quadrant
ax.fill_betweenx(y=[-0.1, 0], x1=0, x2=0.1, color='lightyellow', alpha=0.3)  # Bottom-right quadrant
ax.fill_betweenx(y=[-0.1, 0], x1=-0.1, x2=0, color='lightcoral', alpha=0.3)  # Bottom-left quadrant
existing_coords = []

for (B_EE, B_IE), file_path in zip(coordinates, file_paths):
    if not check_overlap((B_EE, B_IE), existing_coords):
        # Load the saved plot
        img = svg_to_png(file_path)
        # Display the image at the corresponding coordinates
        imagebox = OffsetImage(img, zoom=0.08)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (B_EE, B_IE), frameon=False)
        ax.add_artist(ab)
        existing_coords.append((B_EE, B_IE))
    else:
        print(f"Skipping plot at ({B_EE}, {B_IE}) due to overlap")

ax.set_xlabel(r'$B_{EE}$')
ax.set_ylabel(r'$B_{IE}$')
ax.set_xlim([-0.034, 0.065])
ax.set_ylim([-0.035, 0.065])
ax.set_title('')
red_patch = patches.Patch(color='red', label='Exitatory')
blue_patch = patches.Patch(color='blue', label='Inhibitory')
ax.legend(handles=[red_patch, blue_patch], loc='upper right')

plt.savefig("embedded_plots.svg", format="svg")
plt.savefig("embedded_plots.png")
plt.xlim([-1, 1])
plt.show()