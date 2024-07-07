from utils import collect_simulations
import torch


metrics=["rate_e", "rate_i", "f_w-blow"]
metrics_bound = {'rate_e': [5,50], 'rate_i': [5,50], 'f_w-blow' : [0,0.1]} 
thetas, obs = collect_simulations(metrics=metrics, metrics_bounds=metrics_bound, h5_path="/home/ge84yes/data/run_1/simulations.hdf5")

# Filtering condition
condition = (torch.abs(thetas[:, 0] + thetas[:, 1]) < 0.1) & (torch.abs(thetas[:, 5] + thetas[:, 6]) < 0.1)

# Apply the condition to filter thetas and obs
thetas = thetas[condition]
obs = obs[condition]


thetas = thetas.detach().numpy()
parameter_output = "/home/ge84yes/data/run_1/eval_parameters.txt"
with open(parameter_output, "w") as f:
    lines = []
    for i, th in enumerate(thetas):
        lines = []               
        i += 1
        string_parm_list = ['{:.10f}'.format(x) for x in th]
        parm_line = " ".join(string_parm_list)
        parm_line = f"{i} " + parm_line
        parm_line += "\n"
        lines.append(parm_line)
        f.writelines(lines)
