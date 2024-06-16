import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import argparse
import matplotlib


def grep_flag():
    parser = argparse.ArgumentParser(
        description="plot_nep.py: A utility for plotting NEP data",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'mode',
        nargs='?',  # 
        choices=['test', 'train'],
        default='train',  # 
        help=(
            "Mode of operation (default: train):\n"
            "  test  - Run the script to plot both the NEP train and test data.\n"
            "  train - Run the script to plot the NEP train data.\n"
        )
    )
    parser.add_argument(
        '--version',
        action='version',
        version='plot_nep.py 1.0',
        help="Show program's version number and exit."
    )
    args = parser.parse_args()

    test_flag = 1 if args.mode == 'test' else 0
    print(f"mode: {args.mode}")
    print(f"test_flag = {test_flag}")
    return test_flag




##set figure properties
aw = 3
fontsize = 25
lw = 4.0
font = {'size': fontsize}
line_width = 3
dot_size = 70
number_of_histix_bins=number_of_histix_bins 
matplotlib.rc('font', **font)
matplotlib.rc('axes', lw=aw)

def handle_data(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile, delimiter=' ')
        rows = list(reader)
        
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=' ')

        for row in rows:
            try:
                value = float(row[6])
                if abs(value) < 1e5:
                    writer.writerow(row)
            except ValueError:
                continue

def set_fig_properties(ax_list):
    tl = 7
    tw = 3
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='out', right=False, top=False)

def plot_nep(pout):
    nep = np.loadtxt("./nep.txt", skiprows=7)
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.hist(np.log(np.abs(nep)), bins=100)
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(nep)), nep, s=0.5)
    plt.gcf().set_size_inches(9, 3)
    plt.savefig(pout, dpi=300)

def com_RMSE(fin):
    nclo = int(fin.shape[1] / 2)
    pids = fin[:, nclo] > -1e5
    targe = fin[pids, :nclo].reshape(-1)
    predi = fin[pids, nclo:].reshape(-1)
    return np.sqrt(((predi - targe) ** 2).mean())

def com_MAE(fin):
    nclo = int(fin.shape[1] / 2)
    pids = fin[:, nclo] > -1e5
    targe = fin[pids, :nclo].reshape(-1)
    predi = fin[pids, nclo:].reshape(-1)
    return np.mean(np.abs(predi - targe))

def com_R2(fin):
    nclo = int(fin.shape[1] / 2)
    pids = fin[:, nclo] > -1e5
    targe = fin[pids, :nclo].reshape(-1)
    predi = fin[pids, nclo:].reshape(-1)
    tss = np.sum((targe - np.mean(targe)) ** 2)
    rss = np.sum((predi - targe) ** 2)
    r2 = 1 - (rss / tss)
    return r2


def read_data(file_path):
    try:
        data = []
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    numbers = [float(value) for value in line.split() if float(value) > -1e5]
                    data.extend(numbers)
                except ValueError:
                    print(f"无法转换第 {line_number} 行的值: {line.strip()}")

        if not data:
            print("文件中没有有效数据。")
            return []

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return []
    except Exception as e:
        print(f"发生错误：{e}")
        return []
    return data

##get test flag
if __name__ == "__main__":
    test_flag = grep_flag()

    ##nep train data
    e_train = read_data('energy_train.out')
    f_train = read_data('force_train.out')
    v_train = read_data('virial_train.out')
    s_train = read_data('stress_train.out')

    ##fig 1
    loss = np.loadtxt('loss.out')
    loss[:, 0] = np.arange(1, len(loss) + 1) * 100
    print(f"We have run {loss[-1, 0]} steps!")


    if loss[-1, 7] == 0:
        test_flag = 0

    if test_flag == 1:
        e_test = read_data('energy_test.out')
        f_test = read_data('force_test.out')
        v_test = read_data('virial_test.out')
        s_test = read_data('stress_test.out')


    energy_train = np.loadtxt('energy_train.out')

    ##fig 3
    force_train = np.loadtxt('force_train.out')

    ##fig 4
    virial_train = np.loadtxt('virial_train.out')
    stress_train = np.loadtxt('stress_train.out')

    rmse_ener = com_RMSE(energy_train) * 1000
    rmse_force = com_RMSE(force_train) * 1000
    rmse_virial = com_RMSE(virial_train) * 1000
    rmse_stress = com_RMSE(stress_train) * 1000
    mae_ener = com_MAE(energy_train) * 1000
    mae_force = com_MAE(force_train) * 1000
    mae_virial = com_MAE(virial_train) * 1000
    mae_stress = com_MAE(stress_train) * 1000
    r2_ener = com_R2(energy_train) 
    r2_force = com_R2(force_train) 
    r2_virial = com_R2(virial_train) 
    r2_stress = com_R2(stress_train) 
    if test_flag == 1:
        energy_test = np.loadtxt('energy_test.out')
        force_test = np.loadtxt('force_test.out')
        virial_test = np.loadtxt('virial_test.out')
        stress_test = np.loadtxt('stress_test.out')
####################READ TEST DATA##########################
        rmse_ener_test = com_RMSE(energy_test) * 1000
        rmse_force_test = com_RMSE(force_test) * 1000
        rmse_virial_test = com_RMSE(virial_test) * 1000
        rmse_stress_test = com_RMSE(stress_test) * 1000
        mae_ener_test = com_MAE(energy_test) * 1000
        mae_force_test = com_MAE(force_test) * 1000
        mae_virial_test = com_MAE(virial_test) * 1000
        mae_stress_test = com_MAE(stress_test) * 1000
        r2_ener_test = com_R2(energy_test) 
        r2_force_test = com_R2(force_test) 
        r2_virial_test = com_R2(virial_test) 
        r2_stress_test = com_R2(stress_test) 



    # Set the fig
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    divider = [make_axes_locatable(ax) for ax in axes.flatten()]

    plt.subplot(2, 2, 1)
    set_fig_properties([plt.gca()])
    plt.loglog(loss[:, 0], loss[:, 1], ls="-", lw=lw, c="C1", label="Total")
    plt.loglog(loss[:, 0], loss[:, 2],  ls="-", lw=lw, c = "C4", label=r"$L_{1}$")
    plt.loglog(loss[:, 0], loss[:, 3], ls="-", lw=lw, c="C5", label=r"$L_{2}$")
    plt.loglog(loss[:, 0], loss[:, 4], ls="-", lw=lw, c="C0", label="Energy_train")
    plt.loglog(loss[:, 0], loss[:, 5], ls="-", lw=lw, c="C2", label="Force_train")
    plt.loglog(loss[:, 0], loss[:, 6], ls="-", lw=lw, c="C3", label="Virial_train")

    if test_flag == 1:
        plt.loglog(loss[:, 0], loss[:, 7], ls="--", lw=lw, c="C6", label="Energy_test")
        plt.loglog(loss[:, 0], loss[:, 8], ls="--", lw=lw, c="C7", label="Force_test")
        plt.loglog(loss[:, 0], loss[:, 9], ls="--", lw=lw, c="C8", label="Virial_test")

    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc="lower left", ncol=2, frameon=False, columnspacing=0.2)

    ax_scatter = axes[0, 1]
    set_fig_properties([ax_scatter])
    ax_histx = divider[1].append_axes("top", 1.2, pad=0, sharex=ax_scatter)
    ene_min = np.min([np.min(energy_train), np.min(energy_test)]) if test_flag == 1 else np.min(energy_train)
    ene_max = np.max([np.max(energy_train), np.max(energy_test)]) if test_flag == 1 else np.max(energy_train)
    ene_min -= (ene_max - ene_min) * 0.1
    ene_max += (ene_max - ene_min) * 0.1

    ax_scatter.plot([ene_min, ene_max], [ene_min, ene_max], c="grey", lw=2, zorder=1)
    ax_scatter.scatter(energy_train[:, 1], energy_train[:, 0], marker='o', c="C0", s=dot_size, alpha=0.5,
                       label=f"Train dataset (RMSE={rmse_ener:.2f} meV/atom,MAE={mae_ener:.2f} meV/atom)", zorder=2)
    if test_flag == 1:
        ax_scatter.scatter(energy_test[:, 1], energy_test[:, 0], marker='o', c="C6", s=dot_size, alpha=0.5,
                       label=f"Test dataset (RMSE={rmse_ener_test:.2f} meV/atom,MAE={mae_ener_test:.2f} meV/atom)", zorder=2)
        ax_histx.hist(e_test, bins=number_of_histix_bins, color='C6')
    ax_histx.hist(e_train, bins=number_of_histix_bins, color='C0')

    ax_histx.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_histx.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_scatter.set_xlabel('DFT energy (eV/atom)')
    ax_scatter.set_ylabel('NEP energy (eV/atom)')
    ax_scatter.legend(loc="upper left")
    plt.setp(ax_histx.get_xticklabels(), visible=False)

    # Force
    ax_scatter = axes[1, 0]
    ax_histx = divider[2].append_axes("top", 1.2, pad=0, sharex=ax_scatter)
    set_fig_properties([ax_scatter])
    for_min = np.min([np.min(force_train), np.min(force_test)]) if test_flag == 1 else np.min(force_train)
    for_max = np.max([np.max(force_train), np.max(force_test)]) if test_flag == 1 else np.max(force_train)
    for_min -= (for_max - for_min) * 0.1
    for_max += (for_max - for_min) * 0.1

    ax_scatter.plot([for_min, for_max], [for_min, for_max], c="grey", lw=line_width, zorder=1)
    ax_scatter.scatter(force_train[:, 3], force_train[:, 0], marker='o', c="C2", s=dot_size, alpha=0.5,
                       label=f"Train dataset (RMSE={rmse_force:.2f} meV/${{\\rm{{\AA}}}}$)")
    ax_histx.hist(f_train, bins=number_of_histix_bins, color='C2')
    if test_flag == 1:
        ax_scatter.scatter(force_test[:, 3], force_test[:, 0], marker='o', c="C7", s=dot_size, alpha=0.5,
                       label=f"Test dataset (RMSE={rmse_force_test:.2f} meV/${{\\rm{{\AA}}}}$,MAE={mae_force_test:.2f} meV/${{\\rm{{\AA}}}}$)", zorder=2)
        ax_histx.hist(f_test, bins=number_of_histix_bins, color='C7')

    ax_histx.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_histx.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_scatter.set_xlabel(r'DFT force (eV/$\rm{\AA}$)')
    ax_scatter.set_ylabel(r'NEP force (eV/$\rm{\AA}$)')
    ax_scatter.legend(loc="upper left")
    plt.setp(ax_histx.get_xticklabels(), visible=False)

    # Stress
    ax_scatter = axes[1, 1]
    set_fig_properties([ax_scatter])
    ptra = stress_train[:, -1] > -1e-5
    ptes = stress_test[:, -1] > -1e-5 if test_flag == 1 else ptra
    vir_min = np.min([np.min(stress_train[ptra, :]), np.min(stress_test[ptes, :])]) if test_flag == 1 else np.min(
        stress_train[ptra, :])
    vir_max = np.max([np.max(stress_train[ptra, :]), np.max(stress_test[ptes, :])]) if test_flag == 1 else np.max(
        stress_train[ptra, :])
    vir_min -= (vir_max - vir_min) * 0.1
    vir_max += (vir_max - vir_min) * 0.1
    ax_histx = divider[3].append_axes("top", 1.2, pad=0, sharex=ax_scatter)
    ax_scatter.plot([vir_min, vir_max], [vir_min, vir_max], c="grey", lw=line_width, zorder=1)

    if stress_train.shape[1] == 2:
        ax_scatter.scatter(stress_train[ptra, 1], stress_train[ptra, 0], marker='o', c='C3', s=dot_size, alpha=0.5,
                           label=f"Train dataset:\n(RMSE={rmse_force:.2f} meV/${{\\rm{{\AA}}}}$;\n MAE={mae_force:.2f} meV/${{\\rm{{\AA}}}}$; \n R$^2$={r2_force:.3f})")
        ax_histx.hist(stress_train[ptra, 1], bins=number_of_histix_bins, color='C3')
        
        if test_flag == 1:
            ax_scatter.scatter(stress_test[ptes, 1], stress_test[ptes, 0], marker='o', c="C8", s=dot_size, alpha=0.5,
                       label=f"Test dataset (RMSE={rmse_virial_test:.2f} MPa,MAE={mae_virial_test:.2f} MPa)", zorder=2)
            ax_histx.hist(v_test, bins=number_of_histix_bins, color='C8')

    elif stress_train.shape[1] == 12:
        ax_scatter.scatter(stress_train[ptra, 7:12], stress_train[ptra, 1:6], marker='o', c='C3', s=dot_size, alpha=0.5,
                           label=f"Train dataset:\n(RMSE={rmse_virial:.2f} MPa;\n MAE={mae_virial:.2f} MPa;\n R$^2$={r2_virial:.3f})")
        ax_histx.hist(stress_train[ptra, 1], bins=number_of_histix_bins, color='C3')
        if test_flag == 1:
            ax_scatter.scatter(stress_test[ptes, 7:12], stress_test[ptes, 1:6], marker='o', c="C8", s=dot_size, alpha=0.5,
                       label=f"Test dataset (RMSE={rmse_virial_test:.2f} MPa,MAE={mae_virial_test:.2f} MPa)", zorder=2)
            ax_histx.hist(v_test, bins=number_of_histix_bins, color='C8')

    ax_histx.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_histx.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_scatter.set_xlabel('DFT stress (GPa)')
    ax_scatter.set_ylabel('NEP stress (GPa)')
    ax_scatter.legend(loc="upper left")
    plt.setp(ax_histx.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()
    plt.savefig('RMSE.png')

