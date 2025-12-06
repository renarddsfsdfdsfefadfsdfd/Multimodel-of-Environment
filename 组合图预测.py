import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 基础参数（修正单位，保存原始值）--------------------------
DEFAULT_PARAMS = {
    # Fipronil physicochemical parameters (H修正为atm·L/mol，后续转换为Pa·m³/mol)
    "H_atm_L_mol": 2.4e-3,  # Henry常数（atm·L/mol）
    "logKow": 4.0,
    "half_life": 72.0,       # 半衰期（h）
    "sum_vxy": 0.8,          # 代谢转化比例
    "mol_mass_fipronil": 252.1,  # 分子量（g/mol）
    # Fish physiological parameters
    "fish_weight": 0.5,      # 鱼体重（kg）
    "F_L": 0.05,             # 脂质体积分数
    "F_N": 0.15,             # 非脂质有机相分数
    "F_W": 0.80,             # 水相分数
    "G_W": 0.02,             # 鳃通气速率（m³/h）
    "E_W": 0.75,             # 鳃吸收效率
    "G_D": 0.005,            # 摄食速率（m³/h）
    "E_D": 0.85,             # 肠道吸收效率
    "beta": 0.85,            # 食物吸收分数
    "K_DG": 2.0,             # 肠道-鱼体分配系数
    "k_g": 0.001,            # 生长速率（1/h）
    # Environmental & food parameters
    "f_W": 1.2e-5,           # 水中逸度（Pa）
    "f_D": 3.5e-5,           # 食物中逸度（Pa）
    "food_lipid": 0.03,      # 食物脂质分数
    "food_nonlipid": 0.12,   # 食物非脂质有机相分数
    "food_water": 0.85       # 食物水相分数
}

# 单位转换：atm·L/mol → Pa·m³/mol（1 atm=101325 Pa，1 L=1e-3 m³）
def convert_H(H_atm_L_mol):
    return H_atm_L_mol * 101325 * 1e-3

# -------------------------- 核心计算函数（避免全局变量污染）--------------------------
def calculate_core_params(params):
    """计算核心参数（Z值、D值等），返回字典避免全局变量"""
    # 单位转换
    H = convert_H(params["H_atm_L_mol"])
    logKow = params["logKow"]
    half_life = params["half_life"]
    fish_weight = params["fish_weight"]
    
    # 派生参数
    K_OW = 10**logKow
    k_TR = np.log(2) / half_life if half_life > 0 else 1e-6
    V_B = fish_weight / 1000.0  # 鱼体积（m³）
    
    # Fugacity capacities (Z-values)
    Z_W = 1 / H  # 水相
    Z_O = K_OW / H  # 脂质相（辛醇替代）
    Z_N = 0.035 * K_OW / H  # 非脂质有机相
    Z_B = params["F_L"] * Z_O + params["F_N"] * Z_N + params["F_W"] * Z_W  # 鱼体
    Z_D = params["food_lipid"] * Z_O + params["food_nonlipid"] * Z_N + params["food_water"] * Z_W  # 食物
    
    # Transport/transformation parameters (D-values)
    D_W = params["G_W"] * params["E_W"] * Z_W  # 鳃吸收
    D_D = params["E_D"] * params["G_D"] * Z_D  # 摄食吸收
    D_F = D_D * (1 - params["beta"]) / params["K_DG"]  # 粪便排泄
    D_G = V_B * Z_B * params["k_g"]  # 生长稀释
    D_M = V_B * Z_B * (1 - params["sum_vxy"]) * k_TR  # 生物代谢
    
    return {
        "H": H, "K_OW": K_OW, "k_TR": k_TR, "V_B": V_B,
        "Z_W": Z_W, "Z_O": Z_O, "Z_N": Z_N, "Z_B": Z_B, "Z_D": Z_D,
        "D_W": D_W, "D_D": D_D, "D_F": D_F, "D_G": D_G, "D_M": D_M
    }

# -------------------------- RK4求解器（独立函数，无全局依赖）--------------------------
def rk4_solver(dfdt, t_start, t_end, dt, f_init):
    t_list = np.arange(t_start, t_end + dt, dt)
    f_list = np.zeros_like(t_list, dtype=np.float64)
    f_list[0] = f_init
    conv_threshold = 1e-12
    for i in range(1, len(t_list)):
        t_prev = t_list[i-1]
        f_prev = f_list[i-1]
        k1 = dt * dfdt(t_prev, f_prev)
        k2 = dt * dfdt(t_prev + dt/2, f_prev + k1/2)
        k3 = dt * dfdt(t_prev + dt/2, f_prev + k2/2)
        k4 = dt * dfdt(t_prev + dt, f_prev + k3)
        f_current = f_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
        # 避免数值震荡
        f_list[i] = f_prev if abs(f_current - f_prev) < conv_threshold else f_current
    return t_list, f_list

# -------------------------- 基准模拟（母体化合物）--------------------------
t_start, t_end, dt = 0, 30*24, 1  # 0-30天，1小时步长
core_params = calculate_core_params(DEFAULT_PARAMS)

# 定义微分方程（使用局部参数，避免全局污染）
def dfb_dt(t, f_B):
    input_flux = core_params["D_W"] * DEFAULT_PARAMS["f_W"] + core_params["D_D"] * DEFAULT_PARAMS["f_D"]
    output_flux = (core_params["D_W"] + core_params["D_F"] + core_params["D_M"] + core_params["D_G"]) * f_B
    return (input_flux - output_flux) / (core_params["Z_B"] * core_params["V_B"])

# 运行RK4求解
t_list, f_B_parent = rk4_solver(dfb_dt, t_start, t_end, dt, 0.0)
t_days = t_list / 24  # 转换为天

# 浓度计算（修正单位，ng/g湿重）
mol_mass = DEFAULT_PARAMS["mol_mass_fipronil"]
C_parent = f_B_parent * core_params["Z_B"] * mol_mass * 1e3  # 正确单位转换
steady_base = C_parent[-100:].mean()
steady_base = max(steady_base, 1e-10)  # 避免除零

# -------------------------- 1. 参数灵敏度分析（SRC方法）--------------------------
sensitive_params = ["logKow", "H_atm_L_mol", "F_L", "G_W", "f_W", "half_life", "sum_vxy", "G_D"]
var_ratio = 0.1
src_results = {}

for param in sensitive_params:
    # 复制原始参数
    params_plus = DEFAULT_PARAMS.copy()
    params_minus = DEFAULT_PARAMS.copy()
    
    # 参数+10%
    params_plus[param] *= (1 + var_ratio)
    core_plus = calculate_core_params(params_plus)
    def dfb_dt_plus(t, f_B):
        input_flux = core_plus["D_W"] * params_plus["f_W"] + core_plus["D_D"] * params_plus["f_D"]
        output_flux = (core_plus["D_W"] + core_plus["D_F"] + core_plus["D_M"] + core_plus["D_G"]) * f_B
        return (input_flux - output_flux) / (core_plus["Z_B"] * core_plus["V_B"])
    _, f_B_plus = rk4_solver(dfb_dt_plus, t_start, t_end, dt, 0.0)
    C_plus = f_B_plus * core_plus["Z_B"] * mol_mass * 1e3
    steady_plus = C_plus[-100:].mean()
    
    # 参数-10%
    params_minus[param] *= (1 - var_ratio)
    core_minus = calculate_core_params(params_minus)
    def dfb_dt_minus(t, f_B):
        input_flux = core_minus["D_W"] * params_minus["f_W"] + core_minus["D_D"] * params_minus["f_D"]
        # 修复笔误：G_G → D_G
        output_flux = (core_minus["D_W"] + core_minus["D_F"] + core_minus["D_M"] + core_minus["D_G"]) * f_B
        return (input_flux - output_flux) / (core_minus["Z_B"] * core_minus["V_B"])
    _, f_B_minus = rk4_solver(dfb_dt_minus, t_start, t_end, dt, 0.0)
    C_minus = f_B_minus * core_minus["Z_B"] * mol_mass * 1e3
    steady_minus = C_minus[-100:].mean()
    
    # 计算SRC
    delta_param = (params_plus[param] - params_minus[param]) / DEFAULT_PARAMS[param]
    delta_steady = (steady_plus - steady_minus) / steady_base
    src_results[param] = delta_steady / delta_param if delta_param != 0 else 0

# 排序SRC结果
sorted_src = sorted(src_results.items(), key=lambda x: abs(x[1]), reverse=True)
src_params = [item[0].replace("H_atm_L_mol", "H") for item in sorted_src]  # 显示简化名称
src_values = [item[1] for item in sorted_src]

# -------------------------- 2. 母体与代谢产物动态 --------------------------
# 代谢产物参数（修正单位）
metab_params = {
    "sulfone": {
        "logKow": 4.5,
        "H_atm_L_mol": 1.2e-3  # 修正为atm·L/mol
    },
    "sulfoxide": {
        "logKow": 3.8,
        "H_atm_L_mol": 1.8e-3  # 修正为atm·L/mol
    }
}

# 计算代谢产物Z_B
Z_B_sulfone = calculate_core_params({**DEFAULT_PARAMS, **metab_params["sulfone"]})["Z_B"]
Z_B_sulfoxide = calculate_core_params({**DEFAULT_PARAMS, **metab_params["sulfoxide"]})["Z_B"]

# 代谢转化速率
conv_rate_sulfone = 0.6 * DEFAULT_PARAMS["sum_vxy"] * core_params["k_TR"]
conv_rate_sulfoxide = 0.4 * DEFAULT_PARAMS["sum_vxy"] * core_params["k_TR"]

# 模拟代谢产物浓度
C_sulfone = np.zeros_like(C_parent)
C_sulfoxide = np.zeros_like(C_parent)
for i in range(1, len(t_list)):
    # 母体转化为代谢产物
    C_sulfone[i] = C_sulfone[i-1] + conv_rate_sulfone * C_parent[i-1] * dt
    C_sulfoxide[i] = C_sulfoxide[i-1] + conv_rate_sulfoxide * C_parent[i-1] * dt
    # 代谢产物降解
    C_sulfone[i] *= np.exp(-core_params["k_TR"] * dt)
    C_sulfoxide[i] *= np.exp(-core_params["k_TR"] * dt)

# -------------------------- 3. 模型校准验证 --------------------------
measured_time_days = np.array([0, 3, 7, 10, 14, 18, 22, 25, 28, 30])
measured_conc = np.array([0.1, 2.1, 5.3, 8.7, 12.4, 15.2, 16.8, 17.5, 17.3, 17.1])
predicted_conc = np.interp(measured_time_days, t_days, C_parent)

# 线性拟合（避免除零）
if np.std(measured_conc) > 1e-6 and np.std(predicted_conc) > 1e-6:
    slope, intercept = np.polyfit(measured_conc, predicted_conc, 1)
    corr_coef = np.corrcoef(measured_conc, predicted_conc)[0, 1]
    r_squared = corr_coef**2 if not np.isnan(corr_coef) else 0.0
else:
    slope, intercept = 1.0, 0.0
    r_squared = 0.0
rmse = np.sqrt(np.mean((predicted_conc - measured_conc)**2))

# -------------------------- 4. 摄入途径贡献 --------------------------
gill_flux = core_params["D_W"] * DEFAULT_PARAMS["f_W"]
dietary_flux = core_params["D_D"] * DEFAULT_PARAMS["f_D"]
total_flux = gill_flux + dietary_flux

# -------------------------- 5. 生长稀释效应 --------------------------
# 无生长参数
no_growth_params = DEFAULT_PARAMS.copy()
no_growth_params["k_g"] = 0.0
core_no_growth = calculate_core_params(no_growth_params)

def dfb_dt_no_growth(t, f_B):
    input_flux = core_no_growth["D_W"] * no_growth_params["f_W"] + core_no_growth["D_D"] * no_growth_params["f_D"]
    output_flux = (core_no_growth["D_W"] + core_no_growth["D_F"] + core_no_growth["D_M"] + core_no_growth["D_G"]) * f_B
    return (input_flux - output_flux) / (core_no_growth["Z_B"] * core_no_growth["V_B"])

_, f_B_no_growth = rk4_solver(dfb_dt_no_growth, t_start, t_end, dt, 0.0)
C_no_growth = f_B_no_growth * core_no_growth["Z_B"] * mol_mass * 1e3

# -------------------------- 绘图（确保曲线清晰可见）--------------------------
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)

# (a) 参数灵敏度分析
ax1 = fig.add_subplot(gs[0, 0])
colors = ['crimson' if x > 0 else 'steelblue' for x in src_values]
bars = ax1.barh(src_params, src_values, color=colors, alpha=0.7)
ax1.set_xlabel('Standardized Regression Coefficient (SRC)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Parameter Sensitivity Analysis', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='x', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, src_values):
    ax1.text(val + 0.01 if val > 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)

# (b) 母体与代谢产物动态
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_days, C_parent, label='Fipronil (Parent)', linewidth=2.5, color='navy', linestyle='-')
ax2.plot(t_days, C_sulfone, label='Fipronil Sulfone', linewidth=2.0, color='crimson', linestyle='--')
ax2.plot(t_days, C_sulfoxide, label='Fipronil Sulfoxide', linewidth=2.0, color='darkgreen', linestyle=':')
ax2.set_xlabel('Time (Days)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Concentration (ng/g)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Parent Compound vs. Metabolites Dynamics', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(bottom=0, top=np.max(C_parent)*1.1)

# (c) 模型校准
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(measured_conc, predicted_conc, color='darkorange', s=60, alpha=0.8, edgecolors='black')
ax3.plot([0, max(measured_conc)*1.1], [0, max(measured_conc)*1.1], linestyle='--', color='black', linewidth=1.5, label='1:1 Line')
fit_line = slope * np.array([0, max(measured_conc)*1.1]) + intercept
ax3.plot([0, max(measured_conc)*1.1], fit_line, linestyle='-', color='crimson', linewidth=2, label=f'Fit Line ($R^2={r_squared:.2f}$)')
ax3.set_xlabel('Measured Concentration (ng/g)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Concentration (ng/g)', fontsize=11, fontweight='bold')
ax3.set_title(f'(c) Model Calibration (RMSE={rmse:.2f})', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# (d) 摄入途径贡献
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(t_days, [gill_flux]*len(t_days), label='Gill Respiration', linewidth=2.5, color='steelblue', linestyle='-')
ax4.plot(t_days, [dietary_flux]*len(t_days), label='Dietary Ingestion', linewidth=2.5, color='darkorange', linestyle='-')
ax4.plot(t_days, [total_flux]*len(t_days), label='Total Uptake', linewidth=2.0, color='black', linestyle='--')
ax4.set_xlabel('Time (Days)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Uptake Flux (mol/(Pa·h))', fontsize=11, fontweight='bold')
ax4.set_title('(d) Contribution of Uptake Pathways', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# (e) 生长稀释效应
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(t_days, C_parent, label='With Growth ($k_g=0.001$ 1/h)', linewidth=2.5, color='navy', linestyle='-')
ax5.plot(t_days, C_no_growth, label='Without Growth ($k_g=0$ 1/h)', linewidth=2.5, color='crimson', linestyle='--')
ax5.set_xlabel('Time (Days)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Concentration (ng/g)', fontsize=11, fontweight='bold')
ax5.set_title('(e) Growth Dilution Effect', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_ylim(bottom=0, top=np.max(C_no_growth)*1.1)

# 隐藏空图
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

# 总标题
fig.suptitle('Fipronil Residue Dynamic Simulation in Carp: Key Analyses', fontsize=16, fontweight='bold', y=0.98)

# 保存图片
plt.savefig('Fipronil_Residue_Key_Analyses.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("图表生成成功！浓度范围：")
print(f"母体化合物稳态浓度：{steady_base:.2f} ng/g")
print(f"代谢产物（砜）峰值浓度：{np.max(C_sulfone):.2f} ng/g")
print(f"代谢产物（亚砜）峰值浓度：{np.max(C_sulfoxide):.2f} ng/g")
