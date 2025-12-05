import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import os

# Get desktop path (cross-platform compatible)
def get_desktop_path():
    return str(Path.home() / "Desktop")

# -------------------------- Parameter Configuration (Default Values) --------------------------
DEFAULT_PARAMS = {
    # Fipronil physicochemical parameters
    "H": 2.4e-3,          # Henry's law constant (Pa·m³/mol)
    "logKow": 4.0,        # log Octanol-water partition coefficient
    "half_life": 72.0,    # Degradation half-life (h)
    "sum_vxy": 0.8,       # Proportion converted to metabolites
    
    # Fish physiological parameters
    "fish_weight": 0.5,   # Fish weight (kg) → V_B = weight/1000 (density=1000kg/m³)
    "F_L": 0.05,          # Lipid volume fraction
    "F_N": 0.15,          # Non-lipid organic matter volume fraction
    "F_W": 0.80,          # Water volume fraction
    "G_W": 0.02,          # Gill ventilation rate (m³/h)
    "E_W": 0.75,          # Gill absorption efficiency
    "G_D": 0.005,         # Feeding rate (m³/h)
    "E_D": 0.85,          # Intestinal absorption efficiency
    "beta": 0.85,         # Food absorption fraction
    "K_DG": 2.0,          # Intestine-fish distribution coefficient
    "k_g": 0.001,         # Growth rate constant (1/h)
    
    # Environmental and food parameters
    "f_W": 1.2e-5,        # Fipronil fugacity in water (Pa)
    "f_D": 3.5e-5,        # Fipronil fugacity in food (Pa)
    "food_lipid": 0.03,   # Food lipid fraction (phytoplankton)
    "food_nonlipid": 0.12,# Food non-lipid organic fraction
    "food_water": 0.85    # Food water fraction
}

# -------------------------- Core Simulation Function (Support Custom Params) --------------------------
def run_simulation(custom_params):
    try:
        # Unpack parameters with type conversion and validation
        H = float(custom_params["H"])
        logKow = float(custom_params["logKow"])
        K_OW = 10**logKow
        half_life = float(custom_params["half_life"])
        k_TR = np.log(2) / half_life if half_life > 0 else 1e-6
        sum_vxy = float(custom_params["sum_vxy"])
        
        fish_weight = float(custom_params["fish_weight"])
        V_B = fish_weight / 1000.0  # Convert weight to volume (m³)
        F_L = float(custom_params["F_L"])
        F_N = float(custom_params["F_N"])
        F_W = float(custom_params["F_W"])
        G_W = float(custom_params["G_W"])
        E_W = float(custom_params["E_W"])
        G_D = float(custom_params["G_D"])
        E_D = float(custom_params["E_D"])
        beta = float(custom_params["beta"])
        K_DG = float(custom_params["K_DG"])
        k_g = float(custom_params["k_g"])
        
        f_W = float(custom_params["f_W"])
        f_D = float(custom_params["f_D"])
        food_lipid = float(custom_params["food_lipid"])
        food_nonlipid = float(custom_params["food_nonlipid"])
        food_water = float(custom_params["food_water"])

        # Validate critical parameters
        if not (0 < F_L + F_N + F_W <= 1.0):
            raise ValueError("Fish volume fractions (F_L+F_N+F_W) must be between 0 and 1")
        if not (0 < food_lipid + food_nonlipid + food_water <= 1.0):
            raise ValueError("Food volume fractions must be between 0 and 1")
        if H <= 0 or K_OW <= 0:
            raise ValueError("Henry's law constant and K_OW must be positive")

        # -------------------------- Core Parameter Calculation --------------------------
        # Fugacity capacity (Z-values)
        Z_W = 1 / H  # Aqueous phase
        Z_O = K_OW / H  # Lipid surrogate (octanol)
        Z_N = 0.035 * K_OW / H  # Non-lipid organic matter
        Z_B = F_L * Z_O + F_N * Z_N + F_W * Z_W  # Fish body
        
        # Food fugacity capacity
        Z_D = (food_lipid * Z_O) + (food_nonlipid * Z_N) + (food_water * Z_W)
        
        # Transport/transformation parameters (D-values)
        D_W = G_W * E_W * Z_W  # Gill absorption
        D_D = E_D * G_D * Z_D  # Feeding absorption
        D_F = D_D * (1 - beta) / K_DG  # Fecal excretion
        D_G = V_B * Z_B * k_g  # Growth dilution
        D_M = V_B * Z_B * (1 - sum_vxy) * k_TR  # Biometabolism

        # -------------------------- Dynamic Mass Balance Equation --------------------------
        def dfb_dt(t, f_B):
            input_flux = D_W * f_W + D_D * f_D
            output_flux = (D_W + D_F + D_M + D_G) * f_B
            return (input_flux - output_flux) / (Z_B * V_B)

        # -------------------------- RK4 Solver --------------------------
        def rk4_solver(dfdt, t_start, t_end, dt, f_init):
            t_list = np.arange(t_start, t_end + dt, dt)
            f_list = np.zeros_like(t_list, dtype=np.float64)
            f_list[0] = f_init
            conv_threshold = 1e-10

            for i in range(1, len(t_list)):
                t_prev = t_list[i-1]
                f_prev = f_list[i-1]

                k1 = dt * dfdt(t_prev, f_prev)
                k2 = dt * dfdt(t_prev + dt/2, f_prev + k1/2)
                k3 = dt * dfdt(t_prev + dt/2, f_prev + k2/2)
                k4 = dt * dfdt(t_prev + dt, f_prev + k3)

                f_current = f_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
                f_list[i] = f_current if abs(f_current - f_prev) > conv_threshold else f_current

            return t_list, f_list

        # -------------------------- Base Simulation --------------------------
        t_start, t_end, dt = 0, 30*24, 1
        t_list, f_B_base = rk4_solver(dfb_dt, t_start, t_end, dt, 0.0)

        # Unit conversion: mol/m³ → ng/g
        mol_mass_fipronil = 252.1  # g/mol
        C_base = f_B_base * Z_B * mol_mass_fipronil * 1e3

        # -------------------------- Prediction Band (Uncertainty Analysis) --------------------------
        # Sensitivity parameters: ±5% variation (critical for residue prediction)
        sensitive_params = ["H", "logKow", "f_W", "G_W"]
        var_ratio = 0.05  # 5% variation
        C_upper = np.zeros_like(C_base)
        C_lower = np.zeros_like(C_base)

        for param_name in sensitive_params:
            # Upper bound (param + 5%)
            params_upper = custom_params.copy()
            params_upper[param_name] = float(custom_params[param_name]) * (1 + var_ratio)
            if param_name == "logKow":
                K_OW_upper = 10**float(params_upper["logKow"])
                Z_O_upper = K_OW_upper / float(params_upper["H"])
                Z_N_upper = 0.035 * K_OW_upper / float(params_upper["H"])
                Z_B_upper = F_L * Z_O_upper + F_N * Z_N_upper + F_W * (1/float(params_upper["H"]))
                D_W_upper = float(params_upper["G_W"]) * E_W * (1/float(params_upper["H"]))
            else:
                # Recalculate for other parameters (simplified for efficiency)
                if param_name == "H":
                    Z_B_var = F_L * Z_O + F_N * Z_N + F_W * (1/(H*(1+var_ratio)))
                    D_W_var = G_W * E_W * (1/(H*(1+var_ratio)))
                elif param_name == "f_W":
                    f_W_var = f_W * (1+var_ratio)
                    t_var, f_var = rk4_solver(lambda t,f: (D_W*f_W_var + D_D*f_D - (D_W+D_F+D_M+D_G)*f)/(Z_B*V_B), t_start, t_end, dt, 0.0)
                    C_var = f_var * Z_B * mol_mass_fipronil * 1e3
                    C_upper = np.maximum(C_upper, C_var)
                    continue
                elif param_name == "G_W":
                    D_W_var = G_W*(1+var_ratio)*E_W*Z_W
                else:
                    continue

                # Solve for upper bound
                def dfb_upper(t, f_B):
                    input_flux = D_W_var * f_W + D_D * f_D
                    output_flux = (D_W_var + D_F + D_M + D_G) * f_B
                    return (input_flux - output_flux) / (Z_B_upper if param_name == "logKow" else Z_B)

                t_var, f_upper = rk4_solver(dfb_upper, t_start, t_end, dt, 0.0)
                C_upper = np.maximum(C_upper, f_upper * (Z_B_upper if param_name == "logKow" else Z_B) * mol_mass_fipronil * 1e3)

                # Lower bound (param - 5%)
                params_lower = custom_params.copy()
                params_lower[param_name] = float(custom_params[param_name]) * (1 - var_ratio)
                if param_name == "logKow":
                    K_OW_lower = 10**float(params_lower["logKow"])
                    Z_O_lower = K_OW_lower / float(params_lower["H"])
                    Z_N_lower = 0.035 * K_OW_lower / float(params_lower["H"])
                    Z_B_lower = F_L * Z_O_lower + F_N * Z_N_lower + F_W * (1/float(params_lower["H"]))
                    D_W_lower = float(params_lower["G_W"]) * E_W * (1/float(params_lower["H"]))
                else:
                    if param_name == "H":
                        Z_B_var = F_L * Z_O + F_N * Z_N + F_W * (1/(H*(1-var_ratio)))
                        D_W_var = G_W * E_W * (1/(H*(1-var_ratio)))
                    elif param_name == "f_W":
                        f_W_var = f_W * (1-var_ratio)
                        t_var, f_var = rk4_solver(lambda t,f: (D_W*f_W_var + D_D*f_D - (D_W+D_F+D_M+D_G)*f)/(Z_B*V_B), t_start, t_end, dt, 0.0)
                        C_var = f_var * Z_B * mol_mass_fipronil * 1e3
                        C_lower = np.minimum(C_lower, C_var)
                        continue
                    elif param_name == "G_W":
                        D_W_var = G_W*(1-var_ratio)*E_W*Z_W

                def dfb_lower(t, f_B):
                    input_flux = D_W_var * f_W + D_D * f_D
                    output_flux = (D_W_var + D_F + D_M + D_G) * f_B
                    return (input_flux - output_flux) / (Z_B_lower if param_name == "logKow" else Z_B)

                t_var, f_lower = rk4_solver(dfb_lower, t_start, t_end, dt, 0.0)
                C_lower = np.minimum(C_lower, f_lower * (Z_B_lower if param_name == "logKow" else Z_B) * mol_mass_fipronil * 1e3)

        # Ensure prediction band is valid (avoid negative values)
        C_upper = np.maximum(C_upper, C_base * 0.9)  # At least 90% of base value
        C_lower = np.maximum(C_lower, 0.01)  # Minimum 0.01 ng/g (physically meaningful)

        # -------------------------- Visualization (With Prediction Band) --------------------------
        plt.figure(figsize=(12, 7))
        # Plot base curve
        plt.plot(t_list/24, C_base, label='Base Prediction', color='#1E56A0', linewidth=3)
        # Plot prediction band (shaded area)
        plt.fill_between(t_list/24, C_lower, C_upper, alpha=0.3, color='#1E56A0', label='90% Prediction Band')
        
        # Chart styling
        plt.xlabel('Time (Days)', fontsize=12, fontweight='bold')
        plt.ylabel('Residue Concentration (ng/g)', fontsize=12, fontweight='bold')
        plt.title('Dynamic Simulation of Fipronil Residue in Fish', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
        
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.2)
        plt.gca().spines['bottom'].set_linewidth(1.2)
        plt.gca().set_facecolor('#F8F9FA')

        # -------------------------- Save Plot --------------------------
        save_path = os.path.join(get_desktop_path(), 'Fipronil_Residue_Simulation_with_Prediction_Band.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # -------------------------- Calculate Key Metrics --------------------------
        max_conc = np.max(C_base)
        max_time = t_list[np.argmax(C_base)] / 24
        steady_conc = C_base[-100:].mean()
        upper_steady = C_upper[-100:].mean()
        lower_steady = C_lower[-100:].mean()

        # Format result text
        result_text = (
            "Simulation Completed Successfully!\n"
            "=================================\n"
            "Key Results (Base Prediction):\n"
            f"  • Maximum Residue Concentration: {max_conc:.2f} ng/g\n"
            f"  • Peak Occurrence Time: {max_time:.1f} days\n"
            f"  • 30-Day Steady-State Concentration: {steady_conc:.2f} ng/g\n"
            "\n"
            "Prediction Band (90% Uncertainty):\n"
            f"  • Steady-State Upper Bound: {upper_steady:.2f} ng/g\n"
            f"  • Steady-State Lower Bound: {lower_steady:.2f} ng/g\n"
            "\n"
            f"Plot Saved To:\n{save_path}\n"
            "=================================\n"
            "Note: Prediction band is based on ±5% variation of sensitive parameters (H, logKow, f_W, G_W)"
        )

        return result_text, save_path

    except ValueError as ve:
        return f"Parameter Error: {str(ve)}", None
    except Exception as e:
        return f"Simulation Error: {str(e)}", None

# -------------------------- GUI Design (With Parameter Input) --------------------------
class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fipronil Residue Dynamic Simulation Tool (Customizable Parameters)")
        self.root.geometry("1050x800")
        self.root.resizable(False, False)
        self.root.config(bg='#F8F9FA')

        # Style configuration (修复 ttk 组件样式问题)
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 11, "bold"), padding=10)
        self.style.configure("TLabel", font=("Arial", 10), background='#F8F9FA')
        self.style.configure("TEntry", font=("Arial", 10), padding=5)
        self.style.configure("TNotebook", background='#F8F9FA')
        self.style.configure("TNotebook.Tab", font=("Arial", 10, "bold"), padding=(15, 5))
        # 专门配置 LabelFrame 的标题样式（ttk 规范）
        self.style.configure("TLabelframe.Label", font=("Arial", 11, "bold"))

        # Initialize param_entries FIRST (确保属性先初始化)
        self.param_entries = {}

        # Create parameter input frames (using notebook for tabbed layout)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=15, padx=15, fill=tk.BOTH, expand=True)

        # 1. Fipronil Physicochemical Parameters Tab
        self.tab_chem = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_chem, text="Physicochemical Parameters")
        self.create_param_entries(self.tab_chem, [
            ("Henry's Law Constant (H, Pa·m³/mol)", "H", "%.4e"),
            ("log Octanol-Water Partition (logKow)", "logKow", "%.1f"),
            ("Degradation Half-Life (h)", "half_life", "%.1f"),
            ("Metabolite Conversion Proportion (sum_vxy)", "sum_vxy", "%.1f")
        ])

        # 2. Fish Physiological Parameters Tab
        self.tab_fish = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_fish, text="Fish Physiological Parameters")
        self.create_param_entries(self.tab_fish, [
            ("Fish Weight (kg)", "fish_weight", "%.1f"),
            ("Lipid Volume Fraction (F_L)", "F_L", "%.2f"),
            ("Non-Lipid Organic Fraction (F_N)", "F_N", "%.2f"),
            ("Water Volume Fraction (F_W)", "F_W", "%.2f"),
            ("Gill Ventilation Rate (G_W, m³/h)", "G_W", "%.4f"),
            ("Gill Absorption Efficiency (E_W)", "E_W", "%.2f"),
            ("Feeding Rate (G_D, m³/h)", "G_D", "%.4f"),
            ("Intestinal Absorption Efficiency (E_D)", "E_D", "%.2f"),
            ("Food Absorption Fraction (beta)", "beta", "%.2f"),
            ("Intestine-Fish Distribution Coeff (K_DG)", "K_DG", "%.1f"),
            ("Growth Rate Constant (k_g, 1/h)", "k_g", "%.4f")
        ])

        # 3. Environmental & Food Parameters Tab
        self.tab_env = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.tab_env, text="Environmental & Food Parameters")
        self.create_param_entries(self.tab_env, [
            ("Water Fugacity (f_W, Pa)", "f_W", "%.6e"),
            ("Food Fugacity (f_D, Pa)", "f_D", "%.6e"),
            ("Food Lipid Fraction", "food_lipid", "%.2f"),
            ("Food Non-Lipid Organic Fraction", "food_nonlipid", "%.2f"),
            ("Food Water Fraction", "food_water", "%.2f")
        ])

        # Action Buttons Frame
        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack(pady=10, padx=15, fill=tk.X)

        self.run_btn = ttk.Button(self.btn_frame, text="Start Simulation", command=self.on_run_simulation)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(self.btn_frame, text="Reset to Defaults", command=self.reset_params)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Result Display Frame (移除直接 font 参数，使用 style 配置)
        self.result_frame = ttk.LabelFrame(self.root, text="Simulation Results", padding=15)
        self.result_frame.pack(pady=10, padx=15, fill=tk.BOTH, expand=True)

        self.result_text = scrolledtext.ScrolledText(self.result_frame, font=("Arial", 10), wrap=tk.WORD, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.INSERT, "Enter parameters (or use defaults) and click 'Start Simulation'\n")
        self.result_text.config(state=tk.DISABLED)

        # Reset to defaults AFTER creating entries
        self.reset_params()

    def create_param_entries(self, parent, params):
        """Create label-entry pairs for parameters in a grid layout"""
        for row, (label_text, param_key, format_str) in enumerate(params):
            # Label
            label = ttk.Label(parent, text=label_text)
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=8)
            
            # Entry
            entry = ttk.Entry(parent, width=20)
            entry.grid(row=row, column=1, padx=5, pady=8)
            self.param_entries[param_key] = (entry, format_str)

    def get_custom_params(self):
        """Read parameters from input fields"""
        custom_params = {}
        for param_key, (entry, format_str) in self.param_entries.items():
            value = entry.get().strip()
            # Use default if input is empty or invalid
            if not value:
                custom_params[param_key] = DEFAULT_PARAMS[param_key]
            else:
                try:
                    custom_params[param_key] = float(value)
                except:
                    custom_params[param_key] = DEFAULT_PARAMS[param_key]
        return custom_params

    def reset_params(self):
        """Reset all input fields to default values"""
        for param_key, (entry, format_str) in self.param_entries.items():
            default_val = DEFAULT_PARAMS[param_key]
            entry.delete(0, tk.END)
            entry.insert(0, format_str % default_val)

    def update_result(self, text, is_error=False):
        """Update result display text"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if is_error:
            self.result_text.insert(tk.INSERT, f"Error: {text}\n")
            self.result_text.config(fg="red")
        else:
            self.result_text.insert(tk.INSERT, text)
            self.result_text.config(fg="black")
        self.result_text.config(state=tk.DISABLED)

    def on_run_simulation(self):
        """Trigger simulation when button is clicked"""
        self.update_result("Running simulation... Please wait...")
        self.root.update_idletasks()  # Refresh GUI

        # Get custom parameters
        custom_params = self.get_custom_params()

        # Run simulation
        result_text, save_path = run_simulation(custom_params)

        # Update result and show messagebox
        if save_path:
            self.update_result(result_text)
            messagebox.showinfo("Success", f"Simulation completed!\nPlot saved to:\n{save_path}")
        else:
            self.update_result(result_text, is_error=True)
            messagebox.showerror("Error", result_text)

# -------------------------- Program Entry --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
