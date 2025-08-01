import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
import warnings

try:
    import sympy as sp
    from sympy.printing.latex import LatexPrinter
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Helper Functions for Mathematical Formatting ---

def format_polynomial(coeffs, var='s'):
    """Formats a list of coefficients into a nice polynomial string for LaTeX."""
    terms = []
    degree = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-9:  # Skip terms with zero coefficients
            continue

        power = degree - i
        
        # Handle sign
        sign = " - " if c < 0 else " + "
        c_abs = abs(c)

        # Format coefficient (don't show 1.0 unless it's a constant term)
        if c_abs == 1.0 and power != 0:
            coeff_str = ""
        else:
            coeff_str = f"{c_abs:.2f}"
            
        # Format variable
        if power == 0:
            var_str = ""
        elif power == 1:
            var_str = var
        else:
            var_str = f"{var}^{{{power}}}"

        term = f"{coeff_str}{var_str}"
        
        if not terms: # First term
            terms.append(f"-{term}" if c < 0 else term)
        else:
            terms.append(f"{sign}{term}")
            
    if not terms:
        return "0"
    # Clean up leading " + "
    full_str = "".join(terms).strip()
    if full_str.startswith('+ '):
        full_str = full_str[2:]
    return full_str

class CustomLatexPrinter(LatexPrinter):
    """Custom printer to round numbers in LaTeX output."""
    def _print_Float(self, expr):
        return f"{expr:.2f}"

@st.cache_data
def get_time_domain_expression(num_coeffs, den_coeffs, input_type, amplitude=1.0, sin_freq=1.0):
    """Computes the symbolic time-domain expression y(t) using SymPy."""
    if not SYMPY_AVAILABLE:
        return None, None # Return None if sympy is not installed
        
    try:
        t, s = sp.symbols('t s', real=True, positive=True)
        
        num = [float(c) for c in num_coeffs]
        den = [float(c) for c in den_coeffs]

        num_poly = sp.Poly(num, s)
        den_poly = sp.Poly(den, s)
        
        if den_poly.is_zero: return "H(s) = \\frac{N(s)}{0}", "\\text{Denominator is zero}"

        H_s = num_poly / den_poly
        
        # Define input U(s) based on type, including amplitude
        amp = float(amplitude)
        if input_type == 'Impulse':
            U_s = amp
            U_s_latex = f"{amp:.2f}"
        elif input_type == 'Step':
            U_s = amp/s
            U_s_latex = f"\\frac{{{amp:.2f}}}{{s}}"
        elif input_type == 'Ramp':
            U_s = amp/s**2
            U_s_latex = f"\\frac{{{amp:.2f}}}{{s^2}}"
        elif input_type == 'Sinusoidal':
            w = 2 * np.pi * sin_freq
            U_s = amp * w / (s**2 + w**2)
            U_s_latex = f"\\frac{{{amp * w:.2f}}}{{s^2 + {w**2:.2f}}}"
        else:
            return "Error", "Unknown input"

        Y_s = H_s * U_s
        
        # Perform inverse Laplace transform and simplify
        y_t = sp.inverse_laplace_transform(Y_s, s, t, noconds=True).simplify()
        
        # Use custom printer for cleaner output
        printer = CustomLatexPrinter()
        y_t_latex = printer.doprint(y_t)
        Y_s_latex = f"H(s) \\cdot U(s) = \\left( {sp.latex(H_s)} \\right) \\cdot \\left( {U_s_latex} \\right)"
        
        return Y_s_latex, y_t_latex
    except Exception:
        return "Error", "\\text{Could not compute symbolic solution}"

class TransferFunction:
    """A class to represent and manipulate a transfer function."""
    def __init__(self, num_coeffs, den_coeffs, name="TF"):
        self.num_coeffs = np.array(num_coeffs, dtype=float)
        self.den_coeffs = np.array(den_coeffs, dtype=float)
        self.name = name
        self.color = f"rgb({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)})"

    def get_tf(self):
        if np.all(self.den_coeffs == 0):
            return signal.TransferFunction([1], [1])
        return signal.TransferFunction(self.num_coeffs, self.den_coeffs)

    def get_poles_zeros(self):
        tf = self.get_tf()
        return tf.poles, tf.zeros

    def get_response(self, input_type, t, amplitude=1.0, freq=1.0):
        tf = self.get_tf()
        y = np.zeros_like(t)
        t_resp = t

        if input_type == 'Impulse':
            t_resp, y_unit = signal.impulse(tf, T=t)
            y = y_unit * amplitude
        elif input_type == 'Step':
            t_resp, y_unit = signal.step(tf, T=t)
            y = y_unit * amplitude
        elif input_type == 'Ramp':
            t_resp, step_y_unit = signal.step(tf, T=t)
            ramp_y_unit = np.cumsum(step_y_unit) * (t[1] - t[0]) if len(t) > 1 else np.array([0])
            y = ramp_y_unit * amplitude
        elif input_type == 'Sinusoidal':
            w = 2 * np.pi * freq
            u = amplitude * np.sin(w * t)
            t_resp, y, _ = signal.lsim(tf, u, t)
        
        return t_resp, y

@st.cache_data
def get_time_vector(time_range):
    num_points = int(min(time_range * 50, 1000))
    return np.linspace(0, time_range, num_points)

def add_transfer_function(order):
    tf_count = len(st.session_state.transfer_functions) + 1
    name = f"TF_{tf_count}_{order}{'st' if order == 1 else 'nd'}_order"
    if order == 1:
        num_coeffs, den_coeffs = [1.0, 1.0], [1.0, 1.0]
    else:
        num_coeffs, den_coeffs = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    tf = TransferFunction(num_coeffs, den_coeffs, name)
    st.session_state.transfer_functions.append(tf)
    st.session_state.current_tf_index = len(st.session_state.transfer_functions) - 1
    st.success(f"Added {order}{'st' if order == 1 else 'nd'} order system: {name}")

def create_plots(input_type, time_range, amplitude=1.0, sin_freq=1.0):
    if not st.session_state.transfer_functions:
        return go.Figure(), go.Figure()

    time_fig = go.Figure()
    t = get_time_vector(time_range)
    for i, tf in enumerate(st.session_state.transfer_functions):
        is_current = (i == st.session_state.current_tf_index)
        try:
            t_resp, y = tf.get_response(input_type, t, amplitude, sin_freq)
            line_style = dict(width=3.5) if is_current else dict(width=1.5, dash='dot')
            opacity = 1.0 if is_current else 0.6
            time_fig.add_trace(go.Scatter(
                x=t_resp, y=y, mode='lines', name=tf.name,
                line=dict(color=tf.color, **line_style), opacity=opacity,
                hovertemplate=f'<b>{tf.name}</b><br>Time: %{{x:.3f}}s<br>Amplitude: %{{y:.3f}}<extra></extra>'
            ))
        except Exception:
            continue
    time_fig.update_layout(
        title=f'<b>Time Response - {input_type} Input</b>', xaxis_title='Time (s)', yaxis_title='Amplitude',
        height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    pz_fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 100)
    pz_fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode='lines', name='Unit Circle',
        line=dict(color='gray', dash='dash', width=1), opacity=0.5, showlegend=False, hoverinfo='skip'
    ))
    for i, tf in enumerate(st.session_state.transfer_functions):
        is_current = (i == st.session_state.current_tf_index)
        try:
            poles, zeros = tf.get_poles_zeros()
            size, opacity = (14, 1.0) if is_current else (10, 0.6)
            if len(poles) > 0:
                pz_fig.add_trace(go.Scatter(
                    x=poles.real, y=poles.imag, mode='markers', name=f'{tf.name} Poles',
                    marker=dict(symbol='x', size=size, color=tf.color, line=dict(width=2.5)), opacity=opacity,
                    hovertemplate=f'<b>{tf.name} Pole</b><br>Real: %{{x:.3f}}<br>Imag: %{{y:.3f}}<extra></extra>'
                ))
            if len(zeros) > 0:
                pz_fig.add_trace(go.Scatter(
                    x=zeros.real, y=zeros.imag, mode='markers', name=f'{tf.name} Zeros',
                    marker=dict(symbol='circle-open', size=size, color=tf.color, line=dict(width=2.5)), opacity=opacity,
                    hovertemplate=f'<b>{tf.name} Zero</b><br>Real: %{{x:.3f}}<br>Imag: %{{y:.3f}}<extra></extra>'
                ))
        except Exception:
            continue
    pz_fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)
    pz_fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.4)
    pz_fig.update_layout(
        title='<b>Pole-Zero Plot</b>', xaxis_title='Real Part (σ)', yaxis_title='Imaginary Part (jω)',
        height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(scaleanchor="y", scaleratio=1, zeroline=False), yaxis=dict(zeroline=False),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return time_fig, pz_fig

def coefficient_widget(label, session_key, min_val=-10.0, max_val=10.0, step=0.1):
    st.markdown(f"**{label}**")
    def slider_callback(): st.session_state[session_key] = st.session_state[f"{session_key}_slider"]
    def number_callback(): st.session_state[session_key] = st.session_state[f"{session_key}_number"]
    s_col, n_col = st.columns([2, 1])
    with s_col:
        # Corrected st.slider call with explicit keyword arguments
        st.slider(
            label, 
            min_value=min_val, 
            max_value=max_val, 
            value=st.session_state.get(session_key, 1.0), 
            step=step, 
            on_change=slider_callback, 
            key=f"{session_key}_slider", 
            label_visibility="collapsed"
        )
    with n_col:
        st.number_input(
            label, 
            min_value=min_val, 
            max_value=max_val, 
            value=st.session_state.get(session_key, 1.0), 
            step=step, 
            on_change=number_callback, 
            key=f"{session_key}_number", 
            label_visibility="collapsed"
        )
    return st.session_state.get(session_key, 1.0)

def main():
    st.set_page_config(page_title="Real-time Transfer Function Analyzer", page_icon="🎛️", layout="wide", initial_sidebar_state="collapsed")
    st.title("🎛️ Real-time Transfer Function Analyzer")

    if 'transfer_functions' not in st.session_state:
        st.session_state.transfer_functions, st.session_state.current_tf_index = [], 0

    cols = st.columns([1.2, 1.2, 1.4, 2, 1.5, 2])
    with cols[0]:
        if st.button("➕ Add 1st Order", use_container_width=True): add_transfer_function(1); st.rerun()
    with cols[1]:
        if st.button("➕ Add 2nd Order", use_container_width=True): add_transfer_function(2); st.rerun()
    with cols[2]:
        if st.button("➖ Remove Active TF", use_container_width=True):
            if st.session_state.transfer_functions:
                idx = st.session_state.current_tf_index
                st.session_state.transfer_functions.pop(idx)
                st.session_state.current_tf_index = max(0, min(idx, len(st.session_state.transfer_functions) - 1))
                st.rerun()
    with cols[3]:
        input_type = st.selectbox("Input Type:", ['Step', 'Impulse', 'Ramp', 'Sinusoidal'], key="input_select")
    with cols[4]:
        time_range = st.slider("Time Range (s)", 1.0, 50.0, 10.0, 0.5, key="time_slider")
    with cols[5]:
        if st.session_state.transfer_functions:
            tf_names = [f"TF {i+1}: {tf.name}" for i, tf in enumerate(st.session_state.transfer_functions)]
            idx = st.selectbox("Active TF:", options=range(len(tf_names)), format_func=lambda x: tf_names[x], index=st.session_state.current_tf_index, key="tf_select")
            if st.session_state.current_tf_index != idx: st.session_state.current_tf_index = idx; st.rerun()

    if not st.session_state.transfer_functions:
        st.info("👆 Welcome! Add a transfer function to begin. You can zoom/pan by hovering over the plots.")
        if not SYMPY_AVAILABLE: st.warning("To see symbolic math expressions, please install SymPy: `pip install sympy`")
        return

    current_tf = st.session_state.transfer_functions[st.session_state.current_tf_index]
    is_first_order = len(current_tf.den_coeffs) == 2
    
    st.markdown("---")

    # --- Input Signal Parameter Controls (Moved Up) ---
    st.subheader("Input Signal Parameters")
    input_amplitude, sin_freq = 1.0, 1.0 # Default values
    
    if input_type in ['Step', 'Impulse', 'Ramp']:
        input_amplitude = coefficient_widget("Amplitude", "input_amplitude")
    elif input_type == 'Sinusoidal':
        col1, col2 = st.columns(2)
        with col1:
            input_amplitude = coefficient_widget("Amplitude", "input_amplitude")
        with col2:
            sin_freq = coefficient_widget("Frequency (Hz)", "sin_freq", 0.1, 10.0)
    
    st.markdown("---")
    
    # --- TF Coefficient Controls ---
    if is_first_order:
        st.subheader(f"Coefficients for {current_tf.name}: $H(s) = (b_1s + b_0) / (a_1s + a_0)$")
        labels, arrays, indices = ["b₁", "b₀", "a₁", "a₀"], [current_tf.num_coeffs]*2 + [current_tf.den_coeffs]*2, [0, 1, 0, 1]
    else:
        st.subheader(f"Coefficients for {current_tf.name}: $H(s) = (b_2s^2 + b_1s + b_0) / (a_2s^2 + a_1s + a_0)$")
        labels, arrays, indices = ["b₂", "b₁", "b₀", "a₂", "a₁", "a₀"], [current_tf.num_coeffs]*3 + [current_tf.den_coeffs]*3, [0, 1, 2, 0, 1, 2]

    key_prefix = f"tf_{st.session_state.current_tf_index}"
    for label, arr, idx in zip(labels, arrays, indices):
        session_key = f"{key_prefix}_{label}"
        if session_key not in st.session_state: st.session_state[session_key] = float(arr[idx])

    ui_cols = st.columns(len(labels) // 2 if not is_first_order else len(labels))
    for i, (label, arr, idx) in enumerate(zip(labels, arrays, indices)):
        with ui_cols[i % len(ui_cols)]:
            session_key = f"{key_prefix}_{label}"
            coefficient_widget(label, session_key)
            arr[idx] = st.session_state[session_key]

    st.markdown("---")

    # Display plots first
    time_fig, pz_fig = create_plots(input_type, time_range, input_amplitude, sin_freq)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(time_fig, use_container_width=True)
    with col2:
        st.plotly_chart(pz_fig, use_container_width=True)

    st.markdown("---")
    
    # Display analysis and expressions below the plots
    st.subheader("System Analysis & Expressions")
    
    # System Analysis Metrics
    try:
        poles, zeros = current_tf.get_poles_zeros()
        is_stable = np.all(np.real(poles) < 0) if len(poles) > 0 else True
        stability = "Stable ✅" if is_stable else "Unstable ❌"
        
        cols = st.columns(4)
        cols[0].metric("System Order", f"{len(current_tf.den_coeffs)-1}")
        cols[1].metric("Stability", stability)
        cols[2].metric("Poles", len(poles))
        cols[3].metric("Zeros", len(zeros))

        # Expander for Pole/Zero values
        with st.expander("Show Pole/Zero Values"):
            p_col, z_col = st.columns(2)
            with p_col:
                st.markdown("**Poles**")
                if len(poles) > 0:
                    for i, p in enumerate(poles):
                        st.write(f"p{i+1}: {p.real:.3f} + {abs(p.imag):.3f}j" if p.imag != 0 else f"p{i+1}: {p.real:.3f}")
                else:
                    st.write("No poles.")
            
            with z_col:
                st.markdown("**Zeros**")
                if len(zeros) > 0:
                    for i, z in enumerate(zeros):
                        st.write(f"z{i+1}: {z.real:.3f} + {abs(z.imag):.3f}j" if z.imag != 0 else f"z{i+1}: {z.real:.3f}")
                else:
                    st.write("No finite zeros.")
        
    except Exception as e:
        st.error(f"Could not analyze system properties. Error: {e}")
        
    st.markdown("---")

    # System Expressions
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Transfer Function $H(s)$")
        num_str = format_polynomial(current_tf.num_coeffs)
        den_str = format_polynomial(current_tf.den_coeffs)
        st.latex(f"H(s) = \\frac{{{num_str}}}{{{den_str}}}")

    with col2:
        st.markdown(f"##### Time Response for {input_type} Input")
        if SYMPY_AVAILABLE:
            Y_s_latex, y_t_latex = get_time_domain_expression(tuple(current_tf.num_coeffs), tuple(current_tf.den_coeffs), input_type, input_amplitude, sin_freq)
            st.latex(f"Y(s) = {Y_s_latex}")
            # Use st.latex directly to ensure proper rendering
            st.latex(f"y(t) = {y_t_latex}")
            st.caption("Note: Symbolic solutions can become very complex.")
        else:
            st.warning("Install `sympy` to view symbolic expressions.")


if __name__ == "__main__":
    main()
