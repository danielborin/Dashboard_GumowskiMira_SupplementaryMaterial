import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pynamicalsys import DiscreteDynamicalSystem as dds, PlotStyler

# Define the Gumowski–Mira map
@njit
def Gumowski(x, mu):
    return x * mu + 2 * x * x * (1 - mu) / (1 + x * x)

@njit
def Gumowski_prime(x, mu):
    return mu + 4 * (1 - mu) * x / ((1 + x * x)**2)

@njit
def Gumowski_Mira_map(state, parameters):
    x, y = state
    a, b, mu = parameters
    x_new = a * y * (1 - b * y**2) + y + Gumowski(x, mu)
    y_new = -x + Gumowski(x_new, mu)
    return np.array([x_new, y_new])

@njit
def Gumowski_Mira_map_jacobian(state, parameters, *args):
    x, y = state
    a, b, mu = parameters
    Gx = Gumowski(x, mu)
    x_new = a * y * (1 - b * y * y) + y + Gx
    y_new = -x + Gumowski(x_new, mu)
    Gp_x = Gumowski_prime(x, mu)
    Gp_xnew = Gumowski_prime(x_new, mu)

    dfdx = Gp_x
    dfdy = a * (1 - 3 * b * y * y) + 1
    dgdx = -1 + Gp_xnew * dfdx
    dgdy = Gp_xnew * dfdy

    return np.array([[dfdx, dfdy], [dgdx, dgdy]])

ds = dds(mapping=Gumowski_Mira_map, jacobian=Gumowski_Mira_map_jacobian, system_dimension=2, number_of_parameters=3)

# Function to compute and plot the bifurcation diagram
def GumowskiMira_bifurcation(a=0.30, b=0.20, mu_min=-0.6, mu_max=0.6):
    parameters = np.array([a, b])

    # Total time of interaction and transient time
    total_time = 2000
    transient_time = 1000

    # Define Initial Condition
    u0 = [0, 0.5]

    # Generate the bifurcation diagram for varying 'mu'
    param_index = 2
    param_range = (mu_min, mu_max, 1000)

    param_values, bifurcation_diagram, u_new = ds.bifurcation_diagram(
        u=u0,
        parameters=parameters,
        param_index=param_index,
        param_range=param_range,
        total_time=total_time,
        transient_time=transient_time,
        return_last_state=True
    )

    param_mesh = np.repeat(param_values[:, np.newaxis], bifurcation_diagram.shape[1], axis=1)
    param_values = param_mesh.flatten()
    bifurcation_diagram = bifurcation_diagram.flatten()

    # Set the style for the plot
    ps = PlotStyler()
    ps.apply_style()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    # ps.set_tick_padding(ax[1], pad_x=6))

    # Plot the bifurcation diagram
    ax.scatter(param_values, bifurcation_diagram, color='black', s=0.05, edgecolor='none')

    # Set the labels and limits for the plot
    ax.set_xlim(param_range[0], param_range[1])
    ax.set_ylabel("$x$")
    ax.set_xlabel("$b$")

    plt.tight_layout(pad=0.1)

    return fig


def plot_gumowski_mira(a=0.01, b=0.05, mu=-0.8, n=int(1e6)):
    u0 = [0, 0.85]

    # Set up the colormap
    colormap = mpl.colormaps['GnBu']

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))  # Single plot, adjust size as needed

    # Generate the trajectory
    parameters = [a, b, mu]
    trajectory = ds.trajectory(u0, n, parameters=parameters)  # Assuming ds.trajectory function exists

    # Plot the trajectory
    sc = ax.scatter(
        trajectory[:, 0], trajectory[:, 1],
        c=np.arange(len(trajectory)),   # Gradient along trajectory
        cmap=colormap,
        s=0.1,
        marker='o',
        edgecolor='none',
        linewidth=0
    )

    # Remove axes and background
    ax.axis('off')  # Turn off the axes


    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot in Streamlit
    return fig


# -----------------------------------------

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('Figures/fig_attractors_single_plot.png')
    st.sidebar.title('Gumovski-Mira Map')
    st.text('by Daniel Borin')
    selected = option_menu(
        menu_title = None,
        options = ["Home","Attractors","Bifurcation","Parameter Space μ×b","Parameter Space μ×a","Parameter Space a×b"],
        # icons = ["house","gear","activity","snowflake","envelope"],
        icons = [" "," "," "," "," ", " "],
        menu_icon = None,
        default_index = 0,
        # orientation = "horizontal",
        styles={
            # "container": {"padding": "0!important", "background-color": "#fafafa"},
            # "icon": {"color": "orange", "font-size": "25px"}, 
            # "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link": { "text-align": "left", "margin":"0px", "--hover-color": "lightskyblue"},
            "nav-link-selected": {"background-color": "dodgerblue"},
        }
    )
    st.markdown('[Visit Daniel Borin\'s Homepage](https://danielborin.github.io)')



if selected == "Home":
    st.image('Figures/fig_attractors_homepage_withoutaxis.png')
    st.title("Supplementary Material: Parameter Space for Gumovski-Mira Map")

    st.write("""
    Welcome to the interactive **Dashboard** designed to explore the **parameter space** of the **Gumovski-Mira Map**.
    
    This dashboard provides a visual and interactive platform to study the dynamics of the **Gumovski-Mira map** and its bifurcations, 
    as presented in the paper **"Parameter Space for Gumovski-Mira Map"** by **Diego Fregolent Mendes de Oliveira** and **Daniel Borin**. 
    
    The goal of this supplementary material is to help users understand the behavior of the system by visualizing key aspects of the parameter space.
    Through this app, you can interactively adjust the model parameters and explore their effects on the system's attractors, bifurcation diagrams, and parameter spaces.
    
    #### Main Features:
    1. **Attractors**: Visualize the attractors of the system for different parameter values (`a`, `b`, and `mu`).
    2. **Bifurcation Diagrams**: Explore how the bifurcation diagrams evolve as parameters are varied.
    3. **Parameter Space Exploration**: View interactive videos that show the **parameter space** of the Gumovski-Mira map, including various projections like:
        - $\\mu \\times b$ 
        - $\\mu \\times a$ 
        - $a \\times b$
    
    #### How to Use:
    - **Adjust Parameters**: Use the sliders to modify parameters (`a`, `b`, and `mu`) and observe their impact on the system's dynamics.
    - **Explore Attractors**: Select the "Attractors" tab to visualize the trajectory of the system for the chosen parameters.
    - **Bifurcation Analysis**: Visit the "Bifurcation" section to study how bifurcations unfold as you vary the control parameters.
    - **View Parameter Spaces**: The "Parameter Space" tabs display animations that show how the system behaves in various regions of the parameter space.
             
    Feel free to explore the various sections, adjust parameters, and visually observe how the system behaves across different regimes.
             
    For more information, visit the [author's homepage](https://danielborin.github.io).

    """)

    st.markdown("""
    #### About the Gumovski-Mira Map:
    The **Gumovski-Mira map** is a **nonlinear dynamical system** that exhibits a rich variety of behaviors, including bifurcations, periodic orbits, and chaotic dynamics. The system's behavior depends critically on the choice of its parameters: $a$, $b$, and $\\mu$. 

    The map describes the evolution of two variables $x$ and $y$, and is defined by the following set of equations:

    $
    \\displaystyle x_{n+1} = a  y_n  \\left( 1 - b  y_n^2 \\right) + y_n + G(x_n, \mu)
    $
                
    $
    \\displaystyle y_{n+1} = -x_n + G(x_{n+1}, \mu)
    $

    where:

    - $x_n$ and $y_n$ are the current values of the state variables at time step $n$,
    - $a$, $b$, and $\\mu$ are parameters that influence the system's dynamics,
    - The function $G(x, \\mu)$ is defined as:

    $
    \\displaystyle G(x, \mu) = \\mu x + \\frac{2x(1-\mu)}{1 + x^2}
    $

    """)




if selected == "Attractors":
    st.header(selected)
    # Streamlit sliders for interactive input
    a = st.slider('Parameter a', 0.0, 1.0, 0.01, 0.01)
    b = st.slider('Parameter b', 0.0, 1.0, 0.05, 0.01)
    mu = st.slider('Parameter μ', -0.99, 0.99, -0.8, 0.01)

    # Generate the plot
    fig = plot_gumowski_mira(a, b, mu)

    # Display the figure in Streamlit
    st.pyplot(fig)

if selected == "Bifurcation":
    st.header(selected)
    # Streamlit sliders for interactive input
    a = st.slider('Parameter a', 0.0, 1.0, 0.30, 0.01)
    b = st.slider('Parameter b', 0.0, 1.0, 0.20, 0.01)
    mu_min = st.slider('Minimum μ', -1.0, 0.0, -0.6, 0.01)
    mu_max = st.slider('Maximum μ', 0.0, 1.0, 0.6, 0.01)

    # Generate the plot
    fig = GumowskiMira_bifurcation(a, b, mu_min, mu_max)

    # Display the figure in Streamlit
    st.pyplot(fig)

if selected == "Parameter Space μ×b":
    st.header("Parameter Space μ×b")
    st.video("Videos/Animation_Parameter_Space_a_GumovskiMira.mp4", format='video/mp4')

if selected == "Parameter Space μ×a":
    st.header("Parameter Space μ×a")
    st.video("Videos/Animation_Parameter_Space_b_GumovskiMira.mp4", format='video/mp4')

if selected == "Parameter Space a×b":
    st.header("Parameter Space a×b")

    st.video("Videos/Animation_Parameter_Space_mu_GumovskiMira.mp4", format='video/mp4')

