# Dashboard_GumovskiMira_SupplementaryMaterial
This repository contains the supplementary material for the paper **"Parameter Space for Gumovski-Mira Map"** authored by **Diego Fregolent Mendes de Oliveira** and **Daniel Borin**. The goal of this project is to provide an interactive platform to explore the parameter space of the **Gumovski-Mira Map** using a **Streamlit** dashboard.

## Running the Dashboard Locally

To run this Streamlit dashboard locally, follow these steps:

### 1. Clone the repository:

```
git clone https://github.com/<your-username>/gumovski-mira-dashboard.git
cd gumovski-mira-dashboard
```

### 2. Set up a virtual environment (optional but recommended):

```
virtualenv .venv
source .venv/bin/activate  # For Windows, use: .venv\Scripts\activate
```

### 3. Install the required dependencies:

```
pip install -r requirements.txt
```

### 4. Run the dashboard:

```
streamlit run dashboard_GumovskiMira.py 
```

This will launch the Streamlit app in your default web browser.

## Requirements

To run the dashboard locally, you'll need the following Python libraries:

- `streamlit` – for building the interactive dashboard.
- `numpy` – for numerical operations and arrays.
- `matplotlib` – for creating the visualizations (bifurcation diagrams, attractors).
- `pandas` – for handling data frames (if needed).
- `numba` – for optimized performance in numerical computations.
- `pynamicalsys` – for simulating the dynamical system (Gumovski-Mira map).
- `streamlit-option-menu` – for creating the navigation menu in the sidebar.

These dependencies are listed in the `requirements.txt` file and can be installed using `pip`.

## How to Contribute

If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
- **Daniel Borin**
For more information, visit the [author's homepage](https://danielborin.github.io).
