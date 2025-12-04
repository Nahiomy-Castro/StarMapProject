#view.py
import importlib.util
import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

StarData = import_from_path("star_data", os.path.join(PROJECT_ROOT, "model", "star_data.py")).StarData
ConstellationData = import_from_path("constellation_data", os.path.join(PROJECT_ROOT, "model", "constellation_data.py")).ConstellationData
Controller = import_from_path("controller", os.path.join(PROJECT_ROOT, "controller", "controller.py")).Controller
StarsFactory = import_from_path("stars_factory", os.path.join(PROJECT_ROOT, "model", "stars_factory.py")).StarsFactory


def main():
    st.title("⭐ Star Atlas")
    st.divider()

    # Init MVC
    stars = StarData()
    consts = ConstellationData()
    controller = Controller(stars, consts)

    # UI controls
    distance_ly = st.slider("Distance from target star (light years):", 10, 500, 500)

    options = stars.stars['proper']
    options = options[options != ""].dropna().to_list()

    target_query = st.selectbox("Target star:", options)

    # Convert
    distance_pc = distance_ly / 3.26

    # Find target
    target = controller.get_target(target_query)
    if target is None:
        st.error("Star not found.")
        return

    st.success(f"Centered on: {target['proper'] or target['bf'] or 'HIP '+str(target['hip'])}")

    # Prepare filtered stars
    filtered = controller.prepare_stars(target, distance_pc)

    # Plot
    fig = controller.build_plot(filtered)
    st.plotly_chart(fig, width='stretch')

    # Info panel
    st.subheader("Star Details")
    st.write(f"**Name:** {target['proper'] or 'Unnamed'}")
    st.write(f"**Bayer/Flamsteed:** {target['bf']}")
    st.write(f"**HIP:** {target['hip']}")
    st.write(f"**Distance (pc):** {target['dist']}")
    st.write(f"**Magnitude:** {target['mag']}")
    st.write(f"**Spectral Type:** {target['spect']}")
    st.write(f"**Constellation:** {target['con']}")

if __name__ == "__main__":
    main()
