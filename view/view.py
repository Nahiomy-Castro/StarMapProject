#view.py
import importlib.util
import os
import sys
import streamlit as st
import numpy as np

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


def view(train_model):

    st.set_page_config(
        page_title="₊ ⊹StarMap⭑𓂃.ᐟ",
        page_icon="view/WebStarIcon.png",
        layout="wide"
    )
    st.title(":yellow[₊ ⊹]Star Map :yellow[⭑𓂃.ᐟ]")
    st.subheader("How it works")
    st.text("↪ Adjust the slider to view stars relative to the selected star at certain light years away\n"
            "↪ Select the star you want to view in the search box. You can write the proper name or select it.\n"
            "↪ For best results, make the graph fullscreen.\n"
            "↪ Zoom in or out of the graph to view the constellations and nearby stars.\n"
            "↪ Hover the mouse over a star to preview their information.")
    st.image("view/stars_divider.png")

    # Init MVC
    stars = StarData()
    consts = ConstellationData()
    controller = Controller(stars, consts)

    # UI controls
    distance_ly = st.slider("Distance from target star (light years):", 100, 17000, 17000)
    st.caption("Select 100 for less stars and 17000 for more stars.")

    # Radio to choose search type
    search_type = st.radio(
        "Search by:",
        ("Proper Name", "Bayer/Flamsteed", "HIP")
    )

    target_query = None

    if search_type == "Proper Name":
        proper_options = stars.stars['proper']
        proper_options = proper_options[proper_options != ""].dropna().to_list()
        proper_options.sort()
        default_index = proper_options.index("Sol")
        target_query = st.selectbox("Select Proper Name:", proper_options, index=default_index)

    elif search_type == "Bayer/Flamsteed":
        bf_options = stars.stars['bf']
        bf_options = bf_options[bf_options != ""].dropna().to_list()
        bf_options.sort()
        target_query = st.selectbox("Select Bayer/Flamsteed:", bf_options)

    elif search_type == "HIP":
        target_query = st.text_input("Type HIP number:")

    # Now 'target_query' contains the chosen star for your controller

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

    def nan_hider(value):
        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
            return "N/A"
        return value

    st.subheader("Star Details")

    with st.expander("ⓘ Field Explanations"):
        st.markdown("""
    :blue-background[**Name:**] Common name of the star.  
    :blue-background[**Bayer/Flamsteed:**] Historical designation based on position in constellation.  
    :blue-background[**HIP:**] Hipparcos catalog ID (precise position data).  
    :blue-background[**HD:**] Henry Draper spectral catalog ID.  
    :blue-background[**HR:**] Bright Star Catalog ID.  
    :blue-background[**GL:**] Gliese catalog number (usually nearby stars).  
    :blue-background[**Distance (pc):**] Distance from Earth in parsecs (1 pc = 3.262 ly).  
    :blue-background[**Magnitude:**] Brightness seen from Earth (lower = brighter).  
    :blue-background[**Absolute Magnitude:**] Intrinsic brightness if placed at 10 parsecs.  
    :blue-background[**Spectral Type:**] Classification by temperature and color (O → M).  
    :blue-background[**Mass:**] Compared to the Sun (1.0 = solar mass).  
    :blue-background[**Radius:**] Compared to the Sun’s radius.  
    :blue-background[**Temperature:**] Surface temperature (Kelvin).  
    :blue-background[**Constellation:**] Official sky region the star belongs to.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Name:** {nan_hider(target['proper'])}")
        st.write(f"**Bayer/Flamsteed:** {nan_hider(target['bf'])}")
        st.write(f"**HIP:** {nan_hider(target['hip'])}")
        st.write(f"**HD:** {nan_hider(target['hd'])}")
        st.write(f"**HR:** {nan_hider(target['hr'])}")
        st.write(f"**GL:** {nan_hider(target['gl'])}")
        st.write(f"**Distance (pc):** {nan_hider(target['dist'])}")
        predlum_val = train_model.input_predict(target.get("id"))
        st.markdown(f":blue-background[**Predicted Luminosity:**] {predlum_val:.5f} L☉")
        st.markdown(f":blue-background[**True Luminosity:**] {target['lum']} L☉")

    with col2:

        st.write(f"**Magnitude:** {nan_hider(target['mag'])} m")
        st.write(f"**Absolute Magnitude:** {nan_hider(target['absmag'])} M")
        st.write(f"**Spectral Type:** {nan_hider(target['spect'])}")
        st.write(f"**Mass:** {nan_hider(target['mass'])} M☉")
        st.write(f"**Radius:** {nan_hider(target['radius'])} R☉")
        st.write(f"**Temperature:** {nan_hider(target['temp'])} K")
        st.write(f"**Constellation:** {nan_hider(target['con'])}")

        predstel_class = train_model.stellar_classification(target['temp'], predlum_val)
        st.markdown(f":blue-background[**Predicted Stellar Classification:**] {predstel_class}")
        stel_class = train_model.stellar_classification(target['temp'], target['lum'])
        st.markdown(f":blue-background[**True Stellar Classification:**] {stel_class}")

    st.image("view/clustermodel_results.png", caption="Cluster model results")
    st.image("view/graphmodel_results.png", caption="Graph model results")
    st.image("view/tradmodel_results.png", caption="Traditional model results")






if __name__ == "__main__":
    view()
