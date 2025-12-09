#view.py
# The view.py file is responsible for displaying the UI in Streamlit
# including the Star Map.

import importlib.util   # This import is utilized for dynamically importing Python modules
                        # by file path. Added because importing files and methods just did not work.
import os               # For interacting with the filesystems
import sys              # To manipulate Python path
import streamlit as st  # Streamlit, aka the base app for the whole UI
import numpy as np      # Numpy for calculations and NaN

# This is for importing the modules one level above the current file's directory
# Basically, imports modules no matter where the script is run

# returns the directory portion of a path ↓                ↓ Gives the absolute path (no ../view.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This is the pathname of the file from which the module was loaded ↑
# It lets the machine know where that file is

# Ensure that PROJECT_ROOT is in sys.path so Python can import modules from it
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Function to import a Python module given a module name and a file path
def import_from_path(module_name, path):
    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location(module_name, path)
    # Create a new module object from the spec
    module = importlib.util.module_from_spec(spec)
    # Execute the module (runs its code, making its functions/classes available)
    spec.loader.exec_module(module)
    # Return the imported module object
    return module

# Dynamically import custom project modules using the helper function
# Each module provides specific functionality for the star map app

# StarData handles loading and managing the star database
StarData = import_from_path("star_data", os.path.join(PROJECT_ROOT, "model", "star_data.py")).StarData
# ConstellationData handles loading and managing constellation data
ConstellationData = import_from_path("constellation_data", os.path.join(PROJECT_ROOT, "model", "constellation_data.py")).ConstellationData
# Controller handles the logic between the data and the UI (MVC pattern)
Controller = import_from_path("controller", os.path.join(PROJECT_ROOT, "controller", "controller.py")).Controller
# StarsFactory is responsible for creating Plotly star plots
StarsFactory = import_from_path("stars_factory", os.path.join(PROJECT_ROOT, "model", "stars_factory.py")).StarsFactory

#      .----.     .----..--.      __.....__                  _________   _...._
#       \    \   /    / |__|  .-''         '.         _     _\        |.'      '-. .-.          .-
#        '   '. /'   /  .--. /     .-''"'-.  `. /\    \\   // \        .'```'.    '.\ \        / /
#        |    |'    /   |  |/     /________\   \`\\  //\\ //   \      |       \     \\ \      / /
#        |    ||    |   |  ||                  |  \`//  \'/     |     |        |    | \ \    / /
#        '.   `'   .'   |  |\    .-------------'   \|   |/      |      \      /    .   \ \  / /
#         \        /    |  | \    '-.____...---.    ',.--.      |     |\`'-.-'   .'     \ `  /
#          \      /     |__|  `.             .'     //    \     |     | '-....-'`        \  /
#           '----'              `''-...... -'       \\    /    .'     '.                 / /
#                                                    `'--'   '-----------'           |`-' /
#                                                                                     '..'
def view(train_model):
    # Set up the webpage before anything else
    st.set_page_config(
        page_title="₊ ⊹StarMap⭑𓂃.ᐟ",        # Tab's name
        page_icon="view/WebStarIcon.png",   # Tab's icon
        layout="wide"                       # Content stretches horizontally instead of staying narrow
    )
    st.title(":yellow[₊ ⊹]Star Map :yellow[⭑𓂃.ᐟ]")
    st.subheader("How it works")
    st.text("↪ Adjust the slider to view stars relative to the selected star at certain light years away\n"
            "↪ Select the star you want to view in the search box. You can write the proper name or select it.\n"
            "↪ For best results, make the graph fullscreen.\n"
            "↪ Zoom in or out of the graph to view the constellations and nearby stars.\n"
            "↪ Hover the mouse over a star to preview their information.")
    st.image("view/stars_divider.png")

    # View.py does not handle data, but it must display the data. So, we have helper functions from other files.
    stars = StarData()                     # → Loads all the information related to stars (position, name, bf, etc.)
    consts = ConstellationData()           # → Load the constellations lines and data (names, positions, etc.)
    controller = Controller(stars, consts) # → Connects everything.

    #             Create a slider with this prompt ↓ ...         min val ↓   max ↓    ↓ default value
    distance_ly = st.slider("Distance from target star (light years):", 100, 17000, 17000)
    # ↑ ... and store whatever value the user chooses in distance_ly variable.
    st.caption("Select 100 for less stars and 17000 for more stars.")
    # User guidance ↑

    # Radio to choose search type
    # ↓ save user input inside search_type
    search_type = st.radio(
        "Search by:", # Title of the radio buttons or instructions
        ("Proper Name", "Bayer/Flamsteed", "HIP") # Options
    )

    target_query = None # JIC

    # PROPER NAME OPTION
    if search_type == "Proper Name":
        proper_options = stars.stars['proper']
        # Gets the column 'proper' from the database
        proper_options = proper_options[proper_options != ""].dropna().to_list()
        #       Filter our missing values and empty strings ↑         ↑  Then convert it to a list.
        proper_options.sort()
        #               ↑ Sorts the list in alphabetical order
        default_index = proper_options.index("Sol")
        #                               ↑ Makes "Sol" the default selected star the first time
        #                                 the user sees the dropdown.            ↓ ... and put it here
        target_query = st.selectbox("Select Proper Name:", proper_options, index=default_index)
        # ↑             Create a dropdown menu where the user picks out or types in a proper star name.
        # | Then save it in the variable.

    # BAYER/FLAMSTEED OPTION
    elif search_type == "Bayer/Flamsteed":
        bf_options = stars.stars['bf']
        # Gets the column 'bf' from the database
        bf_options = bf_options[bf_options != ""].dropna().to_list()
        # Filter our missing values and empty strings ↑     ↑  Then convert it to a list.
        bf_options.sort()
        #          ↑ Sorts the list in alphabetical order
        target_query = st.selectbox("Select Bayer/Flamsteed:", bf_options)
        # ↑             Create a dropdown menu where the user picks out or types in a proper star name.
        # | Then save it in the variable.

    # HIP OPTION
    elif search_type == "HIP":
        target_query = st.text_input("Type HIP number:")
        # The user can type only (not dropdown menu)

    # Since is more user-friendly to see "light years" instead of "parsecs,"
    # I've made the slider in light years. But the database only has parsecs units. So we convert it.
    distance_pc = distance_ly / 3.26
    # parsecs = light years divided by 3.26

    #             ↓ this tries to find the chosen star in the database
    target = controller.get_target(target_query)
    if target is None: # If the star does not exist...
        st.error("Star not found.")
        return # ...the code stops and shows an error
    # This prevents the code from crashing.
    #                                                                ↓ HIP + the hip number in the database (HIP 02394)
    st.success(f"Centered on: {target['proper'] or target['bf'] or 'HIP ' + str(target['hip'])}")
    # If we do find the star, we show a success message with the proper name, or bf, or hip code

    filtered = controller.prepare_stars(target, distance_pc)
    # This gets a list of stars near the selected  ↑    one within the chosen distance (from slider above)
    #                                              |
    # ↓ this generates the 3D star map with the filter applied
    fig = controller.build_plot(filtered)

    st.plotly_chart(fig, width='stretch')
    # Then this displays it in the browser via Streamlit.

    # This function that takes a value, checks if it's empty, not a number, or missing, and then replaces it with "N/A"
    # Get the value ↓
    def nan_hider(value):
        #                    ↓ Cheks if the value is a float because only numbers can be NaN, Strings cannot.
        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        # ↑ if value is nothing OR the value is a number AND it is NaN OR the value is an empty string...
            value = "N/A" # Then make that invalid value to just be "N/A"
        return value # Finally, return the value.


    st.subheader("Star Details")

    # An expander is like a text but hidden unless the user clicks on it and expands it. This one
    # defines the different data the program displays.
    with st.expander("ⓘ Field Explanations"):
        # Choose markdown instead of write because I wanted to use a highlighter.
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

    # To better view the information, we split it into two columns.
    col1, col2 = st.columns(2)

    # Column 1 shows half the information...
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

    # ...column 2 shows the other half.
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
        # Note that everything highlighter belongs to the traditional model trained to display the information.

    st.divider() # Draws a horizontal like to divide the content from images
    st.image("view/clustermodel_results.png", caption="Cluster model results")
    st.image("view/graphmodel_results.png", caption="Graph model results")
    st.image("view/tradmodel_results.png", caption="Traditional model results")

# This makes it so that if this file is run instead of the main one, it can run correctly. Otherwise,
# this is untouched.
if __name__ == "__main__":
    view()