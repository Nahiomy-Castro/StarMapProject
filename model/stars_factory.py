# stars_factory.py
#                              _..._             .-'''-.
#                           .-'_..._''.         '   _    \
#                         .' .'      '.\      /   /` '.   \                       _________   _...._
#          _.._          / .'                .   |     \  '       .-.          .- \        |.'      '-. .-.          .-
#        .' .._|        . '               .| |   '      |  '.-,.--.\ \        / /  \        .'```'.    '.\ \        / /
#        | '       __   | |             .' |_\    \     / / |  .-. |\ \      / /    \      |       \     \\ \      / /
#      __| |__  .:--.'. | |           .'     |`.   ` ..' /  | |  | | \ \    / /      |     |        |    | \ \    / /
#     |__   __|/ |   \ |. '          '--.  .-'   '-...-'`   | |  | |  \ \  / /       |      \      /    .   \ \  / /
#        | |   `" __ | | \ '.          .|  |                | |  '-    \ `  /,.--.   |     |\`'-.-'   .'     \ `  /
#        | |    .'.''| |  '. `._____.-'/|  |                | |         \  ///    \  |     | '-....-'`        \  /
#        | |   / /   | |_   `-.______ / |  '.'              | |         / / \\    / .'     '.                 / /
#        | |   \ \._,\ '/            `  |   /               |_|     |`-' /   `'--''-----------'           |`-' /
#        |_|    `--'  `"                `'-'                         '..'                                  '..'

import numpy as np                  # Handles math
import plotly.graph_objects as go   # Used to draw the 3D universe
import pandas as pd                 # Used for the dataframes
import streamlit as st              # Used for the cache

# Stars vary in size. We want to show the user a diverse range of sizes without
# overwhelming the graph. Stars are MASSIVE and would cover the whole screen
# and some (like the Sun) are a dot compared to
# supermassive stars. So we scaled them with this method.
def scale_size_from_luminosity(lum):
    if pd.isna(lum): # Check if the luminosity value is missing from the star
        return 3  # IF it is missing, we give the star a default of 3 in the scale.
    # Otherwise, we return this value:
    return max(2, min(20, np.log10(lum + 1) * 4))
# Some stars are five times brighter than the Sun while others are a million times
# brighter than the Sun. The log compresses extreme luminosity values while keeping
# visible differences between less bright and more bright stars. And Fun Fact! Humans
# do not experience light in a linear way. If light doubled in intensity, you don't experience it
# as "twice as bright." Instead, the brain compresses brightness using something close
# to logarithmics. If we treated brightness linearly, stepping outside on a sunny day would
# blind you instantly every time. BOOM FLASHBANG!!!

# This class is responsible for building the plot.
class StarsFactory:
    @staticmethod
    # I placed a static method decorator to avoid making an object
    # and simply call it directly. Plus, this method doesn't hold any data.
    # It's more like a builder based on data shared
    def create_plot(plot_type, stars_filtered, constellations=None):
        # This is just a decision maker. A bit hardcoded but can be changed.
        # It directly links to controller.py build_plot method.
        if plot_type == "stars_only":
            # Shows just points of the stars
            return StarsFactory.scatter3d(stars_filtered)
        elif plot_type == "constellations":
            # Draws stars and lines connecting them into constellations (currently selected)
            return StarsFactory.constellation_plot(stars_filtered, constellations)
        else:
            # Anything else in the code raises an error.
            raise ValueError(f"Error: Unknown plot type: {plot_type}\n "
                             f"plot_type must be 'stars_only' or 'constellations'")

    @staticmethod
    @st.cache_data
    # Cache the function outputs so repeated calls (like changing distance but not target) load faster.
    # This function is in charge of placing the stars in their respective coordinates
    def scatter3d(stars_filtered):
        #               ↓ builds a 3D scatter from Plotly called star_scatter
        star_scatter = go.Scatter3d(
            # Remember in controller.py the function prepare_stars? That function was in charge
            # of searching the x, y and z coordinates of the target star in the database,
            # storing it, making it the origin, copying the data and then returning that copied data
            # of the coordinates. Now here, we are using those copied coordinates to plot it in the graph.
            x=stars_filtered['cx_scaled'],
            y=stars_filtered['cy_scaled'],
            z=stars_filtered['cz_scaled'],
            mode='markers', # Plot with dots
            # The dictionary configures how each star with look!
            marker=dict(
                size=stars_filtered['marker_size'], # Based on luminosity so stars with higher lum appears bigger
                color=stars_filtered['marker_color'], # Based on marker_size, bigger stars = brighter colors
                colorscale='viridis', # A purple, blue, green and yellow colormap
                opacity=0.5, # the opacity of each star,
                # made half-transparent so user can view all stars without obstruction
                cmin=0, cmax=1 # Normalized color values
            ),
            # We need to display the name or code of the star when the user hover their mouse
            # so we ue this for the label.
            text=stars_filtered.apply(
                # In the stars_filtered, we "apply" this search or one-line method with lambda
                # We don't want a whole column of data, just the best name. So lambda
                # goes "for this row, pick the best name (proper OR bf OR hipp)
                lambda r: r['proper'] or r['bf'] or f"HIP {r['hip']}", axis=1
                #                                                        ↑ apply it row by row
            ),
            # Now we apply it to our hover
            hovertemplate="%{text}",
        )
        # We create a new graph, but this time we input the stars points we made earlier.
        #           ↱ like a blank canvas now suddenly filled with a starfield!
        fig = go.Figure(data=[star_scatter])
        # In here we control background colors, axis labels, font colors, window size, etc.
        #    ↱ Changes the aesthetics of the figure we just made
        fig.update_layout(
            # ↓ refers to the 3D plot or space where the stars are. Like a stage
            scene=dict(
                # In here we label each axis as their respective letter (x, y, z)
                # and eliminate any background from the specific axis.
                xaxis=dict(title='X', showbackground=False),
                yaxis=dict(title='Y', showbackground=False),
                zaxis=dict(title='Z', showbackground=False),
                # We only want to show the black background of the graph itself
                bgcolor='black'
            ),
            paper_bgcolor='black', # This is the whole page outside the 3D scene. We also make it black (no borders)
            font=dict(color='white'), # We make all text labels white for good contrast against the dark background
            height=800, # the figure will be 800 pixels tall, but gets overrided with max screen option
            showlegend=False # I want to maintain the aesthetics of the night sky, so I hide the legend.
        )
        return fig # We finished customizing our figure, so we return it.

    @staticmethod
    # This method makes a 3D plot showing stars and constellations
    def constellation_plot(stars_filtered, constellations):
        # We create a dictionary tying the hip number to a star and its coordinates. This is
        # necessary since the JSON file that has the constellation works by tracing a line without lifting the pencil
        # from star to star using their hip numbers only.
        hip_to_star = {
            #     ↓ the star's hip # converted to int
            int(row['hip']): (row['cx_scaled'], row['cy_scaled'], row['cz_scaled'])
            #                ◟_________________________↧__________________________◞
            #                         a tuple with the star's coordinates
            for _, row in stars_filtered.iterrows() # ← loop over each row in the table of stars
            #   ↑ that's a throwaway variable. Its value is not used. We don't care about it.
            if pd.notna(row['hip'])
            # Only include stars that have HIP # and skip the missing ones.
        }

        # Create an empty set.
        const_hips = set()
        #             ↑ An empty set is like a list but without duplicates
        for const in constellations:
            # Loop through each constellation
            for line in const["lines"]:
                # Each constellation has "lines" (in the json file), so we loop through these lines
                for hip in line:
                    # Each line is a list of HIP numbers (in the json file) of stars it connects. We loop through them.
                    const_hips.add(hip)
                    # We then add this HIP number to our set.
        # Now, we finally have all the HIP numbers of stars that are part of constellations, without duplicates
        # inside const_hips or the set.

        stars_filtered['in_constellation'] = stars_filtered['hip'].apply(
            # We then add a new column to the star table and run a small lambda "function" on each
            # star's HIP.                   ↓ if the HIP exists, check...
            lambda x: x in const_hips if pd.notna(x) else False #     ← If the HIP is missing, mark false.
            #         ↑ ... if this star is part of any constellation, False otherwise.
        )   # Now every star has a column showing whether it's in a constellation or not.

        #                  ↓ How big the star will be based on scaled luminosity
        stars_filtered['marker_size'] = stars_filtered['lum'].apply(scale_size_from_luminosity)
        #                                                ↑  We use the star's luminosity and pass it
        #                                                |  through our lum check method

        #                                                  ↓ We use the computed marker size and divide it by 20
        stars_filtered['marker_color'] = stars_filtered['marker_size'] / 20
        #                                                    that way we have a # between 0 - 1 for the colormap

        # We create a Plotly graph to draw points in 3D (again, I know)
        star_scatter = go.Scatter3d(
            x=stars_filtered['cx_scaled'],
            y=stars_filtered['cy_scaled'],
            z=stars_filtered['cz_scaled'],
            mode='markers',
            marker=dict(
                size=stars_filtered['marker_size'],
                color=stars_filtered['marker_color'],
                colorscale='viridis',
                opacity=0.5,
                cmin=0, cmax=1,
                showscale=False
            ),
            text=stars_filtered.apply(
                lambda r: r['proper'] or r['bf'] or f"HIP {r['hip']}", axis=1
            ),
            hovertemplate="%{text}<extra></extra>", # This is HTML. Is to avoid Plotly to put stuff in my hover text
            showlegend=False
        )

        line_traces = [] # will store the 3D lines connecting the stars in the constellations
        label_traces = [] # will store text labels for each constellation so we can show their names

        for const in constellations:
            #           ↑ a list of constellations (JSON)
            for line in const["lines"]:
                # Goes through each line
                coords = [hip_to_star[hip] for hip in line if hip in hip_to_star]
                # for each HIP # in the line, get its (x ,y, z) coordinates only if that
                # HIP exists in the hip_to_star mapping

                if len(coords) > 1: # IF we have more than 1 star to draw a line with...
                    #... then
                    # we make three empty lists to store all sequences of x, y, and z separately
                    xs = []
                    ys = []
                    zs = []

                    for c in coords:
                        # go star by star in the constellation line
                        x, y, z = c # each "c" is a tuple (x,y,z). This line unpacks it into
                        # separate x,y,z variables.
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

                    line_traces.append(
                        go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            mode='lines',
                            line=dict(color='white', width=3),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
            coords = [hip_to_star[hip] for line in const["lines"]
                      for hip in line if hip in hip_to_star]
            if coords:
                xs = []
                ys = []
                zs = []
                for c in coords:
                    x, y, z = c
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

                label_traces.append(
                    go.Scatter3d(
                        x=[np.mean(xs)],
                        y=[np.mean(ys)],
                        z=[np.mean(zs)],
                        mode='text',
                        text=[const.get("common_name", {}).get("english", const.get("iau", ""))],
                        textfont=dict(color="yellow", size=12),
                        hoverinfo='none',
                        showlegend=False
                    )
                )

        fig = go.Figure(data=[star_scatter] + line_traces + label_traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', showbackground=False),
                yaxis=dict(title='Y', showbackground=False),
                zaxis=dict(title='Z', showbackground=False),
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font=dict(color='white'),
            height=800,
            showlegend=False
        )
        return fig
