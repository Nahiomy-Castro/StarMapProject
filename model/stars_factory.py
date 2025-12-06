# stars_factory.py
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def scale_size_from_luminosity(lum):
    """Scale luminosity to marker size for plotting."""
    if pd.isna(lum):
        return 3  # fallback size
    return max(2, min(20, np.log10(lum + 1) * 4))  # log scale

class StarsFactory:
    @staticmethod
    def create_plot(plot_type, stars_filtered, constellations=None):
        if plot_type == "stars_only":
            return StarsFactory._scatter3d(stars_filtered)
        elif plot_type == "constellations":
            return StarsFactory._constellation_plot(stars_filtered, constellations)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    @staticmethod
    @st.cache_data
    def _scatter3d(stars_filtered):
        star_scatter = go.Scatter3d(
            x=stars_filtered['cx_scaled'],
            y=stars_filtered['cy_scaled'],
            z=stars_filtered['cz_scaled'],
            mode='markers',
            marker=dict(
                size=stars_filtered['marker_size'],
                color=stars_filtered['marker_color'],
                colorscale='viridis',
                opacity=0.8,
                colorbar=dict(title='Brightness'),
                cmin=0, cmax=1
            ),
            text=stars_filtered.apply(
                lambda r: r['proper'] or r['bf'] or f"HIP {r['hip']}", axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )
        fig = go.Figure(data=[star_scatter])
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

    @staticmethod
    def _constellation_plot(stars_filtered, constellations):
        # Map HIP → coords
        hip_to_star = {
            int(row['hip']): (row['cx_scaled'], row['cy_scaled'], row['cz_scaled'])
            for _, row in stars_filtered.iterrows()
            if pd.notna(row['hip'])
        }

        # Marker sizes/colors based on luminosity
        const_hips = set()
        for const in constellations:
            for line in const["lines"]:
                for hip in line:
                    const_hips.add(hip)

        stars_filtered['in_constellation'] = stars_filtered['hip'].apply(
            lambda x: x in const_hips if pd.notna(x) else False
        )
        stars_filtered['marker_size'] = stars_filtered['lum'].apply(scale_size_from_luminosity)
        stars_filtered['marker_color'] = stars_filtered['marker_size'] / 20  # optional brightness mapping

        # Scatter
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
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )

        # Lines & labels
        line_traces = []
        label_traces = []
        for const in constellations:
            for line in const["lines"]:
                coords = [hip_to_star[hip] for hip in line if hip in hip_to_star]
                if len(coords) > 1:
                    xs, ys, zs = zip(*coords)
                    line_traces.append(
                        go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            mode='lines',
                            line=dict(color='white', width=3),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
            # label
            coords = [hip_to_star[hip] for line in const["lines"]
                      for hip in line if hip in hip_to_star]
            if coords:
                xs, ys, zs = zip(*coords)
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
