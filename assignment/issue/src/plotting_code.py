import altair as alt
import torch
import pandas as pd

alt.data_transformers.enable('default', max_rows=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_density_chart(P, d=7.0, step=0.1, save_path=None):
    xv, yv = torch.meshgrid([
        torch.arange(-d, d, step), 
        torch.arange(-d, d, step)
    ])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()
    
    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),
    })
    
    chart = alt.Chart(df).mark_point().encode(
        x='x:Q',
        y='y:Q',
        color=alt.Color('p:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['x','y','p']
    )
    
    if save_path:
        chart.save(save_path)
    
    return chart

def get_particles_chart(X, save_path=None):
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
    })

    chart = alt.Chart(df).mark_circle(color='red').encode(
        x='x:Q',
        y='y:Q'
    )
    
    if save_path:
        chart.save(save_path)
    
    return chart

if __name__ == "__main__":
    pass