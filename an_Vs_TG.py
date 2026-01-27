import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
mpl.use('Agg')

# ------------------------------
# FUNZIONI
# ------------------------------

def read_coo_file(coo_file):
    """Legge il file .coo e restituisce un DataFrame con lat, lon, name, obs_path"""
    df = pd.read_csv(
        coo_file,
        sep=';',
        comment='#',
        header=None,
        usecols=[0,1,2,3],
        names=['lat', 'lon', 'name', 'obs_path']
    )
    return df

def read_obs_csv(obs_folder, hourly_mean=True):
    from pathlib import Path
    import pandas as pd
    import numpy as np

    obs_folder = Path(obs_folder)
    # supporta file singolo o cartella
    obs_files = [obs_folder] if obs_folder.is_file() else list(obs_folder.glob('*.csv'))
    obs_dict = {}

    for file_path in obs_files:
        try:
            times = []
            values = []

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # split massimo in 3 parti: tempo | ssh | resto
                    parts = line.split(' ', 2)
                    if len(parts) < 2:
                        continue

                    times.append(parts[0])
                    values.append(parts[1] if parts[1] != '' else np.nan)

            df = pd.DataFrame({'time': times, 'ssh_obs': values})
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ', utc=True, errors='coerce')
            df['ssh_obs'] = pd.to_numeric(df['ssh_obs'], errors='coerce')
            df = df.dropna(subset=['time','ssh_obs'])
            if df.empty:
                print(f"Obs error: Nessun dato valido in {file_path.name}")
                continue

            df = df.set_index('time')

            # media oraria centrata alla mezza ora
            if hourly_mean:
                df = df.resample('1H', label='right', closed='right').mean()
                df.index = df.index - pd.Timedelta(minutes=30)

            obs_dict[file_path.stem] = df

        except Exception as e:
            print(f"Obs error for {file_path.name}: {e}")

    # se c'è solo un file, restituisci direttamente il DataFrame
    if len(obs_dict) == 1:
        return list(obs_dict.values())[0]
    return obs_dict

def read_model_nc(nc_file, obs_index):
    """
    Legge il NetCDF del modello e taglia la serie al periodo delle osservazioni.
    """
    ds = xr.open_dataset(nc_file)
    ssh = ds['sossheig']  # shape (time, lat, lon)
    
    time = pd.to_datetime(ds['time_counter'].values)
    time = time.tz_localize('UTC')

    ssh_vals = ssh[:,0,0].values
    df_mod = pd.DataFrame({'ssh_mod': ssh_vals}, index=time)

    # taglio al periodo delle osservazioni
    start, end = obs_index.min(), obs_index.max()
    df_mod_cut = df_mod.loc[start:end]

    return df_mod_cut

import pandas as pd

def plot_tg(name, obs, mod, outdir):
    df = obs.join(mod, how='inner')
    if df.empty:
        print(f'Nessuna sovrapposizione temporale per {name}')
        return None

    # offset media
    offset = df['ssh_obs'].mean() - df['ssh_mod'].mean()
    df['ssh_mod_offset'] = df['ssh_mod'] + offset

    # definisci intervallo evento
    event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
    event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
    df_event = df.loc[event_start:event_end]

    # calcola massimi nell'evento
    max_obs = df_event['ssh_obs'].max()
    max_mod = df_event['ssh_mod_offset'].max()

    # parametri stile
    fontsize = 20
    linewidth = 3
    legend_fontsize = 20

    plt.figure(figsize=(12,6))

    # sfondo grigio evento
    plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)

    # plot linee con massimo nell'evento in legenda
    plt.plot(df.index, df['ssh_obs'], 
             label=f'OBS (max event {max_obs:.3f} m)', lw=linewidth)
    plt.plot(df.index, df['ssh_mod_offset'], 
             label=f'MODEL (max event {max_mod:.3f} m, offset {offset:.3f} m)', lw=linewidth)

    plt.title(name, fontsize=fontsize)
    plt.ylabel('Sea level [m]', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)

    # legenda in alto a sinistra, leggermente trasparente
    plt.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.7)

    # assi più leggibili
    plt.xticks(fontsize=fontsize-2, rotation=30)
    plt.yticks(fontsize=fontsize-2)
    plt.grid()
    plt.tight_layout()

    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / f'{name}_obs_vs_mod.png', dpi=150)
    plt.close()

    return df

def plot_tg_map(df_coo, outdir, figsize=(14,8)):
    """
    Plot della mappa del Mediterraneo con tutti i punti delle tide-gauges.
    df_coo deve contenere colonne 'lat', 'lon', 'name'.
    L'estensione è leggermente tagliata a ovest e a est per zoomare sul bacino centrale,
    ma include tutte le TG.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Calcolo i limiti a partire dai dati per includere tutte le TG
    min_lon = df_coo['lon'].min() - 1
    max_lon = df_coo['lon'].max() + 1
    min_lat = df_coo['lat'].min() - 1
    max_lat = df_coo['lat'].max() + 1
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Terreno e coste
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Punti e nomi TG
    for _, row in df_coo.iterrows():
        lat = row['lat']
        lon = row['lon']
        name = row['name']
        ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
        ax.text(lon+0.2, lat, name, fontsize=16, transform=ccrs.PlateCarree())
    
    # Titolo aggiornato
    plt.title('Tide-Gauges', fontsize=20)
    
    # Salva
    outdir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outdir / 'tg_map_mediterranean.png', dpi=150)
    plt.close()

def compute_stats(df, name):
    """Calcola statistiche di base"""
    from sklearn.metrics import mean_squared_error
    obs = df['ssh_obs']
    mod = df['ssh_mod_offset']

    rmse = np.sqrt(mean_squared_error(obs, mod))
    corr = obs.corr(mod)
    offset = mod.mean() - df['ssh_mod'].mean()
    n_points = len(df)

    return {
        'name': name,
        'n_points': n_points,
        'obs_mean': obs.mean(),
        'mod_mean': df['ssh_mod'].mean(),
        'offset_applied': offset,
        'rmse': rmse,
        'corr': corr
    }

# ------------------------------
# MAIN
# ------------------------------

def main():
    coo_file = 'TGs_harry.coo'
    mod_dir = Path('/work/cmcc/ag15419/harry/mod_extr/')
    outdir = Path('/work/cmcc/ag15419/harry/plot_ts/')
    stats_list = []

    print ('TGs file:',coo_file)
    print ('Model dir:',mod_dir)
    print ('Output dir:',outdir)

    df_coo = read_coo_file(coo_file)

    for _, row in df_coo.iterrows():
        name = row['name']
        obs_path = row['obs_path']
        print(f'Processing {name}')

        # OBS
        try:
            obs = read_obs_csv(obs_path)
        except Exception as e:
            print(f"Obs error: {e}")
            continue
        print(f"{name} OBS time range: {obs.index.min()} → {obs.index.max()}")  # <-- stampa range obs

        # MODEL
        nc_file = mod_dir / f'{name}_mod_MedFS_analysis_26012026.nc'
        if not nc_file.exists():
            print(f"Missing model file for {name}")
            continue
        mod = read_model_nc(nc_file, obs.index)
        print(f"{name} MODEL time range: {mod.index.min()} → {mod.index.max()}")  # <-- stampa range modello

        # PLOT e statistiche
        df_plot = plot_tg(name, obs, mod, outdir)
        plot_tg_map(df_coo, outdir)
        if df_plot is not None:
            stats = compute_stats(df_plot, name)
            stats_list.append(stats)

    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats.to_csv(outdir / 'tg_obs_vs_mod_stats.csv', index=False)
        print("Statistiche salvate in tg_obs_vs_mod_stats.csv")
    else:
        print("Nessuna statistica calcolata")

if __name__ == '__main__':
    main()
