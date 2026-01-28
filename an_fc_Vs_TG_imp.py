import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.fft as F
import scipy as sp
from scipy import signal
mpl.use('Agg')

# ------------------------------
# FUNZIONI GENERICHE
# ------------------------------

def read_coo_file(coo_file):
    df = pd.read_csv(
        coo_file,
        sep=';',
        comment='#',
        header=None,
        usecols=[0,1,2,3],
        names=['lat', 'lon', 'name', 'obs_path']
    )
    return df


from pathlib import Path
import pandas as pd

def read_obs_csv(obs_folder, hourly_mean=True, interpolate_gaps=True, verbose=True):
    """
    Legge file CSV di osservazioni di sea-level e restituisce dataframe con indice temporale.
    
    Parametri:
    -----------
    obs_folder : str o Path
        Cartella contenente i CSV o singolo file CSV.
    hourly_mean : bool
        Se True, resample a frequenza oraria.
    interpolate_gaps : bool
        Se True, interpolazione lineare delle ore mancanti dopo il resample.
    verbose : bool
        Se True, stampa info diagnostica.

    Ritorna:
    --------
    dict di DataFrame se più file, altrimenti singolo DataFrame.
    """
    obs_folder = Path(obs_folder)
    obs_files = [obs_folder] if obs_folder.is_file() else list(obs_folder.glob('*.csv'))
    obs_dict = {}

    for file_path in obs_files:
        try:
            # -----------------------------
            # Lettura dati
            # -----------------------------
            times, values = [], []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue
                    times.append(parts[0])
                    values.append(parts[1])

            df = pd.DataFrame({'time': times, 'ssh_obs': values})
            df['time'] = pd.to_datetime(
                df['time'],
                format='%Y-%m-%dT%H:%M:%SZ',
                utc=True,
                errors='coerce'
            )
            df['ssh_obs'] = pd.to_numeric(df['ssh_obs'], errors='coerce')
            df = df.dropna(subset=['time', 'ssh_obs'])
            df = df.set_index('time')

            # -----------------------------
            # Diagnostica duplicati
            # -----------------------------
            n_dup = df.index.duplicated().sum()
            if verbose and n_dup > 0:
                print(f"[OBS WARNING] {file_path.name}: {n_dup} duplicated timestamps found")

            df = df[~df.index.duplicated(keep='last')]

            # -----------------------------
            # Ordina tempo
            # -----------------------------
            df = df.sort_index()

            # -----------------------------
            # Info step temporale
            # -----------------------------
            if verbose:
                dt_counts = df.index.to_series().diff().value_counts().head(3)
                print(f"[OBS INFO] {file_path.name} time-step (top):")
                print(dt_counts)

            # -----------------------------
            # Resample orario
            # -----------------------------
            if hourly_mean:
                df = df.resample('1H', label='right', closed='right').mean()
                df.index = df.index - pd.Timedelta(minutes=30)

                # -----------------------------
                # Interpolazione dei gap
                # -----------------------------
                if interpolate_gaps:
                    df['ssh_obs'] = df['ssh_obs'].interpolate(method='time', limit_direction='both')

            obs_dict[file_path.stem] = df

        except Exception as e:
            print(f"[OBS ERROR] {file_path.name}: {e}")

    if len(obs_dict) == 1:
        return list(obs_dict.values())[0]

    return obs_dict


def read_model_nc(nc_file, obs_index=None):
    ds = xr.open_dataset(nc_file)
    ssh = ds['sossheig']  # (time, lat, lon)
    time = pd.to_datetime(ds['time_counter'].values).tz_localize('UTC')
    ssh_vals = ssh[:,0,0].values
    df_mod = pd.DataFrame({'ssh_mod': ssh_vals}, index=time)
    if obs_index is not None:
        start, end = obs_index.min(), obs_index.max()
        df_mod = df_mod.loc[start:end]
    return df_mod

# ------------------------------
# FUNZIONI PLOT
# ------------------------------

def plot_tg(name, obs, mod, outdir):
    # Usa solo dati resample orario
    df = obs.join(mod, how='inner')
    if df.empty:
        print(f'Nessuna sovrapposizione temporale per {name}')
        return None

    # Allinea media MODEL a OBS
    offset = df['ssh_obs'].mean() - df['ssh_mod'].mean()
    df['ssh_mod_offset'] = df['ssh_mod'] + offset

    # Seleziona periodo evento per annotazioni
    event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
    event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
    df_event = df.loc[event_start:event_end]
    #xlim_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
    #xlim_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')

    # Check valori validi nell'evento
    if df_event['ssh_obs'].notna().any():
        idx_max_obs = df_event['ssh_obs'].idxmax()
        max_obs = df_event['ssh_obs'].max()
    else:
        idx_max_obs = None
        max_obs = None

    if df_event['ssh_mod_offset'].notna().any():
        idx_max_mod = df_event['ssh_mod_offset'].idxmax()
        max_mod = df_event['ssh_mod_offset'].max()
    else:
        idx_max_mod = None
        max_mod = None

    # --- PLOT ---
    fontsize, linewidth, legend_fontsize = 20, 3, 18
    plt.figure(figsize=(12,6))
    plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
    
    # Plotta solo dati resample orario
    plt.plot(df.index, df['ssh_obs'], color='tab:orange',
             lw=linewidth,
             label=f'OBS (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})' if max_obs is not None else 'OBS (no data)')
    plt.plot(df.index, df['ssh_mod_offset'], color='blue',
             lw=linewidth,
             label=f'ANALYSIS (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})' if max_mod is not None else 'ANALYSIS (no data)')
    
    plt.title(name, fontsize=fontsize)
    plt.ylabel('Sea level [m]', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.7)
    plt.xticks(fontsize=fontsize-2, rotation=30)
    plt.yticks(fontsize=fontsize-2)
    #plt.xlim(xlim_start, xlim_end)
    plt.grid()
    plt.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / f'{name}_obs_vs_mod.png', dpi=150)
    plt.close()
    return df


def plot_tg_map(df_coo, outdir, figsize=(20,7)):
    """
    Mappa con tutte le TG sul Mediterraneo.
    """
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 20
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Estensione per tutto il Mediterraneo
    min_lon, max_lon = -6, 37   # Ovest: Marocco/Spagna, Est: Levante/Cipro
    min_lat, max_lat = 30, 46   # Sud: Nord Africa, Nord: coste francesi e Balcani
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    for _, row in df_coo.iterrows():
        lat, lon, name = row['lat'], row['lon'], row['name']
        ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
        ax.text(lon+0.2, lat, name, fontsize=16, transform=ccrs.PlateCarree())

    plt.title('Tide-Gauges – Mediterranean', fontsize=20)
    outdir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outdir / 'tg_map_mediterranean.png', dpi=150)
    plt.close()

def plot_tg_map_single(df_coo, tg_name, outdir, figsize=(10,3)):
    """
    Mappa per slide: tutto il Mediterraneo, ma evidenzia solo la TG selezionata.
    """
    row = df_coo[df_coo['name'] == tg_name]
    if row.empty:
        print(f"[MAP WARNING] TG {tg_name} non trovata")
        return

    lat, lon = row.iloc[0]['lat'], row.iloc[0]['lon']

    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 14
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Stesso dominio del Mediterraneo
    min_lon, max_lon = -6, 37
    min_lat, max_lat = 30, 46
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Evidenzia solo la TG
    ax.plot(lon, lat, 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax.text(lon+0.2, lat, tg_name, fontsize=16, transform=ccrs.PlateCarree())

    plt.title(f'Tide-Gauge: {tg_name}', fontsize=20)
    outdir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outdir / f'{tg_name}_map.png', dpi=150)
    plt.close()

def compute_stats(df, name):
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
# DETIDING / SPETTRI
# ------------------------------

def fft2bands(nelevation, low_bound=1./20.0, high_bound=1./30.0,
              low_bound_1=1/11.2, high_bound_1=1/13.5, alpha=0.4, invert='False'):
    if len(nelevation) % 2:
        result = F.rfft(nelevation, len(nelevation))
    else:
        result = F.rfft(nelevation)
    freq = F.fftfreq(len(nelevation))[:result.shape[0]]
    factor = np.ones_like(result)
    sl = np.logical_and(high_bound < freq, freq < low_bound)
    sl_2 = np.logical_and(high_bound_1 < freq, freq < low_bound_1)
    a = factor[sl]
    b = factor[sl_2]
    lena = a.shape[0]
    lenb = b.shape[0]
    a = 1 - sp.signal.tukey(lena, alpha)
    b = 1 - sp.signal.tukey(lenb, alpha)
    factor[sl] = a[:lena]
    factor[sl_2] = b[:lenb]
    if invert=='False':
        result = result * factor
    else:
        result = result * (-(factor-1))
    relevation = F.irfft(result, len(nelevation))
    return relevation, np.abs(factor)

def plot_detided_ts(name, df, outdir):
    fontsize, linewidth, legend_fontsize = 20, 3, 18

    # Allinea media MODEL detided a OBS detided
    offset_detided = df['ssh_obs_detided'].mean() - df['ssh_mod_offset_detided'].mean()
    df['ssh_mod_offset_detided_aligned'] = df['ssh_mod_offset_detided'] + offset_detided

    # Seleziona periodo evento
    event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
    event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
    df_event = df.loc[event_start:event_end]

    # OBS detided
    if df_event['ssh_obs_detided'].notna().any():
        idx_max_obs = df_event['ssh_obs_detided'].idxmax()
        max_obs = df_event['ssh_obs_detided'].max()
        obs_label = f'OBS detided (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})'
    else:
        idx_max_obs = None
        max_obs = None
        obs_label = 'OBS detided (no valid data)'

    # MODEL detided
    if df_event['ssh_mod_offset_detided_aligned'].notna().any():
        idx_max_mod = df_event['ssh_mod_offset_detided_aligned'].idxmax()
        max_mod = df_event['ssh_mod_offset_detided_aligned'].max()
        mod_label = f'ANALYSIS detided (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})'
    else:
        idx_max_mod = None
        max_mod = None
        mod_label = 'ANALYSIS detided (no valid data)'

    # --- PLOT ---
    plt.figure(figsize=(12,6))
    plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)

    # Plotta solo dati resample orario
    plt.plot(df.index, df['ssh_obs_detided'], color='tab:orange', lw=linewidth, label=obs_label)
    plt.plot(df.index, df['ssh_mod_offset_detided_aligned'], color='blue', lw=linewidth, label=mod_label)

    plt.title(f'{name} – Detided sea level', fontsize=fontsize)
    plt.ylabel('Sea level [m]', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.7)
    plt.xticks(fontsize=fontsize-2, rotation=30)
    plt.yticks(fontsize=fontsize-2)
    plt.grid()
    plt.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / f'{name}_detided_ts.png', dpi=150)
    plt.close()

def compute_spectrum_amp(ts, dt_hours):
    ts = ts - np.nanmean(ts)
    n = len(ts)
    fft = np.fft.rfft(ts)
    amp = np.abs(fft)/n*2
    freq = np.fft.rfftfreq(n, d=dt_hours*3600)
    period = 1/freq/3600
    return period[1:], amp[1:]

def plot_spectra(name, df, outdir, dt_hours=1.0):
    fontsize = 20
    per_obs, amp_obs = compute_spectrum_amp(df['ssh_obs'].values, dt_hours)
    per_mod, amp_mod = compute_spectrum_amp(df['ssh_mod_offset'].values, dt_hours)
    per_obs_d, amp_obs_d = compute_spectrum_amp(df['ssh_obs_detided'].values, dt_hours)
    per_mod_d, amp_mod_d = compute_spectrum_amp(df['ssh_mod_offset_detided'].values, dt_hours)
    plt.figure(figsize=(12,6))
    plt.semilogx(per_obs, amp_obs, label='OBS')
    plt.semilogx(per_mod, amp_mod, label='MODEL')
    plt.semilogx(per_obs_d, amp_obs_d, '--', label='OBS detided')
    plt.semilogx(per_mod_d, amp_mod_d, '--', label='MODEL detided')
    plt.xlabel('Period [h]', fontsize=fontsize)
    plt.ylabel('Amplitude [m]', fontsize=fontsize)
    plt.title(f'{name} – Amplitude spectra', fontsize=fontsize)
    plt.legend(fontsize=fontsize-2, framealpha=0.7)
    plt.grid(which='both')
    plt.xlim(2, 100)
    plt.axvline(12, color='gray', linestyle='--', lw=2)
    plt.axvline(24, color='gray', linestyle='--', lw=2)
    plt.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / f'{name}_spectra_amp.png', dpi=150)
    plt.close()


def combine_all_plots(df_coo, outdir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    for _, row in df_coo.iterrows():
        name = row['name']
        
        # Percorsi delle figure già prodotte
        map_file = outdir / f"{name}_map.png"
        obs_vs_mod_file = outdir / f"{name}_obs_vs_mod.png"
        detided_file = outdir / f"{name}_detided_ts.png"
        forecast_file = outdir / f"{name}_forecast_all_original.png"
        forecast_detided_file = outdir / f"{name}_forecast_all_detided.png"
        
        # Nuova figura
        fig = plt.figure(figsize=(12,10))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1,1,1], width_ratios=[1,1])
        
        # --- Mappa in alto (centro) ---
        ax_map = fig.add_subplot(gs[0, :])
        map_img = plt.imread(map_file)
        ax_map.imshow(map_img)
        ax_map.axis('off')  # togli assi
        
        # --- 2x2 plot sotto ---
        ax1 = fig.add_subplot(gs[1,0])
        img1 = plt.imread(obs_vs_mod_file)
        ax1.imshow(img1)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1,1])
        img2 = plt.imread(detided_file)
        ax2.imshow(img2)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[2,0])
        img3 = plt.imread(forecast_file)
        ax3.imshow(img3)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[2,1])
        img4 = plt.imread(forecast_detided_file)
        ax4.imshow(img4)
        ax4.axis('off')
        
        # Ridurre gli spazi bianchi tra i subplot
        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.97,
            bottom=0.03,
            hspace=0.05,
            wspace=0.05
        )
        
        # Salva figura combinata
        fig_file = outdir / f"{name}_combined.png"
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved: {fig_file}")


# ------------------------------
# MAIN
# ------------------------------

def main():
    coo_file = 'TGs_harry.coo'
    mod_dir = Path('/work/cmcc/ag15419/harry/mod_extr/')
    outdir = Path('/work/cmcc/ag15419/harry/plot_ts_fc/')
    stats_list = []

    semid_tides_band = [11.2, 17.0]
    diurnal_tides_band = [21.0, 32.0]

    df_coo = read_coo_file(coo_file)

    for _, row in df_coo.iterrows():
        name, obs_path = row['name'], row['obs_path']
        print(f'Processing {name}')

        try:
            obs = read_obs_csv(obs_path)
            # -------------------------
            # Controllo resample e NaN
            # -------------------------
            print(f"Controllo ultimi valori OBS per {name}:")
            print(obs.tail(20))
            print("NaN presenti nelle ultime 20 righe?", obs.isna().tail(20).any())

        except Exception as e:
            print(f"Obs error: {e}")
            continue

        # -------------------------------
        # CONTROLLO QUALITÀ DATI OBS
        # -------------------------------
        print(f"\nControllo dati OBS per {name}:")
    
        # leggere dati raw senza resample
        df_raw = read_obs_csv(obs_path, hourly_mean=False)
        print("Numero campioni totali:", len(df_raw))
        print("Duplicati timestamps:", df_raw.index.duplicated().sum())
    
        # dopo il resample orario
        print("Numero campioni dopo resample orario:", len(obs))
        print("Conteggio per ora (prime 10):")
        print(obs['ssh_obs'].resample('1H').count().head(10))  # prime 10 ore come esempio
    
        # controllo valori nell'intervallo dell'evento
        event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
        event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
        df_event = obs.loc[event_start:event_end]
        print(f"Numero valori validi nell'evento ({event_start} -> {event_end}): {df_event['ssh_obs'].notna().sum()}")

        # MODEL
        nc_file = mod_dir / f'{name}_mod_MedFS_analysis_26012026.nc'
        if not nc_file.exists(): 
            print(f"Missing model file for {name}")
            continue
        mod = read_model_nc(nc_file, obs.index)

        df_plot = plot_tg(name, obs, mod, outdir)

        # Mappa generale
        plot_tg_map(df_coo, outdir)

        # Mappe singole per slide
        for _, row in df_coo.iterrows():
          plot_tg_map_single(df_coo, row['name'], outdir)

        if df_plot is None: continue

        # -------------------------
        # DETIDING OBS/MODEL
        # -------------------------
        low_bound_d = 1.0 / diurnal_tides_band[1]
        high_bound_d = 1.0 / diurnal_tides_band[0]
        low_bound_sd = 1.0 / semid_tides_band[1]
        high_bound_sd = 1.0 / semid_tides_band[0]

        obs_detided, _ = fft2bands(
            df_plot['ssh_obs'].values,
            low_bound=high_bound_d, high_bound=low_bound_d,
            low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
            alpha=0.4, invert='False'
        )
        df_plot['ssh_obs_detided'] = obs_detided

        mod_detided, _ = fft2bands(
            df_plot['ssh_mod_offset'].values,
            low_bound=high_bound_d, high_bound=low_bound_d,
            low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
            alpha=0.4, invert='False'
        )
        df_plot['ssh_mod_offset_detided'] = mod_detided

        plot_detided_ts(name, df_plot, outdir)
        plot_spectra(name, df_plot, outdir, dt_hours=1.0)

        # -------------------------
        # FORECASTS
        # -------------------------
        event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
        event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
        xlim_start = pd.Timestamp('2026-01-14 00:00:00', tz='UTC')
        xlim_end   = pd.Timestamp('2026-01-22 23:59:59', tz='UTC')
        
        # --- PLOT 1: forecast ORIGINAL ---
        plt.figure(figsize=(12,6))
        plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
        
        # calcolo max su ANALYSIS e OBS
        df_event = df_plot.loc[event_start:event_end]
        idx_max_obs = df_event['ssh_obs'].idxmax()
        max_obs = df_event['ssh_obs'].max()
        idx_max_mod = df_event['ssh_mod_offset'].idxmax()
        max_mod = df_event['ssh_mod_offset'].max()
        
        plt.plot(
            df_plot.index,
            df_plot['ssh_obs'],
            color='tab:orange',
            lw=3,
            label=f'OBS (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})'
        )
        plt.plot(
            df_plot.index,
            df_plot['ssh_mod_offset'],
            color='blue',
            lw=3,
            label=f'ANALYSIS (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})'
        )
        
        # forecast
        bdates = pd.date_range('2026-01-11','2026-01-21', freq='D')
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=len(bdates)-1)
        
        for i, bdate in enumerate(bdates):
            fc_file = mod_dir / f'{name}_mod_MedFS_forecast_b{bdate:%Y%m%d}.nc'
            if not fc_file.exists():
                print(f"Missing forecast file: {fc_file}")
                continue
            df_fc = read_model_nc(fc_file)
            # offset rispetto alle OBS presenti nel periodo forecast
            mask_obs = (df_plot.index >= df_fc.index.min()) & (df_plot.index <= df_fc.index.max())
            offset_fc = df_plot['ssh_obs'].loc[mask_obs].mean() - df_fc['ssh_mod'].mean() if mask_obs.any() else 0.0
            df_fc['ssh_mod_offset'] = df_fc['ssh_mod'] + offset_fc
        
            # calcolo massimo per questa forecast
            df_event_fc = df_fc.loc[event_start:event_end]
            if not df_event_fc.empty:
                idx_max_fc = df_event_fc['ssh_mod_offset'].idxmax()
                max_fc = df_event_fc['ssh_mod_offset'].max()
            else:
                idx_max_fc = df_fc.index[0]
                max_fc = df_fc['ssh_mod_offset'].max()
        
            color = cmap(norm(i))
            plt.plot(
                df_fc.index,
                df_fc['ssh_mod_offset'],
                lw=2,
                color=color,
                label=f'FC {bdate:%d %b} (max {max_fc:.3f} m @ {idx_max_fc:%d/%m %H:%M})'
            )
        
        plt.title(f'{name}', fontsize=20)
        plt.ylabel('Sea level [m]', fontsize=20)
        plt.xlabel('Time', fontsize=20)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.7)
        plt.xticks(fontsize=16, rotation=30)
        plt.yticks(fontsize=16)
        plt.xlim(xlim_start, xlim_end)
        plt.grid()
        plt.tight_layout()
        outdir.mkdir(exist_ok=True, parents=True)
        plt.savefig(outdir / f'{name}_forecast_all_original.png', dpi=150)
        plt.close()
        
        # --- PLOT 2: forecast DETIDED ---
        plt.figure(figsize=(12,6))
        plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
        
        # calcolo max su ANALYSIS detided e OBS detided
        offset_detided = df_plot['ssh_obs_detided'].mean() - df_plot['ssh_mod_offset_detided'].mean()
        df_plot['ssh_mod_offset_detided_aligned'] = df_plot['ssh_mod_offset_detided'] + offset_detided
        
        df_event = df_plot.loc[event_start:event_end]
        idx_max_obs = df_event['ssh_obs_detided'].idxmax()
        max_obs = df_event['ssh_obs_detided'].max()
        idx_max_mod = df_event['ssh_mod_offset_detided_aligned'].idxmax()
        max_mod = df_event['ssh_mod_offset_detided_aligned'].max()
        
        plt.plot(
            df_plot.index,
            df_plot['ssh_obs_detided'],
            color='tab:orange',
            lw=3,
            label=f'OBS detided (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})'
        )
        plt.plot(
            df_plot.index,
            df_plot['ssh_mod_offset_detided_aligned'],
            color='blue',
            lw=3,
            label=f'ANALYSIS detided (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})'
        )
        
        # forecast detided
        for i, bdate in enumerate(bdates):
            fc_file = mod_dir / f'{name}_mod_MedFS_forecast_b{bdate:%Y%m%d}.nc'
            if not fc_file.exists():
                continue
            df_fc = read_model_nc(fc_file)
            mask_obs = (df_plot.index >= df_fc.index.min()) & (df_plot.index <= df_fc.index.max())
            offset_fc = df_plot['ssh_obs'].loc[mask_obs].mean() - df_fc['ssh_mod'].mean() if mask_obs.any() else 0.0
            df_fc['ssh_mod_offset'] = df_fc['ssh_mod'] + offset_fc
        
            df_fc_detided, _ = fft2bands(
                df_fc['ssh_mod_offset'].values,
                low_bound=high_bound_d, high_bound=low_bound_d,
                low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
                alpha=0.4, invert='False'
            )
            df_fc_detided_series = pd.Series(df_fc_detided, index=df_fc.index)
        
            df_event_fc = df_fc_detided_series.loc[event_start:event_end]
            if not df_event_fc.empty:
                idx_max_fc = df_event_fc.idxmax()
                max_fc = df_event_fc.max()
            else:
                idx_max_fc = df_fc.index[0]
                max_fc = df_fc_detided_series.max()
        
            color = cmap(norm(i))
            plt.plot(
                df_fc.index,
                df_fc_detided_series,
                lw=2,
                color=color,
                label=f'FC {bdate:%d %b} detided (max {max_fc:.3f} m @ {idx_max_fc:%d/%m %H:%M})'
            )
        
        plt.title(f'{name} – Detided sea level', fontsize=20)
        plt.ylabel('Sea level [m]', fontsize=20)
        plt.xlabel('Time', fontsize=20)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.7)
        plt.xticks(fontsize=16, rotation=30)
        plt.yticks(fontsize=16)
        plt.xlim(xlim_start, xlim_end)
        plt.grid()
        plt.tight_layout()
        outdir.mkdir(exist_ok=True, parents=True)
        plt.savefig(outdir / f'{name}_forecast_all_detided.png', dpi=150)
        plt.close()

        # -------------------------
        # STATISTICHE
        # -------------------------
        stats = compute_stats(df_plot, name)
        stats_list.append(stats)

    # -------------------------
    # PLOT COMBINATI
    # -------------------------
    combine_all_plots(df_coo, outdir)


    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats.to_csv(outdir / 'tg_obs_vs_mod_stats.csv', index=False)
        print("Statistiche salvate in tg_obs_vs_mod_stats.csv")
    else:
        print("Nessuna statistica calcolata")

if __name__ == '__main__':
    main()
