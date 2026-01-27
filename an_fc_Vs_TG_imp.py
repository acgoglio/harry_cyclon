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

def read_obs_csv(obs_folder, hourly_mean=True):
    obs_folder = Path(obs_folder)
    obs_files = [obs_folder] if obs_folder.is_file() else list(obs_folder.glob('*.csv'))
    obs_dict = {}
    for file_path in obs_files:
        try:
            times, values = [], []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(' ', 2)
                    if len(parts) < 2: continue
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
            if hourly_mean:
                df = df.resample('1H', label='right', closed='right').mean()
                df.index = df.index - pd.Timedelta(minutes=30)
            obs_dict[file_path.stem] = df
        except Exception as e:
            print(f"Obs error for {file_path.name}: {e}")
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
    df = obs.join(mod, how='inner')
    if df.empty:
        print(f'Nessuna sovrapposizione temporale per {name}')
        return None
    offset = df['ssh_obs'].mean() - df['ssh_mod'].mean()
    df['ssh_mod_offset'] = df['ssh_mod'] + offset
    event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
    event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
    df_event = df.loc[event_start:event_end]
    idx_max_obs = df_event['ssh_obs'].idxmax()
    max_obs = df_event['ssh_obs'].max()
    idx_max_mod = df_event['ssh_mod_offset'].idxmax()
    max_mod = df_event['ssh_mod_offset'].max()
    fontsize, linewidth, legend_fontsize = 20, 3, 18
    plt.figure(figsize=(12,6))
    plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
    plt.plot(df.index, df['ssh_obs'], color='tab:orange', label=f'OBS (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})', lw=linewidth)
    plt.plot(df.index, df['ssh_mod_offset'], color='blue', label=f'ANALYSIS (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M}, offset {offset:.3f} m)', lw=linewidth)
    plt.title(name, fontsize=fontsize)
    plt.ylabel('Sea level [m]', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.7)
    plt.xticks(fontsize=fontsize-2, rotation=30)
    plt.yticks(fontsize=fontsize-2)
    plt.grid()
    plt.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / f'{name}_obs_vs_mod.png', dpi=150)
    plt.close()
    return df

def plot_tg_map(df_coo, outdir, figsize=(14,8)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    min_lon = df_coo['lon'].min() - 1
    max_lon = df_coo['lon'].max() + 1
    min_lat = df_coo['lat'].min() - 1
    max_lat = df_coo['lat'].max() + 1
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    for _, row in df_coo.iterrows():
        lat, lon, name = row['lat'], row['lon'], row['name']
        ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
        ax.text(lon+0.2, lat, name, fontsize=16, transform=ccrs.PlateCarree())
    plt.title('Tide-Gauges', fontsize=20)
    outdir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outdir / 'tg_map_mediterranean.png', dpi=150)
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
    offset_detided = df['ssh_obs_detided'].mean() - df['ssh_mod_offset_detided'].mean()
    df['ssh_mod_offset_detided_aligned'] = df['ssh_mod_offset_detided'] + offset_detided
    event_start = pd.Timestamp('2026-01-19 00:00:00', tz='UTC')
    event_end   = pd.Timestamp('2026-01-21 23:59:59', tz='UTC')
    df_event = df.loc[event_start:event_end]
    idx_max_obs = df_event['ssh_obs_detided'].idxmax()
    max_obs = df_event['ssh_obs_detided'].max()
    idx_max_mod = df_event['ssh_mod_offset_detided_aligned'].idxmax()
    max_mod = df_event['ssh_mod_offset_detided_aligned'].max()
    plt.figure(figsize=(12,6))
    plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
    plt.plot(df.index, df['ssh_obs_detided'], color='tab:orange', label=f'OBS detided (max {max_obs:.3f} m @ {idx_max_obs:%d/%m %H:%M})', lw=linewidth)
    plt.plot(df.index, df['ssh_mod_offset_detided_aligned'], color='blue', label=f'ANALYSIS detided (max {max_mod:.3f} m @ {idx_max_mod:%d/%m %H:%M})', lw=linewidth)
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
        except Exception as e:
            print(f"Obs error: {e}")
            continue

        nc_file = mod_dir / f'{name}_mod_MedFS_analysis_26012026.nc'
        if not nc_file.exists(): 
            print(f"Missing model file for {name}")
            continue
        mod = read_model_nc(nc_file, obs.index)

        df_plot = plot_tg(name, obs, mod, outdir)
        plot_tg_map(df_coo, outdir)
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
        # Limiti plot
        xlim_start = pd.Timestamp('2026-01-18 00:00:00', tz='UTC')
        xlim_end   = pd.Timestamp('2026-01-22 23:59:59', tz='UTC')
        
        # --- PLOT 1: forecast ORIGINAL ---
        plt.figure(figsize=(12,6))
        plt.axvspan(event_start, event_end, color='lightgrey', alpha=0.3)
        plt.plot(df_plot.index, df_plot['ssh_obs'], color='tab:orange', label='OBS', lw=3)
        plt.plot(df_plot.index, df_plot['ssh_mod_offset'], color='blue', label='ANALYSIS', lw=3)
        
        bdates = pd.date_range('2026-01-11', '2026-01-21', freq='D')
        cmap = plt.cm.viridis   # puoi cambiare: Reds, Purples, viridis, ecc.
        norm = plt.Normalize(vmin=0, vmax=len(bdates)-1)

        for bdate in pd.date_range('2026-01-11','2026-01-21', freq='D'):
            fc_file = mod_dir / f'{name}_mod_MedFS_forecast_b{bdate:%Y%m%d}.nc'
            if not fc_file.exists(): 
                print(f"Missing forecast file: {fc_file}")
                continue
            df_fc = read_model_nc(fc_file)
            start_fc, end_fc = df_fc.index.min(), df_fc.index.max()
            mask_obs = (df_plot.index >= start_fc) & (df_plot.index <= end_fc)
            offset_fc = df_plot['ssh_obs'].loc[mask_obs].mean() - df_fc['ssh_mod'].mean() if mask_obs.any() else 0.0
            df_fc['ssh_mod_offset'] = df_fc['ssh_mod'] + offset_fc
        
            i = bdates.get_loc(bdate)
            color = cmap(norm(i))

            # plot originale
            plt.plot(df_fc.index, df_fc['ssh_mod_offset'], color=color, label=f'FC bulletin {bdate:%d %b %Y}', lw=2)
        
        plt.title(f'{name} – Forecast ORIGINAL', fontsize=20)
        plt.ylabel('Sea level [m]', fontsize=20)
        plt.xlabel('Time', fontsize=20)
        plt.legend(fontsize=12, loc='upper left', framealpha=0.7)
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
        plt.plot(df_plot.index, df_plot['ssh_obs_detided'], color='tab:orange', label='OBS detided', lw=3)
        plt.plot(df_plot.index, df_plot['ssh_mod_offset_detided'], color='blue', label='ANALYSIS detided', lw=3)
        
        bdates = pd.date_range('2026-01-11', '2026-01-21', freq='D')
        cmap = plt.cm.viridis   # puoi cambiare: Reds, Purples, viridis, ecc.
        norm = plt.Normalize(vmin=0, vmax=len(bdates)-1)

        for bdate in pd.date_range('2026-01-11','2026-01-21', freq='D'):
            fc_file = mod_dir / f'{name}_mod_MedFS_forecast_b{bdate:%Y%m%d}.nc'
            if not fc_file.exists(): 
                continue
            df_fc = read_model_nc(fc_file)
            start_fc, end_fc = df_fc.index.min(), df_fc.index.max()
            mask_obs = (df_plot.index >= start_fc) & (df_plot.index <= end_fc)
            offset_fc = df_plot['ssh_obs'].loc[mask_obs].mean() - df_fc['ssh_mod'].mean() if mask_obs.any() else 0.0
            df_fc['ssh_mod_offset'] = df_fc['ssh_mod'] + offset_fc

            i = bdates.get_loc(bdate)
            color = cmap(norm(i))
        
            # detiding forecast
            df_fc_detided, _ = fft2bands(
                df_fc['ssh_mod_offset'].values,
                low_bound=high_bound_d, high_bound=low_bound_d,
                low_bound_1=high_bound_sd, high_bound_1=low_bound_sd,
                alpha=0.4, invert='False'
            )
            plt.plot(df_fc.index, df_fc_detided, '-', color=color, lw=2, label=f'FC detided {bdate:%d %b %Y}')
        
        plt.title(f'{name} – Forecast DETIDED', fontsize=20)
        plt.ylabel('Sea level [m]', fontsize=20)
        plt.xlabel('Time', fontsize=20)
        plt.legend(fontsize=12, loc='upper left', framealpha=0.7)
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

    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats.to_csv(outdir / 'tg_obs_vs_mod_stats.csv', index=False)
        print("Statistiche salvate in tg_obs_vs_mod_stats.csv")
    else:
        print("Nessuna statistica calcolata")

if __name__ == '__main__':
    main()
