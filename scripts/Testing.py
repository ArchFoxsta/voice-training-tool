import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import parselmouth
import time
import sys

# === SETTINGS ===
samplerate = 44100
blocksize = 1024
duration = 5  # seconds of scrollback
fft_size = 1024
gain_default = 10
pitch_floor = 75
pitch_interval_frames = 5  # pitch calculation every N frames
pitch_buffer_duration = 0.3  # seconds

# === PITCH BUFFER ===
pitch_buffer = np.zeros(0, dtype=np.float32)
pitch_buffer_target_size = int(samplerate * pitch_buffer_duration)
pitch_values = []

# === NOTE CONVERSION ===
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def hz_to_note(hz):
    if hz <= 0 or np.isnan(hz):
        return "—"
    note_num = 12 * np.log2(hz / 440.0) + 69
    note_index = int(round(note_num)) % 12
    octave = int(round(note_num)) // 12 - 1
    return f"{NOTE_NAMES[note_index]}{octave}"

# === GUI SETUP ===
n_blocks = int(duration * samplerate / blocksize)
freqs = np.fft.rfftfreq(fft_size, d=1 / samplerate)
n_freq_bins = len(freqs)
spec_data = np.zeros((n_freq_bins, n_blocks))

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)
img = ax.imshow(spec_data, origin='lower', aspect='auto',
                extent=(0, duration, freqs[0], freqs[-1]),
                cmap='magma', vmin=0, vmax=1)
pitch_line, = ax.plot([], [], 'c-', linewidth=2, label='Pitch line', animated=True) # solid cyan line
pitch_dots, = ax.plot([], [], 'o', color='cyan',
                      markersize=10, markeredgecolor='black', markeredgewidth=2,
                      zorder=100, label='Pitch Dots',
                      animated=True)
cb = plt.colorbar(img, ax=ax, label='Intensity (normalized)')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")

# === Pitch label text ===
pitch_label = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                      fontsize=14, color='cyan', verticalalignment='top')

# === Gain slider ===
ax_gain = plt.axes([0.2, 0.1, 0.65, 0.03])
gain_slider = Slider(ax_gain, 'Gain', 1, 100, valinit=gain_default, valstep=1)

# === WINDOW MAXIMIZATION ===
try:
    figManager = plt.get_current_fig_manager()
    if sys.platform.startswith('win'):
        figManager.window.showMaximized()
except Exception:
    pass

# === Pitch detection ===
def parselmouth_pitch(samples, sr):
    snd = parselmouth.Sound(samples, sampling_frequency=sr)
    duration = snd.get_total_duration()
    time_step = duration / 4
    try:
        pitch_obj = snd.to_pitch(time_step=time_step, pitch_floor=pitch_floor)
        freqs = pitch_obj.selected_array['frequency']
        voiced = freqs[freqs > 0]
        return np.median(voiced) if len(voiced) else np.nan
    except Exception as e:
        print(f"[Pitch Error] {e}")
        return np.nan

# === Audio callback ===
frame_counter = 0
def audio_callback(indata, outdata, frames, time_info, status):
    global spec_data, pitch_buffer, pitch_values, frame_counter
    outdata[:] = indata  # feedback

    # Spectrogram
    windowed = indata[:, 0] * np.hanning(len(indata))
    spectrum = np.abs(np.fft.rfft(windowed, n=fft_size))
    spectrum = np.clip(spectrum * gain_slider.val / fft_size, 0, 1)
    spec_data = np.roll(spec_data, -1, axis=1)
    spec_data[:, -1] = spectrum

    # Pitch buffer update
    pitch_buffer = np.concatenate((pitch_buffer, indata[:, 0]))
    if len(pitch_buffer) > pitch_buffer_target_size:
        pitch_buffer = pitch_buffer[-pitch_buffer_target_size:]

    frame_counter += 1
    if frame_counter % pitch_interval_frames == 0:
        pitch = parselmouth_pitch(pitch_buffer, samplerate)
        now = time.time()
        pitch_values.append((now, pitch))
        if len(pitch_values) > n_blocks:
            pitch_values.pop(0)

        # === Debugging ===
        print(f"[DEBUG] buffer_len: {len(pitch_buffer)}, pitch: {pitch:.1f} Hz")
        if not np.isnan(pitch):
            print(f"[SANITY] Valid pitch detected: {pitch:.1f} Hz")
        else:
            print("[SANITY] Invalid pitch (NaN) — skipping")
        # === ... ===

# === Plot update ===
def update_plot(frame):
    # === Manual sanity injection ===
    if len(pitch_values) < 5:  # Only do this once
        now = time.time()
        test_times = np.linspace(now - duration + 1, now - 1, 5)
        test_pitches = [100, 200, 300, 250, 180]
        pitch_values.clear()
        pitch_values.extend(zip(test_times, test_pitches))
        print("[SANITY] Injected test pitch points (centered on visible region).")

    # === End of debugging pitch plot points ===
    
    now = time.time()
    img.set_data(spec_data)
    img.set_extent((now - duration, now, freqs[0], freqs[-1]))
    ax.set_xlim(now - duration, now)
#    ax.set_ylim(0, freqs[-1])
    ax.set_ylim(0, 15000)  # Only show 0–15 kHz

    if pitch_values:
        times, pitches = zip(*pitch_values)
        rel_times = np.array(times) - now + duration
        pitches = np.array(pitches)

        # Filter to keep only recent data that's visible on screen
        visible = (rel_times > 0) & (rel_times <= duration) & (pitches > 0)
        if np.any(visible):
            pitch_line.set_data(rel_times[visible], pitches[visible])
            pitch_dots.set_data(rel_times[visible], pitches[visible])
            pitch_dots.set_markersize(30)  # Forcefully large for debug visibility
            
            latest_pitch = pitches[visible][-1]
            pitch_label.set_text(f"Pitch: {latest_pitch:.1f} Hz ({hz_to_note(latest_pitch)})")
        else:
            pitch_line.set_data([], [])
            pitch_dots.set_data([], [])
            pitch_label.set_text("Pitch: — Hz (—)")


        print(f"[DEBUG] showing {np.sum(visible)} pitch points") # Debugging Pitch Plotting
        print(f"[SANITY] Pitch Dots: {list(zip(rel_times[visible], pitches[visible]))}")

    # Manual test if pitch_dots aren't showing
#    pitch_dots.set_data([1, 2, 3], [100, 200, 300])


    ax.set_title(f"Spectrogram | Gain: {gain_slider.val:.0f} | Interval: 50 ms")
    return img, pitch_line, pitch_dots, pitch_label

# === Stream & Animation Start ===
stream = sd.Stream(samplerate=samplerate, blocksize=blocksize,
                   channels=1, callback=audio_callback)

ani = animation.FuncAnimation(fig, update_plot, interval=50,
                              blit=True, cache_frame_data=False)

with stream:
    plt.show()
