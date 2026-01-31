import flask
from flask import Flask, render_template, request, flash, session, url_for, redirect, make_response
import json
import random
import string
import wave
import math
import array
import io
import struct
from thefuzz import fuzz  # Corrected import

# Initialize the Flask application
app = Flask(__name__)
# A strong secret key is crucial for session security
app.config['SECRET_KEY'] = 'a-very-secret-and-secure-key-for-captcha'

# --- Audio Generation Constants ---
SAMPLERATE = 22050  # Hz (CD quality is 44100, this is good enough)
REFLEX_BUFFER = 0.150  # 150ms buffer for human reflexes (default)
BOT_THRESHOLD_MS = 150  # Min human reaction time

# --- Statistical Bot Detection Constants ---
# Jitter check for 'normal' rule (log-space)
MIN_REACTION_JITTER_LOG = 0.1
# Jitter check for 'normal' rule (ms-space)
MIN_HOLD_JITTER_MS = 2
# NEW: Jitter checks for 'inverted' rule (ms-space)
MIN_INVERTED_HOLD_JITTER_MS = 10  # Expect *some* variation in hold times
MIN_INVERTED_GAP_JITTER_MS = 10  # Expect *some* variation in time between holds

# --- Decoupled Gaussian/Log-Normal Scoring (DGS) Constants ---
# These act as "strictness" penalties in the scoring function exp(-lambda * (delta^2))
LAMBDA_REACTION = 2.0  # Lower penalty for reaction time inconsistency (now in log-space)
LAMBDA_DURATION = 5.0  # Higher penalty for incorrect hold duration (remains in linear-space)


# --- Riddle Loading ---
def load_riddles():
    """Loads the riddle JSON file."""
    try:
        with open('riddles.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: riddles.json not found.")
        return []
    except json.JSONDecodeError:
        print("ERROR: riddles.json is not valid JSON.")
        return []


# --- Helper Functions ---
def generate_captcha_text(length=6):
    """Generates a random alphanumeric string for the audio captcha."""
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))


def generate_timing_audio(beep_data, total_duration):
    """
    Generates an in-memory STEREO WAV file with an advanced "hostile audio environment".
    """
    total_samples = total_duration * SAMPLERATE

    # --- 0. Setup Stereo Audio Array ---
    audio_data = array.array('h', [0] * total_samples * 2)

    # --- 1. Generate Background "Rumble" (Stereo Brown Noise) ---
    last_val_L = 0
    last_val_R = 0
    for i in range(total_samples):
        last_val_L += random.randint(-100, 100)
        last_val_R += random.randint(-100, 100)
        last_val_L = max(-16000, min(16000, last_val_L))
        last_val_R = max(-16000, min(16000, last_val_R))
        audio_data[i * 2] = int(last_val_L * 0.6)  # L Channel
        audio_data[i * 2 + 1] = int(last_val_R * 0.6)  # R Channel

    # --- 2. High-Frequency "Hiss" (Stereo White Noise) ---
    hiss_amp_L = 1.0
    hiss_amp_R = 1.0
    hiss_mod_freq_L = random.uniform(0.1, 0.3)
    hiss_mod_freq_R = random.uniform(0.1, 0.3)

    for i in range(total_samples):
        t = float(i) / SAMPLERATE
        mod_L = 1.0 - (0.2 * ((math.sin(2 * math.pi * hiss_mod_freq_L * t) + 1) / 2))
        mod_R = 1.0 - (0.2 * ((math.sin(2 * math.pi * hiss_mod_freq_R * t) + 1) / 2))
        hiss_L = random.randint(-4000, 4000) * mod_L
        hiss_R = random.randint(-4000, 4000) * mod_R
        audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + int(hiss_L)))
        audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + int(hiss_R)))

    # --- 3. Generate "Babble" (Advanced 5-Oscillator Stereo Simulation) ---
    oscillators = []
    for _ in range(5):
        oscillators.append({
            'freq': random.uniform(200, 800), 'amp': random.uniform(600, 1200),
            'mod_freq': random.uniform(0.1, 0.5), 'mod_phase_L': random.uniform(0, math.pi),
            'mod_phase_R': random.uniform(0, math.pi), 'warp_freq': random.uniform(0.05, 0.2),
            'warp_depth': random.uniform(0.01, 0.05)
        })

    for i in range(total_samples):
        t = float(i) / SAMPLERATE
        babble_signal_L = 0
        babble_signal_R = 0
        for osc in oscillators:
            warp_mod = osc['warp_depth'] * math.sin(2 * math.pi * osc['warp_freq'] * t)
            current_freq = osc['freq'] * (1 + warp_mod)
            mod_L = (math.sin(2 * math.pi * osc['mod_freq'] * t + osc['mod_phase_L']) + 1) / 2
            mod_R = (math.sin(2 * math.pi * osc['mod_freq'] * t + osc['mod_phase_R']) + 1) / 2
            wave_val = math.sin(2 * math.pi * current_freq * t)
            babble_signal_L += int(osc['amp'] * mod_L * wave_val)
            babble_signal_R += int(osc['amp'] * mod_R * wave_val)
        babble_signal_L //= len(oscillators)
        babble_signal_R //= len(oscillators)
        audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + int(babble_signal_L * 0.7)))
        audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + int(babble_signal_R * 0.7)))

    # --- 4. Generate "Feigning Beeps" (Rogue Tones) ---
    num_feigning_beeps = random.randint(5, 10)
    for _ in range(num_feigning_beeps):
        feign_duration_samples = int(random.uniform(0.1, 0.5) * SAMPLERATE)
        start_sample = random.randint(0, total_samples - feign_duration_samples)
        feign_freq = random.uniform(500, 1200)
        feign_pan = random.uniform(0.0, 1.0)
        feign_pan_L = 1.0 - feign_pan
        feign_pan_R = feign_pan

        for i in range(start_sample, start_sample + feign_duration_samples):
            if i >= total_samples: break
            t = float(i) / SAMPLERATE
            feign_signal = 8000 * math.sin(2 * math.pi * feign_freq * t)
            fade_len = SAMPLERATE // 100  # 10ms
            if (i - start_sample) < fade_len:
                feign_signal *= ((i - start_sample) / fade_len)
            elif (start_sample + feign_duration_samples - i) < fade_len:
                feign_signal *= ((start_sample + feign_duration_samples - i) / fade_len)
            signal_L = int(feign_signal * feign_pan_L)
            signal_R = int(feign_signal * feign_pan_R)
            audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + signal_L))
            audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + signal_R))

    # --- 5. Generate Digital Glitches (Chaos) ---
    num_glitches = random.randint(10, 15)
    for _ in range(num_glitches):
        start_sample = random.randint(0, total_samples - (SAMPLERATE // 20))
        glitch_duration_samples = int(random.uniform(0.01, 0.05) * SAMPLERATE)
        for i in range(start_sample, start_sample + glitch_duration_samples):
            if i >= total_samples: break
            glitch_signal = random.randint(-15000, 15000)
            audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + glitch_signal))
            audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + glitch_signal))

    # --- 6. Psychoacoustic Masking Glitches ---
    for beep in beep_data:
        if beep['is_masked']:
            start_sample = int((beep['start'] - 0.05) * SAMPLERATE)
            mask_duration_samples = int(0.1 * SAMPLERATE)
            for i in range(start_sample, start_sample + mask_duration_samples):
                if i < 0 or i >= total_samples: continue
                glitch_signal = random.randint(-12000, 12000)
                audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + glitch_signal))
                audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + glitch_signal))

    # --- 7. Generate REAL Beeps (with ADSR, Timbre, Panning, Vibrato, Distortion, Reverb) ---
    attack_samples = int(SAMPLERATE * 0.01)  # 10ms
    decay_samples = int(SAMPLERATE * 0.05)  # 50ms
    sustain_level = 0.7
    release_samples = int(SAMPLERATE * 0.02)  # 20ms
    reverb_delay_samples = int(SAMPLERATE * 0.1)  # 100ms

    for beep in beep_data:
        start_sample = int(beep['start'] * SAMPLERATE)
        end_sample = int(beep['end'] * SAMPLERATE)
        beep_duration_samples = end_sample - start_sample
        freq1 = beep['freq']
        freq2 = freq1 * 2
        pan = beep['pan']
        pan_L = 1.0 - pan
        pan_R = pan
        amp_wave1 = beep['harmonic_mix']
        amp_wave2 = 1.0 - beep['harmonic_mix']
        vibrato_freq = random.uniform(5, 8)
        vibrato_depth = random.uniform(0.01, 0.02)

        for i in range(start_sample, end_sample + release_samples):
            if i >= total_samples: break
            t = float(i) / SAMPLERATE
            sample_in_beep = i - start_sample
            envelope_gain = 0.0
            if sample_in_beep < 0: continue
            if sample_in_beep < attack_samples:
                envelope_gain = sample_in_beep / attack_samples
            elif sample_in_beep < (attack_samples + decay_samples):
                envelope_gain = 1.0 - (1.0 - sustain_level) * (sample_in_beep - attack_samples) / decay_samples
            elif sample_in_beep < beep_duration_samples:
                envelope_gain = sustain_level
            elif sample_in_beep < (beep_duration_samples + release_samples):
                envelope_gain = sustain_level * (1.0 - (sample_in_beep - beep_duration_samples) / release_samples)
            else:
                envelope_gain = 0.0
            if envelope_gain <= 0.0: continue
            current_freq1 = freq1 * (1 + vibrato_depth * math.sin(2 * math.pi * vibrato_freq * t))
            current_freq2 = freq2 * (1 + vibrato_depth * math.sin(2 * math.pi * vibrato_freq * t))
            wave1 = 16383 * math.sin(2 * math.pi * current_freq1 * t) * amp_wave1
            wave2 = 16383 * math.sin(2 * math.pi * current_freq2 * t) * amp_wave2
            beep_signal = int(wave1 + wave2)
            beep_signal = 32767 * (2 / math.pi) * math.atan(beep_signal / 8192.0)
            beep_signal = int(beep_signal * envelope_gain)
            signal_L = int(beep_signal * pan_L)
            signal_R = int(beep_signal * pan_R)
            audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + signal_L))
            audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + signal_R))
            reverb_index = i - reverb_delay_samples
            if reverb_index >= 0:
                reverb_L_in = audio_data[reverb_index * 2]
                reverb_R_in = audio_data[reverb_index * 2 + 1]
                reverb_out_L = int((reverb_L_in * 0.2) + (reverb_R_in * 0.4))
                reverb_out_R = int((reverb_L_in * 0.4) + (reverb_R_in * 0.2))
                audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2] + reverb_out_L))
                audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1] + reverb_out_R))

    # --- 8. Apply Tremolo (Nonlinear Volume Variation) ---
    tremolo_freq_L = random.uniform(0.3, 0.8)
    tremolo_depth_L = random.uniform(0.2, 0.4)
    tremolo_freq_R = random.uniform(0.3, 0.8)
    tremolo_depth_R = random.uniform(0.2, 0.4)
    for i in range(total_samples):
        t = float(i) / SAMPLERATE
        volume_mod_L = 1 - (tremolo_depth_L * ((math.sin(2 * math.pi * tremolo_freq_L * t) + 1) / 2))
        volume_mod_R = 1 - (tremolo_depth_R * ((math.sin(2 * math.pi * tremolo_freq_R * t) + 1) / 2))
        audio_data[i * 2] = int(audio_data[i * 2] * volume_mod_L)
        audio_data[i * 2 + 1] = int(audio_data[i * 2 + 1] * volume_mod_R)
        audio_data[i * 2] = max(-32768, min(32767, audio_data[i * 2]))
        audio_data[i * 2 + 1] = max(-32768, min(32767, audio_data[i * 2 + 1]))

    # --- 9. Create in-memory WAV file ---
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(2)  # STEREO
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLERATE)
        wav_file.writeframes(audio_data.tobytes())
    audio_buffer.seek(0)
    return audio_buffer


def calculate_std_dev(data):
    """
    Calculates the standard deviation from a list of numbers.
    """
    n = len(data)
    if n < 2:
        return 0  # Cannot calculate variance with < 2 data points
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)  # Sample variance
    std_dev = math.sqrt(variance)
    return std_dev


def calculate_timing_confidence_IoU(user_intervals, correct_intervals, total_duration, rule):
    """
    This function calculates a score based on simple IoU.
    It is ONLY used for the 'inverted' rule, which is simpler.
    """
    if not user_intervals and not correct_intervals:
        return 100  # Correctly did nothing

    target_intervals = []

    # This function is now ONLY used for the 'inverted' rule
    if rule == 'inverted':
        # Rule is 'inverted'. Target intervals ARE THE SILENCE.

        # 1. Generate the "silence" intervals (using unbuffered beep times)
        non_beep_intervals = []
        current_time = 0.0
        sorted_correct_intervals = sorted(correct_intervals, key=lambda x: x[0])
        for start, end in sorted_correct_intervals:
            if start > current_time:
                non_beep_intervals.append([current_time, start])
            current_time = end

        if current_time < total_duration:
            non_beep_intervals.append([current_time, total_duration])

        # 2. Apply reflex buffer by *shrinking* silence intervals
        buffered_silence_intervals = []
        for i, (start, end) in enumerate(non_beep_intervals):
            buffered_start = start
            buffered_end = end

            if i > 0:  # If not the very beginning, add buffer to the start
                buffered_start = start + REFLEX_BUFFER
            if i < len(non_beep_intervals) - 1:  # If not the very end, subtract buffer from the end
                buffered_end = end - REFLEX_BUFFER

            if buffered_end > buffered_start:  # Ensure interval is valid
                buffered_silence_intervals.append([buffered_start, buffered_end])
            elif i == 0 and i == len(non_beep_intervals) - 1 and buffered_end > buffered_start:
                buffered_silence_intervals.append([buffered_start, buffered_end])

        target_intervals = buffered_silence_intervals
    else:
        # This function should not be called for the 'normal' rule,
        # but we leave a fallback just in case.
        target_intervals = [
            [max(0, start - REFLEX_BUFFER), end + REFLEX_BUFFER]
            for start, end in correct_intervals
        ]

    # --- Standard IoU Calculation using target_intervals ---
    if not user_intervals and not target_intervals: return 100
    if not user_intervals and target_intervals: return 0
    if not target_intervals and user_intervals: return 0

    max_user_time = max(ivl[1] for ivl in user_intervals) if user_intervals else 0
    max_time = max(max_user_time, total_duration)

    resolution = 100  # 10ms increments
    timeline_len = int(max_time * resolution) + 1

    user_timeline = [0] * timeline_len
    for start, end in user_intervals:
        start_idx = int(start * resolution)
        end_idx = int(end * resolution)
        for i in range(start_idx, end_idx):
            if i < timeline_len: user_timeline[i] = 1

    correct_timeline = [0] * timeline_len
    for start, end in target_intervals:
        start_idx = int(start * resolution)
        end_idx = int(end * resolution)
        for i in range(start_idx, end_idx):
            if i < timeline_len: correct_timeline[i] = 1

    intersection = 0
    union = 0
    for i in range(timeline_len):
        if user_timeline[i] == 1 and correct_timeline[i] == 1:
            intersection += 1
            union += 1
        elif user_timeline[i] == 1 or correct_timeline[i] == 1:
            union += 1

    if union == 0: return 100 if intersection == 0 else 0
    return (intersection / union) * 100


def get_reaction_data(user_intervals, correct_intervals):
    """
    Calculates all reaction times and hold durations for a 'normal' rule attempt.
    Returns:
    - reaction_times_sec: List of individual reaction times (float, seconds)
    - hold_durations_sec: List of individual hold durations (float, seconds)
    - matched_hit_data: List of dicts for DGS scoring
    - avg_reaction_time_ms: Average reaction time (int, milliseconds)
    """
    reaction_times_sec = []
    hold_durations_sec = []
    matched_hit_data = []

    if not user_intervals or not correct_intervals:
        return [], [], [], 0  # Return four items

    for c_start, c_end in correct_intervals:
        # Find a corresponding user interval (one that overlaps)
        for u_start, u_end in user_intervals:
            # Check for overlap:
            if max(c_start, u_start) < min(c_end, u_end):
                reaction = u_start - c_start
                if reaction > 0:  # Only count positive reaction times
                    reaction_times_sec.append(reaction)

                hold_duration = u_end - u_start
                if hold_duration > 0:
                    hold_durations_sec.append(hold_duration)

                # Store the raw data for DGS
                matched_hit_data.append({
                    "u_start": u_start,
                    "s_start": c_start,
                    "u_hold": hold_duration,
                    "s_hold": c_end - c_start
                })

                # Assume one-to-one mapping, break after finding
                break

    if not reaction_times_sec:
        return [], [], [], 0

    avg_reaction_time_sec = sum(reaction_times_sec) / len(reaction_times_sec)
    avg_reaction_time_ms = int(avg_reaction_time_sec * 1000)

    return reaction_times_sec, hold_durations_sec, matched_hit_data, avg_reaction_time_ms


# --- Flask Routes ---
@app.route('/')
def index():
    """Displays the captcha page with the chosen challenge."""
    captcha_type = request.args.get('type', 'audio')  # Default to 'audio'

    spoken_text = ""
    prompt = ""
    answer = ""

    if captcha_type == 'audio':
        answer = generate_captcha_text()
        spoken_text = f"The code is: {', '.join(answer)}. I repeat, {', '.join(answer)}."
        prompt = "Press the play button to hear the code, then type it in the box below."

    elif captcha_type == 'riddle':
        riddles = load_riddles()
        if not riddles:
            flash('Error: Could not load riddles. Please contact support.', 'error')
            return render_template('index.html', captcha_type='audio', prompt='Error loading riddles.')

        challenge = random.choice(riddles)
        answer = challenge['answer']
        spoken_text = challenge['riddle']
        prompt = "Press the play button to hear the riddle, then type your answer in the box below."

    elif captcha_type == 'timing':
        # Setup for the new timing captcha
        total_duration = random.randint(7, 15)
        num_beeps = random.randint(3, 7)
        beeps_data = []
        current_time = 1.0  # Start 1 sec in

        rule = random.choice(['normal', 'inverted'])

        for _ in range(num_beeps):
            current_time += random.uniform(0.5, 2.0)
            beep_duration = random.uniform(1.0, 2.5)
            jitter = random.uniform(-0.05, 0.05)
            start = round(current_time + jitter, 3)
            end = round(start + beep_duration, 3)
            if end > total_duration - 0.5: break
            beep_freq = random.randint(500, 1200)
            pan = random.uniform(0.1, 0.9)
            harmonic_mix = random.uniform(0.2, 0.8)
            is_masked = random.choice([True, False])

            beeps_data.append({
                "start": start, "end": end, "freq": beep_freq,
                "pan": pan, "harmonic_mix": harmonic_mix, "is_masked": is_masked
            })
            current_time = end

        answer_data = {
            "duration": total_duration,
            "beeps": beeps_data,
            "rule": rule
        }
        answer = json.dumps(answer_data)

        if rule == 'normal':
            prompt = "Press play, then hold SPACEBAR *only* when you hear the sounds. Your response will be submitted automatically when the audio finishes."
        else:  # 'inverted'
            prompt = "Press play, then hold SPACEBAR *only* during the SILENCE. *Do not* hold it on the sounds. Your response will be submitted automatically."

    session['correct_answer'] = answer
    session['captcha_type'] = captcha_type

    return render_template('index.html', captcha_type=captcha_type, prompt=prompt, spoken_text=spoken_text)


@app.route('/timing_audio')
def timing_audio():
    """Generates and streams the unique timing challenge audio."""
    answer_json = session.get('correct_answer')

    if not answer_json:
        return "No captcha answer found in session. Please go back and select the challenge first.", 400

    try:
        answer_data = json.loads(answer_json)
        beep_data = answer_data['beeps']
        total_duration = answer_data['duration']
    except (json.JSONDecodeError, KeyError):
        return "Invalid captcha data in session.", 400

    audio_buffer = generate_timing_audio(beep_data, total_duration)

    response = make_response(audio_buffer.read())
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'inline; filename=challenge.wav'
    return response


@app.route('/validate', methods=['POST'])
def validate():
    """Validates the user's captcha submission."""
    user_input = request.form.get('captcha_input', '').strip()
    correct_answer = session.pop('correct_answer', None)
    captcha_type = session.pop('captcha_type', None)

    if not correct_answer or not captcha_type:
        flash('Session expired or tampered. Please try again.', 'error')
        return redirect(url_for('index'))

    # --- Validation Logic ---
    if captcha_type == 'audio':
        if user_input.upper() == correct_answer:
            flash(f'Success! You are verified.', 'success')
        else:
            flash(f'Verification failed. Please try again.', 'error')

    elif captcha_type == 'riddle':
        similarity = fuzz.token_set_ratio(user_input.lower(), correct_answer.lower())
        if similarity > 85:  # Over 85% similar
            flash(f'Success! You are verified.', 'success')
        else:
            flash(f'Verification failed. Please try again.', 'error')

    elif captcha_type == 'timing':
        try:
            user_intervals = json.loads(user_input)
            correct_answer_data = json.loads(correct_answer)
            correct_intervals = [[beep['start'], beep['end']] for beep in correct_answer_data['beeps']]
            total_duration = correct_answer_data['duration']
            rule = correct_answer_data['rule']

        except (json.JSONDecodeError, KeyError):
            flash('Error! Invalid response data. Please try again.', 'error')
            return redirect(url_for('index', type='timing'))

        avg_reaction_time_ms = 0
        final_confidence = 0

        if rule == 'normal':
            reaction_times_sec, hold_durations_sec, matched_hit_data, avg_reaction_time_ms = get_reaction_data(
                user_intervals, correct_intervals)

            if not matched_hit_data:  # User hit 0 beeps
                flash('Verification failed. Your timing was 0% accurate.', 'error')
                return redirect(url_for('index', type='timing'))

            # --- Bot Check 1: Superhuman reaction (too fast) ---
            for rt in reaction_times_sec:
                if (rt * 1000) < BOT_THRESHOLD_MS:
                    flash(
                        f'Verification failed. Superhuman reaction time of {rt * 1000:.0f}ms detected. Please try again.',
                        'error')
                    return redirect(url_for('index', type='timing'))

            # --- Bot Check 2: Lack of human "jitter" (too consistent) ---
            if len(reaction_times_sec) > 1:  # Need at least 2 data points

                # Calculate jitter in log-space for reactions
                log_reaction_times = [math.log(rt) for rt in reaction_times_sec if rt > 0]
                reaction_jitter_log = calculate_std_dev(log_reaction_times)

                # Calculate jitter in ms-space for hold durations
                hold_durations_ms = [hd * 1000 for hd in hold_durations_sec]
                hold_jitter_ms = calculate_std_dev(hold_durations_ms)

                if reaction_jitter_log < MIN_REACTION_JITTER_LOG:
                    flash(
                        f'Verification failed. Unhumanly consistent reaction time (log-jitter: {reaction_jitter_log:.3f}). Please try again.',
                        'error')
                    return redirect(url_for('index', type='timing'))

                if hold_jitter_ms < MIN_HOLD_JITTER_MS:
                    flash(
                        f'Verification failed. Unhumanly consistent hold duration ({hold_jitter_ms:.1f}ms jitter). Please try again.',
                        'error')
                    return redirect(url_for('index', type='timing'))

            # --- DGS (Decoupled Gaussian/Log-Normal Scoring) ---

            # 1. Get mean of log-reaction times
            log_reaction_times = [math.log(rt) for rt in reaction_times_sec if rt > 0]
            mu_log_react = 0
            if log_reaction_times:
                mu_log_react = sum(log_reaction_times) / len(log_reaction_times)

            total_score = 0
            valid_hits = 0

            for hit in matched_hit_data:
                press_offset = hit['u_start'] - hit['s_start']

                # Can't score reaction times that are <= 0
                if press_offset <= 0:
                    continue

                valid_hits += 1
                log_press_offset = math.log(press_offset)

                # (log(rt_i) - mu_log_react)^2
                delta_log_reaction = log_press_offset - mu_log_react

                # (hold_dur_i - sound_dur_i)^2
                delta_duration = hit['u_hold'] - hit['s_hold']

                # Score_react = exp(-lambda * delta_log^2)
                score_reaction = math.exp(-LAMBDA_REACTION * (delta_log_reaction ** 2))

                # Score_dur = exp(-lambda * delta_dur^2)
                score_duration = math.exp(-LAMBDA_DURATION * (delta_duration ** 2))

                beep_score = score_reaction * score_duration
                total_score += beep_score

            if valid_hits == 0:
                final_confidence = 0
            else:
                final_confidence = (total_score / valid_hits) * 100

        else:  # --- 'inverted' rule ---

            # --- NEW: Bot Check for 'inverted' rule ---
            if len(user_intervals) > 1:  # Need at least 2 data points
                hold_durations_ms = [(end - start) * 1000 for start, end in user_intervals]

                # Calculate time *between* holds
                gap_durations_ms = []
                for i in range(len(user_intervals) - 1):
                    gap = (user_intervals[i + 1][0] - user_intervals[i][1]) * 1000
                    if gap > 0:
                        gap_durations_ms.append(gap)

                hold_jitter_ms = calculate_std_dev(hold_durations_ms)
                gap_jitter_ms = calculate_std_dev(gap_durations_ms) if gap_durations_ms else 0

                if hold_jitter_ms < MIN_INVERTED_HOLD_JITTER_MS:
                    flash(
                        f'Verification failed. Unhumanly consistent hold duration ({hold_jitter_ms:.1f}ms jitter). Please try again.',
                        'error')
                    return redirect(url_for('index', type='timing'))

                if gap_durations_ms and gap_jitter_ms < MIN_INVERTED_GAP_JITTER_MS:
                    flash(
                        f'Verification failed. Unhumanly consistent timing between holds ({gap_jitter_ms:.1f}ms jitter). Please try again.',
                        'error')
                    return redirect(url_for('index', type='timing'))

            # --- End of new bot check ---

            # For inverted rule, DGS/Log-Normal model doesn't apply.
            # We use the legacy IoU calculation on the "silence" intervals.
            final_confidence = calculate_timing_confidence_IoU(user_intervals, correct_intervals, total_duration, rule)

        # --- Report Final Score ---
        if final_confidence > 70:  # Require over 70% score
            success_message = f'Success! Your human-likeness score was {final_confidence:.0f}%.'
            if avg_reaction_time_ms > 0:
                success_message += f' Your average reaction time was {avg_reaction_time_ms} ms.'
            flash(success_message, 'success')
        else:
            fail_message = f'Verification failed. Your human-likeness score was {final_confidence:.0f}%.'
            if avg_reaction_time_ms > 0:
                fail_message += f' Your average reaction time was {avg_reaction_time_ms} ms.'
            flash(fail_message, 'error')

    # Redirect back to a new challenge of the same type
    return redirect(url_for('index', type=captcha_type))


# --- Main entry point ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)