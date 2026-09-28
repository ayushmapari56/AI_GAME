import pygame
import numpy as np
import math

class SoundManager:
    """
    Procedural audio generator for arcade sounds without requiring external .wav/.mp3 files.
    Generates pure waveforms dynamically with automatic graceful fallback.
    """
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.enabled = False
        self.sounds = {}
        
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
            self.enabled = True
            self._generate_sound_effects()
        except Exception as e:
            # Headless or audio device not present
            self.enabled = False

    def _generate_sound_effects(self):
        try:
            self.sounds["collect"] = self._create_synth_sound(freq_start=587.33, freq_end=880.0, duration=0.08, wave_type="sine")
            self.sounds["powerup"] = self._create_synth_sound(freq_start=350.0, freq_end=1050.0, duration=0.25, wave_type="triangle")
            self.sounds["emp"] = self._create_synth_sound(freq_start=800.0, freq_end=120.0, duration=0.35, wave_type="sawtooth")
            self.sounds["alert"] = self._create_synth_sound(freq_start=440.0, freq_end=440.0, duration=0.10, wave_type="square")
            self.sounds["gameover"] = self._create_synth_sound(freq_start=280.0, freq_end=70.0, duration=0.6, wave_type="sawtooth")
            self.sounds["victory"] = self._create_synth_sound(freq_start=440.0, freq_end=880.0, duration=0.45, wave_type="sine")
        except Exception:
            pass

    def _create_synth_sound(self, freq_start: float, freq_end: float, duration: float, wave_type: str = "sine") -> pygame.mixer.Sound:
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Frequency sweep
        freq = np.linspace(freq_start, freq_end, num_samples)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        if wave_type == "sine":
            waveform = np.sin(phase)
        elif wave_type == "square":
            waveform = np.sign(np.sin(phase))
        elif wave_type == "triangle":
            waveform = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif wave_type == "sawtooth":
            waveform = 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        else:
            waveform = np.sin(phase)

        # Decay envelope
        envelope = np.linspace(1.0, 0.0, num_samples)**1.5
        waveform = waveform * envelope * 0.35 # volume scale

        # Convert to 16-bit stereo PCM
        audio = (waveform * 32767).astype(np.int16)
        stereo_audio = np.column_stack((audio, audio))
        return pygame.sndarray.make_sound(stereo_audio)

    def play(self, sound_name: str):
        if not self.enabled:
            return
        sound = self.sounds.get(sound_name)
        if sound:
            try:
                sound.play()
            except Exception:
                pass
