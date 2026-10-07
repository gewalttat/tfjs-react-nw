import { useCallback, useEffect, useRef, useState } from 'react';

/** Procedural audio: no downloads, autoplay, or external assets. */
export function useAmbience(experiment: string) {
  const busy = useRef(false);
  const active = useRef(true);
  const [starting, setStarting] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const engine = useRef<{ context: AudioContext; master: GainNode; dispose: () => void } | null>(null);
  const chime = useCallback(() => {
    const audio = engine.current;
    if (!audio || audio.context.state !== 'running') return;
    [220, 330, 440, 660].forEach((frequency, index) => {
      const oscillator = audio.context.createOscillator(), gain = audio.context.createGain();
      const start = audio.context.currentTime + index * .085;
      oscillator.type = 'sine'; oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0, start); gain.gain.linearRampToValueAtTime(.045, start + .025);
      gain.gain.exponentialRampToValueAtTime(.0001, start + .65);
      oscillator.connect(gain); gain.connect(audio.master);
      oscillator.start(start); oscillator.stop(start + .7);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
    });
  }, []);
  useEffect(() => { if (enabled) chime(); }, [experiment, enabled, chime]);
  useEffect(() => {
    active.current = true;
    const visibility = () => {
      const context = engine.current?.context;
      if (context) (document.hidden ? context.suspend() : context.resume()).catch(() => {});
    };
    document.addEventListener('visibilitychange', visibility);
    return () => { active.current = false; document.removeEventListener('visibilitychange', visibility); engine.current?.dispose(); engine.current = null; };
  }, []);
  async function toggle() {
    if (busy.current) return;
    if (engine.current) { engine.current.dispose(); engine.current = null; setEnabled(false); return; }
    busy.current = true; setStarting(true);
    let context: AudioContext | undefined;
    try {
      const Constructor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!Constructor) { setUnavailable(true); return; }
      context = new Constructor();
      await context.resume();
      if (!active.current) { await context.close(); return; }
      const ctx = context;
      const master = ctx.createGain(); master.gain.value = .3; master.connect(ctx.destination);
      const buffer = ctx.createBuffer(1, ctx.sampleRate * 3, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      let previous = 0;
      for (let i = 0; i < data.length; i++) { previous = (previous + (Math.random() * 2 - 1) * .025) / 1.025; data[i] = previous * 3; }
      const wind = ctx.createBufferSource(), filter = ctx.createBiquadFilter(), windGain = ctx.createGain();
      wind.buffer = buffer; wind.loop = true; filter.type = 'lowpass'; filter.frequency.value = 380;
      windGain.gain.value = .12; wind.connect(filter); filter.connect(windGain); windGain.connect(master); wind.start();
      const tones = [110, 164.81, 220].map((frequency) => {
        const oscillator = ctx.createOscillator(), gain = ctx.createGain();
        oscillator.frequency.value = frequency; oscillator.type = 'sine'; gain.gain.value = .012;
        oscillator.connect(gain); gain.connect(master); oscillator.start(); return { oscillator, gain };
      });
      // Gentle pitch bends and breath-shaped envelopes avoid a metronomic loop.
      const call = (frequency: number, start: number, duration: number, volume: number, owl: boolean) => {
        const voice = ctx.createOscillator(), gain = ctx.createGain();
        voice.type = 'sine';
        voice.frequency.setValueAtTime(frequency * (owl ? .94 : 1), start);
        voice.frequency.linearRampToValueAtTime(frequency * (owl ? 1.04 : 1.015), start + duration * .3);
        voice.frequency.linearRampToValueAtTime(frequency * (owl ? .88 : .99), start + duration);
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(volume, start + (owl ? .09 : .008));
        gain.gain.exponentialRampToValueAtTime(.0001, start + duration);
        voice.connect(gain); gain.connect(master);
        voice.start(start); voice.stop(start + duration + .02);
        voice.onended = () => { voice.disconnect(); gain.disconnect(); };
      }
      let timer = 0, disposed = false;
      const schedule = (delay: number) => {
        timer = window.setTimeout(() => {
          if (disposed) return;
          if (ctx.state === 'running' && !document.hidden) {
            const start = ctx.currentTime;
            if (Math.random() < .35) {
              const pitch = 290 + Math.random() * 65;
              call(pitch, start, .45, .07, true);
              call(pitch * .96, start + .65 + Math.random() * .2, .7, .055, true);
            } else {
              const pitch = 3700 + Math.random() * 900;
              const pulses = 3 + Math.floor(Math.random() * 4);
              const spacing = .13 + Math.random() * .04;
              for (let i = 0; i < pulses; i++) call(pitch, start + i * spacing, .055, .012 + Math.random() * .008, false);
            }
          }
          schedule(8000 + Math.random() * 14000);
        }, delay);
      }
      schedule(2000 + Math.random() * 4000);
      engine.current = { context: ctx, master, dispose: () => {
        disposed = true; window.clearTimeout(timer); wind.stop(); wind.disconnect(); filter.disconnect(); windGain.disconnect();
        tones.forEach(({ oscillator, gain }) => { oscillator.stop(); oscillator.disconnect(); gain.disconnect(); });
        master.disconnect(); void ctx.close().catch(() => {});
      } };
      setEnabled(true); setUnavailable(false);
    } catch { if (context) void context.close().catch(() => {}); if (active.current) setUnavailable(true); }
    finally { busy.current = false; if (active.current) setStarting(false); }
  }
  return { enabled, starting, unavailable, toggle };
}
