// Simple WebAudio beeps to avoid loading/decoding audio files
let audioCtx = null;

function ensureCtx() {
	if (!audioCtx) {
		audioCtx = new (window.AudioContext || window.webkitAudioContext)();
	}
	return audioCtx;
}

function beep({ freq = 440, duration = 0.12, type = "square", volume = 0.15 } = {}) {
	const ctx = ensureCtx();
	const t = ctx.currentTime;
	const osc = ctx.createOscillator();
	const gain = ctx.createGain();
	osc.type = type;
	osc.frequency.value = freq;
	gain.gain.setValueAtTime(0, t);
	gain.gain.linearRampToValueAtTime(volume, t + 0.01);
	gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);
	osc.connect(gain).connect(ctx.destination);
	osc.start(t);
	osc.stop(t + duration + 0.02);
}

export function playJump() {
	// Upward blip
	beep({ freq: 700, duration: 0.12, type: "square", volume: 0.18 });
	setTimeout(() => beep({ freq: 900, duration: 0.09, type: "square", volume: 0.14 }), 60);
}

export function playCoin() {
	// Bright ding
	beep({ freq: 1200, duration: 0.08, type: "triangle", volume: 0.16 });
	setTimeout(() => beep({ freq: 1600, duration: 0.06, type: "triangle", volume: 0.12 }), 50);
}

export function playStep() {
	// Soft tick
	beep({ freq: 220, duration: 0.06, type: "sawtooth", volume: 0.08 });
}
