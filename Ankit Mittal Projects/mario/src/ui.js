// UI helpers: score / deaths / level HUD at top-left
export function createHUD(initial = { score: 0, deaths: 0, level: 1 }) {
	let score = initial.score ?? 0;
	let deaths = initial.deaths ?? 0;
	let level = initial.level ?? 1;

	const hud = add([
		text("Score: 0\nDeaths: 0\nLevel: 1", { size: 18, lineSpacing: 6 }),
		pos(12, 12),
		anchor("topleft"),
		color(rgb(250, 250, 250)),
		fixed(),
		z(1000),
	]);

	function render() {
		hud.text = `Score: ${score}\nDeaths: ${deaths}\nLevel: ${level}`;
	}
	render();

	return {
		addScore(n = 1) { score += n; render(); },
		addDeath(n = 1) { deaths += n; render(); },
		setLevel(n) { level = n; render(); },
		getState() { return { score, deaths, level }; },
		spawnFloatText(value, at) {
			const t = add([
				text(`+${value}`, { size: 14 }),
				pos(at),
				color(rgb(255, 255, 255)),
				outline(2, rgb(40, 40, 40)),
				z(100),
			]);
			const startY = t.pos.y;
			const life = 0.6;
			let elapsed = 0;
			onUpdate(() => {
				elapsed += dt();
				t.pos.y = startY - elapsed * 40; // float up
				t.opacity = 1 - Math.min(1, elapsed / life); // fade out
				if (elapsed >= life) destroy(t);
			});
		},
	};
}

export function setupHelpOverlay() {
	// HTML help remains
}
