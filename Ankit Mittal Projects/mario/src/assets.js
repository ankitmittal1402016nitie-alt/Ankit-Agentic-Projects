// Asset loader: registers inline SVG sprites to avoid CORS issues
// We use small SVG data URLs for visuals; audio is synthesized in sound.js

export async function loadAssets() {
	// Modern-styled pixel Mario (still minimal, more colors and shape)
	const marioSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' shape-rendering='crispEdges'>
			<rect width='20' height='20' fill='#0000' />
			<!-- hat -->
			<rect x='5' y='2' width='10' height='2' fill='#B71C1C'/>
			<rect x='6' y='4' width='8' height='1' fill='#D32F2F'/>
			<!-- face -->
			<rect x='6' y='5' width='8' height='4' fill='#F4C28B'/>
			<rect x='6' y='6' width='2' height='1' fill='#3E2723'/>
			<!-- body (overalls) -->
			<rect x='5' y='9' width='10' height='6' fill='#1565C0'/>
			<rect x='8' y='10' width='1' height='2' fill='#FFEB3B'/>
			<rect x='11' y='10' width='1' height='2' fill='#FFEB3B'/>
			<!-- boots -->
			<rect x='5' y='15' width='4' height='3' fill='#5D4037'/>
			<rect x='11' y='15' width='4' height='3' fill='#5D4037'/>
		</svg>
	`);

	// Coins
	const coinGoldSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'>
			<circle cx='9' cy='9' r='8' fill='#FDD835' stroke='#F9A825' stroke-width='2'/>
			<circle cx='9' cy='9' r='5' fill='#FFEB3B' stroke='#FBC02D' stroke-width='1'/>
			<path d='M5 7 Q9 3 13 7' stroke='#FFF59D' stroke-width='1' fill='none'/>
			<rect x='8' y='5' width='2' height='8' fill='#FFF176'/>
		</svg>
	`);
	const coinSilverSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'>
			<circle cx='9' cy='9' r='8' fill='#CFD8DC' stroke='#90A4AE' stroke-width='2'/>
			<circle cx='9' cy='9' r='5' fill='#ECEFF1' stroke='#B0BEC5' stroke-width='1'/>
			<path d='M5 7 Q9 3 13 7' stroke='#FFFFFF' stroke-width='1' fill='none'/>
			<rect x='8' y='5' width='2' height='8' fill='#FFFFFF'/>
		</svg>
	`);
	const coinBronzeSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'>
			<circle cx='9' cy='9' r='8' fill='#FFB74D' stroke='#F57C00' stroke-width='2'/>
			<circle cx='9' cy='9' r='5' fill='#FFCC80' stroke='#FB8C00' stroke-width='1'/>
			<path d='M5 7 Q9 3 13 7' stroke='#FFE0B2' stroke-width='1' fill='none'/>
			<rect x='8' y='5' width='2' height='8' fill='#FFE0B2'/>
		</svg>
	`);

	// Enemy (simple goomba-like)
	const enemySVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='18' height='14' shape-rendering='crispEdges'>
			<rect width='18' height='14' fill='#0000'/>
			<rect x='2' y='4' width='14' height='8' fill='#8D6E63'/>
			<rect x='4' y='6' width='3' height='2' fill='#000000'/>
			<rect x='11' y='6' width='3' height='2' fill='#000000'/>
			<rect x='4' y='12' width='4' height='2' fill='#5D4037'/>
			<rect x='10' y='12' width='4' height='2' fill='#5D4037'/>
		</svg>
	`);

	// Fire enemy (stationary) and fireball
	const fireEnemySVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='18' height='16' shape-rendering='crispEdges'>
			<rect width='18' height='16' fill='#0000'/>
			<rect x='3' y='6' width='12' height='8' fill='#6A1B9A'/>
			<rect x='5' y='4' width='8' height='4' fill='#8E24AA'/>
		</svg>
	`);
	const fireballSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'>
			<circle cx='5' cy='5' r='4' fill='#FF5722'/>
			<circle cx='4' cy='4' r='2' fill='#FF9800'/>
		</svg>
	`);

	// Flag and stair
	const flagSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='10' height='24' shape-rendering='crispEdges'>
			<rect x='4' y='0' width='2' height='24' fill='#795548'/>
			<polygon points='6,4 6,12 16,8' fill='#43A047'/>
		</svg>
	`);
	const stairSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' shape-rendering='crispEdges'>
			<rect width='16' height='16' fill='#795548'/>
			<rect x='0' y='8' width='16' height='8' fill='#6D4C41'/>
		</svg>
	`);

	// Tiles & background
	const groundSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' shape-rendering='crispEdges'>
			<rect width='32' height='32' fill='#8D6E63'/>
			<rect width='32' height='10' y='0' fill='#6D4C41'/>
			<circle cx='6' cy='14' r='1' fill='#6D4C41'/>
			<circle cx='16' cy='18' r='1' fill='#6D4C41'/>
			<circle cx='26' cy='13' r='1' fill='#6D4C41'/>
			<circle cx='10' cy='26' r='1' fill='#6D4C41'/>
		</svg>
	`);
	const blockSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' shape-rendering='crispEdges'>
			<rect width='32' height='32' fill='#795548'/>
			<path d='M0 10h32M0 20h32M8 0v32M24 0v32' stroke='#4E342E' stroke-width='2'/>
		</svg>
	`);
	const cloudSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='48' height='24'>
			<ellipse cx='16' cy='14' rx='14' ry='8' fill='#FAFAFA'/>
			<ellipse cx='28' cy='12' rx='12' ry='8' fill='#FFFFFF'/>
			<ellipse cx='36' cy='16' rx='10' ry='6' fill='#F5F5F5'/>
		</svg>
	`);
	const grassSVG = encodeSVG(`
		<svg xmlns='http://www.w3.org/2000/svg' width='64' height='16'>
			<rect width='64' height='16' fill='#43A047'/>
			<path d='M0 16L4 8L8 16M8 16L12 9L16 16M16 16L20 7L24 16M24 16L28 9L32 16M32 16L36 8L40 16M40 16L44 9L48 16M48 16L52 8L56 16M56 16L60 9L64 16' stroke='#2E7D32' stroke-width='2'/>
		</svg>
	`);

	await Promise.all([
		loadSprite("mario2", marioSVG),
		loadSprite("coin_gold", coinGoldSVG),
		loadSprite("coin_silver", coinSilverSVG),
		loadSprite("coin_bronze", coinBronzeSVG),
		loadSprite("enemy", enemySVG),
		loadSprite("enemy_fire", fireEnemySVG),
		loadSprite("fireball", fireballSVG),
		loadSprite("flag", flagSVG),
		loadSprite("stair", stairSVG),
		loadSprite("ground", groundSVG),
		loadSprite("block", blockSVG),
		loadSprite("cloud", cloudSVG),
		loadSprite("grass", grassSVG),
	]);
}

function encodeSVG(svg) {
	const encoded = svg
		.replace(/\n/g, "")
		.replace(/\t/g, "")
		.replace(/\s{2,}/g, " ")
		.replace(/#/g, "%23")
		.replace(/</g, "%3C")
		.replace(/>/g, "%3E");
	return `data:image/svg+xml,${encoded}`;
}
