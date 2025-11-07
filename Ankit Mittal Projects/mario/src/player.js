// Player entity factory. Uses the improved mario2 sprite avatar.
// Returns the created kaboom game object.
import { JUMP_FORCE, MOVE_SPEED } from "./constants.js";
import { playJump, playStep } from "./sound.js";

export function createPlayer(startX = 80, startY = 0) {
	const player = add([
		sprite("mario2"),
		pos(startX, startY),
		area(),
		body(),
		scale(2),
		anchor("center"),
		outline(2, rgb(40, 40, 40)),
		z(10),
		"player",
	]);
	return player;
}

// Movement system using key polling (reliable in iframes like StackBlitz)
export function useMovement(player) {
	let stepAccum = 0;
	onUpdate(() => {
		const moveRight = isKeyDown("right") || isKeyDown("d");
		const moveLeft = isKeyDown("left") || isKeyDown("a");
		const directionX = (moveRight ? 1 : 0) - (moveLeft ? 1 : 0);
		if (directionX !== 0) {
			player.move(directionX * MOVE_SPEED, 0);
			// Soft step while grounded
			if (player.isGrounded()) {
				stepAccum += dt();
				if (stepAccum > 0.35) {
					playStep();
					stepAccum = 0;
				}
			}
		} else {
			stepAccum = 0;
		}
	});
}

// Jump helpers
export function useJump(player) {
	function tryJump() {
		if (player.isGrounded()) {
			player.jump(JUMP_FORCE);
			playJump();
		}
	}
	onKeyPress(["space", "up", "w"], tryJump);
	onUpdate(() => {
		if ((isKeyDown("space") || isKeyDown("up") || isKeyDown("w")) && player.isGrounded()) {
			player.jump(JUMP_FORCE);
			playJump();
		}
	});
}
