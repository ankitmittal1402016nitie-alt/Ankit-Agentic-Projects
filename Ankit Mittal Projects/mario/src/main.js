// Scenes-based multi-level game

console.log('Starting to import Kaboom...');
let kaboom;
try {
    const kaboomModule = await import('https://cdn.jsdelivr.net/npm/kaboom@3000.1.17/dist/kaboom.mjs');
    console.log('Kaboom module loaded:', kaboomModule);
    kaboom = kaboomModule.default;
    console.log('Kaboom initialized successfully');
} catch (error) {
    console.error('Error loading Kaboom:', error);
    throw error; // Re-throw the error to prevent continuing if Kaboom fails to load
}
import { createPlayer, useMovement, useJump } from './player.js';
import { levelDefs, makeLevel } from './level.js'; 
import { createHUD, setupHelpOverlay } from './ui.js';
import { loadAssets } from './assets.js';
import { createBackground } from './background.js';
import { playCoin } from './sound.js';
 
kaboom({
  canvas: document.getElementById('game'),
  background: [135, 206, 250],
});
setGravity(1200);
await loadAssets();

let score = 0;
let deaths = 0;
let currentLevel = 0;

scene('start', () => {
  add([
    text('Mario Platformer\nPress Space to Start', {
      size: 24,
      lineSpacing: 8,
    }),
    pos(center()),
    anchor('center'),
    color(rgb(255, 255, 255)),
  ]);
  onKeyPress('space', () => go('level', currentLevel));
});

scene('win', () => {
  add([
    text(`You Win!\nScore: ${score}\nDeaths: ${deaths}`, {
      size: 24,
      lineSpacing: 8,
    }),
    pos(center()),
    anchor('center'),
    color(rgb(255, 255, 255)),
  ]);
  onKeyPress('space', () => {
    score = 0;
    deaths = 0;
    currentLevel = 0;
    go('start');
  });
});

scene('level', (idx) => {
  createBackground();
  const maps = levelDefs();
  const map = maps[idx] ?? maps[maps.length - 1];
  const spawns = makeLevel(map);

  const player = createPlayer(spawns.player?.x ?? 80, spawns.player?.y ?? 0);
  useMovement(player);
  useJump(player);

  const hud = createHUD({ score, deaths, level: idx + 1 });

  function handleDeath() {
    deaths += 1;
    hud.addDeath(1);
    go('level', idx);
  }

  // Camera follow and bounds
  onUpdate(() => {
    camPos(vec2(Math.max(player.pos.x, 200), 200));
    if (player.pos.y > height()) handleDeath();
  });

  // Coin collection
  onCollide('player', 'coin', (p, c, col) => {
    const value = c.coinValue ?? 10;
    score += value;
    hud.addScore(value);
    hud.spawnFloatText(value, c.pos.clone());
    playCoin(); // Restore coin sound
    destroy(c);
  });

  // Enemy stomp / damage
  onCollide('player', 'enemy', (p, e, col) => {
    if (col && col.isBottom()) {
      destroy(e);
      p.jump(500);
    } else {
      handleDeath();
    }
  });

  // Fire enemy - always kills Mario on any contact
  onCollide('player', 'enemy_fire', () => handleDeath());

  // Fireball damage
  onCollide('player', 'fireball', () => handleDeath());

  // Flag win
  onCollide('player', 'flag', (p, f) => {
    // Flag slides down smoothly with sound
    playCoin(); // Use coin sound as flag sound for now
    tween(
      f.pos.y,
      f.pos.y + 100,
      1,
      (v) => (f.pos.y = v),
      easings.easeInOutSine
    );
    wait(1.2, () => {
      const maps2 = levelDefs();
      if (idx + 1 < maps2.length) {
        currentLevel = idx + 1;
        go('level', currentLevel);
      } else {
        go('win');
      }
    });
  });

  // Focus
  const cv = document.getElementById('game');
  if (cv) {
    cv.tabIndex = 0;
    cv.focus();
  }
});

setupHelpOverlay();

// Start directly at start scene
go('start');
