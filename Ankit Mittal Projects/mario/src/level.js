// Level utilities: platforms, coins, camera follow, enemies
import { COLORS } from './constants.js';

export function createPlatform(x, y, tilesWide = 3, spriteName = 'ground') {
  const tile = 32;
  for (let i = 0; i < tilesWide; i++) {
    add([
      sprite(spriteName),
      pos(x + i * tile, y),
      area(),
      body({ isStatic: true }),
    ]);
  }
}

export function createBrickPlatform(x, y, tilesWide = 3, tilesHigh = 1) {
  const tile = 32;
  for (let yy = 0; yy < tilesHigh; yy++) {
    for (let xx = 0; xx < tilesWide; xx++) {
      add([
        sprite('block'),
        pos(x + xx * tile, y + yy * tile),
        area(),
        body({ isStatic: true }),
      ]);
    }
  }
}

export function createDefaultLevel() {
  // Long ground path
  for (let i = 0; i < 60; i++) {
    createPlatform(i * 32, 360, 1, 'ground');
  }
  // Elevated platforms
  createPlatform(140, 280, 4, 'block');
  createPlatform(320, 220, 5, 'block');
  createPlatform(520, 160, 3, 'block');
  createPlatform(700, 220, 6, 'block');
  createPlatform(980, 280, 6, 'block');
}

export function spawnCoins(positions) {
  positions.forEach((p) => {
    add([sprite('coin'), pos(p), area(), anchor('center'), scale(1.2), 'coin']);
  });
}

export function spawnCoinsValued(positions) {
  const variants = [
    { sprite: 'coin_bronze', value: 10 },
    { sprite: 'coin_silver', value: 20 },
    { sprite: 'coin_gold', value: 30 },
  ];
  positions.forEach((p) => {
    const v = variants[randi(0, variants.length)];
    add([
      sprite(v.sprite),
      pos(p),
      area(),
      anchor('center'),
      scale(1.2),
      { coinValue: v.value },
      'coin',
    ]);
  });
}

export function spawnEnemies(ranges) {
  ranges.forEach(({ x, y, width }) => {
    const enemy = add([
      sprite('enemy'),
      pos(x, y - 14),
      area({ height: 12 }),
      body(),
      scale(1.5),
      anchor('botleft'),
      { dir: 1, left: x, right: x + width },
      'enemy',
    ]);
    enemy.onUpdate(() => {
      enemy.move(60 * enemy.dir, 0);
      if (enemy.pos.x < enemy.left) enemy.dir = 1;
      if (enemy.pos.x > enemy.right) enemy.dir = -1;
    });
  });
}

export function followCamera(target) {
  onUpdate(() => {
    camPos(vec2(Math.max(target.pos.x, 200), 200));
  });
}

// Multi-level utilities: addLevel mapping, spawners, patrol enemies, fire enemies

const TILE = 32; // align with 32x32 ground/block sprites
const BASELINE_Y = 360; // fixed baseline so bottom row is visible

export function levelDefs() {
  const maps = [
    [
      '                                                                    ',
      '     $      E              $   $        E              $   $     ',
      '   ===   SSSS     $     F  === SSSS     $     F  === SSSS     $  ',
      '             ==       ===        ==       ===        ==       === ',
      '  @     G       ===        G       ===        G       ===        ',
      '==================================================================',
    ],
    [
      '                                                                      ',
      '      $$         G               $$         G               $$      ',
      '  SSS===      SSSS     $     F  ===      SSSS     $     F  ===     ',
      '          ==            ===            ==            ===            ',
      '  @   G         E               G         E               G        ',
      '====================================================================',
    ],
    [
      '                                                                        ',
      '   $   E      $$$         G           $   E      $$$         G        ',
      '  SSSS   ===        SSS     $     F  SSSS   ===        SSS     $     ',
      '            ==                 ===            ==                 ===   ',
      '  @      G         G                  G         G                  G  ',
      '======================================================================',
    ],
  ];
  return maps;
}

export function levelConfig() {
  return {
    tileWidth: 16,
    tileHeight: 16,
    tiles: {
      '=': () => [
        sprite('ground'),
        area(),
        body({ isStatic: true }),
        anchor('botleft'),
      ],
      S: () => [
        sprite('stair'),
        area(),
        body({ isStatic: true }),
        anchor('botleft'),
      ],
      '@': () => ['spawn_player', anchor('botleft')],
      $: () => ['spawn_coin', anchor('center')],
      G: () => ['spawn_enemy', anchor('botleft')],
      E: () => ['spawn_fire_enemy', anchor('botleft')],
      F: () => ['spawn_flag', anchor('botleft')],
    },
  };
}

export function spawnCoinAt(p) {
  const variants = [
    { sprite: 'coin_bronze', value: 10 },
    { sprite: 'coin_silver', value: 20 },
    { sprite: 'coin_gold', value: 30 },
  ];
  const v = variants[randi(0, variants.length)];
  return add([
    sprite(v.sprite),
    pos(p),
    area(),
    scale(1.2),
    { coinValue: v.value },
    'coin',
  ]);
}

export function spawnEnemyGoombaAt(p) {
  const enemy = add([
    sprite('enemy'),
    pos(p),
    area(),
    body(),
    scale(1.5),
    anchor('bot'),
    { dir: 1, left: p.x - 60, right: p.x + 60 },
    'enemy',
  ]);
  enemy.onUpdate(() => {
    enemy.move(60 * enemy.dir, 0);
    if (enemy.pos.x < enemy.left) enemy.dir = 1;
    if (enemy.pos.x > enemy.right) enemy.dir = -1;
  });
  return enemy;
}

export function spawnEnemyFireAt(p) {
  const e = add([
    sprite('enemy_fire'),
    pos(p),
    area(),
    body({ isStatic: true }),
    anchor('bot'),
    'enemy_fire',
  ]);
  loop(1.5, () => {
    add([
      sprite('fireball'),
      pos(e.pos.add(8, -8)),
      area(),
      move(LEFT, 70),
      lifespan(3),
      'fireball',
    ]);
  });
  return e;
}

export function spawnFlagAt(p) {
  // Create a taller flag pole with multiple segments
  const flagPole = add([
    sprite('flag'),
    pos(p.x, p.y - TILE * 4),
    area(),
    anchor('botleft'),
    scale(1.2),
    'flag',
  ]);
  // Add flag segments to make it taller
  for (let i = 1; i < 4; i++) {
    add([
      sprite('flag'),
      pos(p.x, p.y - TILE * (4 - i)),
      area(),
      anchor('botleft'),
      scale(1.2),
    ]);
  }
  return flagPole;
}

export function makeLevel(m) {
  const spawns = { player: null };
  // Place bottom row at fixed baseline
  const bottomRowIndex = m.length - 1;
  m.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const worldY = BASELINE_Y - (bottomRowIndex - y) * TILE;
      const world = vec2(x * TILE, worldY);
      switch (ch) {
        case '=':
          add([
            sprite('ground'),
            pos(world),
            area(),
            body({ isStatic: true }),
            anchor('botleft'),
          ]);
          break;
        case 'S':
          // Create proper stairs - stacked blocks
          for (let i = 0; i < 3; i++) {
            add([
              sprite('stair'),
              pos(world.x, world.y - i * TILE),
              area(),
              body({ isStatic: true }),
              anchor('botleft'),
            ]);
          }
          break;
        case '@':
          spawns.player = vec2(world.x + TILE / 2, world.y - TILE);
          break;
        case '$':
          spawnCoinAt(vec2(world.x + TILE / 2, world.y - TILE / 2));
          break;
        case 'G':
          spawnEnemyGoombaAt(vec2(world.x + TILE / 2, world.y));
          break;
        case 'E':
          spawnEnemyFireAt(vec2(world.x + TILE / 2, world.y));
          break;
        case 'F':
          spawnFlagAt(vec2(world.x, world.y));
          break;
      }
    }
  });
  return spawns;
}
