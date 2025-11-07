// Background: sky, clouds, ground blocks, and long grass strip
export function createBackground() {
  // Clouds drifting slowly
  for (let i = 0; i < 5; i++) {
    const c = add([
      sprite('cloud'),
      pos(rand(0, width()), rand(10, 120)),
      fixed(),
      opacity(0.9),
      z(-120),
    ]);
    onUpdate(() => {
      c.pos.x += 0.05;
      if (c.pos.x > width() + 24) c.pos.x = -48;
    });
  }

  // Ground tiles along the bottom for visual interest
  const tile = 32;
  const rows = 2; // tile two rows
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < Math.ceil(width() / tile) + 1; x++) {
      add([
        sprite('ground'),
        pos(x * tile, height() - (y + 1) * tile),
        fixed(),
        z(-100),
      ]);
    }
  }

  // Long grass strip overlay
  for (let i = 0; i < Math.ceil(width() / 64) + 1; i++) {
    add([sprite('grass'), pos(i * 64, height() - 48), fixed(), z(-90)]);
  }
}
