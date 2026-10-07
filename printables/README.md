# Printable manipulation objects

STL coordinates are **millimeters**. Print upright with the flat bottom on the bed; these solids need no supports. Color is supplied by your filament: print a copy in red, green, blue, and yellow. There is no embedded STL color.

- `cube_30mm.stl`: baseline 30 mm cube, matching the simulator.
- `cylinder_30mm.stl`: baseline 30 mm diameter × 30 mm height cylinder, matching the simulator.
- 24 mm and 36 mm variants: size shifts for held-out evaluation. Check the actual gripper opening before printing a large batch.
- `goal_disk_80mm.stl`: 80 mm diameter × 2 mm thick target pad, matching the simulated disk. Print multiple colors for sorting; a pad is easier to reproduce than a bin.

Suggested starting print settings: PLA, matte filament, 0.2 mm layers, 3 perimeters, 10–15% infill. Measure finished dimensions and mass; the simulator uses 15 g objects, an approximation rather than a measured print specification. Use consistent pad/surface friction across initial experiments. Record filament, mass, dimensions and gripper contact material with each collection session.

Run `python printables/generate.py` to regenerate all seven watertight meshes. These are project-created primitives, not replicas of the published datasets' exact objects. See [the experiment plan](../docs/SO101_MANIPULATION.md) for collection and evaluation.
