# Image Editing Test Data

## Files
- `empty_room_top_view.png` — 768×768 top-down view of an empty living room (4.5m × 5.0m)
- `prompt.txt` — Full prompt (2737 characters) used by scenesmith for furniture context generation

## Task
Edit the empty room image to add furniture according to the scene description in the prompt, 
while preserving the exact architectural layout (walls, doors, windows, floor boundaries).

## Expected Output
A top-down view with furniture added (sofas, coffee table, armchairs, plants, side table), 
maintaining the same viewpoint, canvas size, and room architecture.

## Reference
- Original okcodex response time: ~35-55 seconds with full prompt
- Image format: PNG, 768×768
- Source: scenesmith critic_probe run okcodex_test_20260806_111035
