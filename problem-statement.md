What is a satellite image to a computer?

To a human, a satellite images shows rivers, trees, and tin roofs. To a computer, a satellite image is nothing but a massive grid of numbers arranged in rows and columns.

For a standard color image, every single pixel contains three numbers:
1. How much RED is in a dot(0-255)
2. How much GREEN is in a dot(0-255)
3. How much BLUE is in a dot (0-255)

RGB Data.

The core problem:
In informal settlements e.g. Kasarani houses are built so close together that their corrugated iron roofs touch.
- a  basic computer model looks at the pixel, sees a massive, continous block of numbers.
- the computer clumps the houses together into one giant blob.
- a basic government model counts the blobs resulting to a count of 118.

Our solution innovation.

We break this problem down into a two step process to solve the blob limitation.
1. The Material Highlighter(Random Forest):
- we look at pixels one by one to isolate the exact color and texture of corrugated iron.

2. Structure Boundary Tracer (YOLOv8-seg)
- we train a neural network to look at patterns, scanning for the tiny dark shadow lines and angular corners where one roof ends and another begins.
