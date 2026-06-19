# Pythoncraft

Pythoncraft is an open-source, voxel-based sandbox engine built with Python, Pyglet, and modern OpenGL (GLSL shaders). It features advanced multi-threaded chunk generation, custom lighting and shadow systems, water/air physics accurately mimicking classic mechanics, and a highly optimized render pipeline.

<img width="2559" height="1367" alt="image" src="https://github.com/user-attachments/assets/310551e9-c8ba-45ef-84f9-56abeda0b851" />

<img width="2559" height="1391" alt="image" src="https://github.com/user-attachments/assets/10969f8e-4729-4f27-be38-3edec143ecd3" />

<img width="1273" height="740" alt="image" src="https://github.com/user-attachments/assets/05451d93-cdef-4674-9a6e-bd650aeeb500" />


## 🚀 Features
- **Infinite World Generation**: Multi-threaded terrain and biome generation using Perlin Noise.
- **Advanced Physics**: Tick-based physics engine with highly accurate movement and swimming mechanics.
- **Lighting Engine**: Smooth lighting, ambient occlusion, and transparency support.
- **High Performance**: Numba JIT optimizations, frustum culling, and VAO/VBO-based block rendering.
- **Voxel Mechanics**: Dynamic block placement and destruction.

## 🛠️ Installation & Running

1. Clone the repository:
   ```bash
   git clone https://github.com/kivi-man/Python-craft.git
   cd pythoncraft
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the game:
   ```bash
   python main.py
   ```

*(You can also use the included `guncelleme_araci.bat` file to automatically pull the latest updates from the community and start the game).*

### ⚙️ Command-Line Arguments

The engine supports optional command-line parameters to customize performance and rendering:
- **Render Distance**: Pass an integer to set the render distance. For example, `python main.py 6` sets it to 6 chunks. Default is 4.
- **Fast Leaves (`-fast`)**: Forces leaf blocks to be opaque, which significantly boosts FPS on low-end machines.
- **Debug Mode (`-debug`)**: Enables detailed performance metrics, printouts, and rendering logs in the console.

You can combine these arguments:
```bash
python main.py -fast -debug 6
```

### 🎮 Controls & Interface

- **Movement**:
  - `W`, `A`, `S`, `D`: Walk around.
  - `Space`: Jump / Fly Up (press `Tab` to toggle fly mode).
  - `Shift`: Crouch / Fly Down.
  - `ESC`: Release mouse and exit.
- **Voxel Placement & Breaking**:
  - **Left Click**: Break targeted block (triggers held block swing animation).
  - **Right Click**: Place selected block (triggers held block swing animation).
- **Hotbar & Item Selection**:
  - **Keys `1` to `6`**: Directly select slots 1 through 6 in your hotbar.
  - **Mouse Scroll Wheel**: Scroll up or down to cycle through your hotbar slots.
- **HUD & Hand Viewmodel**:
  - **Crosshair**: A pixel-perfect, color-inverting crosshair centered on the screen.
  - **Hotbar**: A Minecraft-authentic quick-select hotbar showing 3D isometric previews of selectable blocks (Stone, Grass, Glass, Leaves, Water, Cactus).
  - **3D Held Block**: Shows a 3D model of the selected block in your hand with idle breathing, y-axis walking bobbing, and swing animations when clicking.

## 🤝 Community Support & Update Guidelines

We welcome community support, pull requests, and forks! However, this project is governed by strict update rules to maintain its vision:

**Update Rule:** Any new feature, mechanic, block, or system proposed in a Pull Request **MUST** be a system that currently exists in Minecraft. Custom or non-vanilla systems will not be accepted into the main branch.

## ⚖️ License & Attribution

This project is open-source and free to use, modify, and fork under the provided License.

**CRITICAL MANDATORY RULE:** 
If you fork, redistribute, or use this code in your own project, **you must explicitly credit the original creator: the GitHub user `Kivi-man`**. You must also clearly state that the code was originally obtained from this repository. Failure to provide this attribution violates the terms of the License.
