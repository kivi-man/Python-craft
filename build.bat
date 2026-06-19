@echo off
echo Building Pythoncraft...
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --noconfirm --onedir --windowed --add-data "assets;assets" --add-data "shaders;shaders" main.py
echo Build complete. Executable is in the 'dist' folder.
pause
