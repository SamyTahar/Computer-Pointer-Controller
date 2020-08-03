#! python3
import pyautogui, sys
print('Press Ctrl-C to quit.')
try:
    while True:
        screenWidth, screenHeight = pyautogui.size() # Returns two integers, the width and height of the screen. (The primary monitor, in multi-monitor setups.)
        currentMouseX, currentMouseY = pyautogui.position() # Returns two integers, the x and y of the mouse cursor's current position.
        pyautogui.moveTo(100, 200)   # moves mouse to X of 100, Y of 200.
        pyautogui.moveTo(None, 500)  # moves mouse to X of 100, Y of 500.
        pyautogui.moveTo(600, None)

except KeyboardInterrupt:
    print('\n')