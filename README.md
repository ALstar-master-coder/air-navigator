\# Air Gesture Controller (Air Scroll \& Mouse)



A touchless hand gesture controller for Windows using your webcam and MediaPipe. Control mouse navigation, click, zoom, close tabs, toggle the Start menu, and switch windows using intuitive hand gestures.



\## Prerequisites

\- Windows 10 / 11 (64-bit)

\- Python 3.10 – 3.12 (Check "Add Python to PATH" during installation)

\- A working webcam



\## Installation \& Setup



1\. \*\*Clone or Download this repository:\*\*

&#x20;  ```bash

&#x20;  git clone https://github.com/ALstar-master-coder/air-navigator.git

&#x20;  cd air-navigator.git


2.(Optional but recommended) Create a Virtual Environment:



bash



python -m venv venv

.\\venv\\Scripts\\activate

Install Dependencies:



3.bash



pip install -r requirements.txt

Run the Program:



4.bash



python air\_scroll.py

(On first run, the hand tracking AI model will download automatically).



Gestures Guide

Gesture	Hand Posture	Action

Point \& Navigate	Index finger pointing, middle \& ring curled	Moves mouse cursor smoothly across screen

Pinky Flash Click	While pointing, flick/flash your pinky finger	Left Click (zero cursor drift)

Close Tab	Hold JUST middle finger up for 0.55s	Closes focused tab (Ctrl + W)

Start Menu	Flash open \& curl 5 fingers swiftly 2×	Toggles Windows Start Menu (Win Key)

Alt + Tab	Hold open palm for \~250ms	Alt+Tab overlay (wipe L/R, fist to select)

Zoom	Thumb + Index pinch / spread	Zoom In / Out (Ctrl + Wheel)

Scroll	2 fingers up (Peace sign)	Scroll window up / down

Controls in Preview Window

\+ / -: Increase / decrease gesture sensitivity

i: Invert scroll direction

p: Pause / resume gesture tracking

q or Esc: Quit

