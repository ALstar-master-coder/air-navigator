"""
Webcam Hand Gesture Controller:

Gestures:
  1. Navigate & Click:
     - 1 finger pointing (Index finger extended): steers the mouse cursor smoothly.
     - Flash Click: while pointing, quickly flash your index finger (curl and snap open)
       to click. Cursor stays 100% frozen on target during the flash for perfect accuracy.
  2. Tab Switching (Alt + Tab):
     - Open palm (all 5 fingers spread): activates and HOLDS the Windows Alt+Tab switcher.
     - Wipe hand right: moves window selection to the right.
     - Wipe hand left: moves window selection to the left.
     - Curl all fingers down (fist): confirms and switches directly to that window!
  3. Zoom:
     - Thumb & Index finger open (Middle, Ring, Pinky folded):
       Small gradual spread -> Zoom In (Ctrl + Wheel Up)
       Small gradual pinch -> Zoom Out (Ctrl + Wheel Down)
       (Fast sudden jumps are filtered out for smooth, controlled zooming)
  4. Scroll:
     - Two fingers (peace sign: Index + Middle), move up/down -> scroll window.

Keys (preview window):
  q / Esc  quit
  i        invert scroll direction
  + / -    change sensitivity
  p        pause / resume gestures
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
import urllib.request
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

PREVIEW_TITLE = "Air Scroll & Gesture Control"
WHEEL_DELTA = 120
WM_MOUSEWHEEL = 0x020A

# Landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP, THUMB_CMC = 4, 3, 2, 1
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]

user32 = ctypes.windll.user32

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
GA_ROOT = 2
GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008

SM_CXSCREEN = 0
SM_CYSCREEN = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

VK_TAB = 0x09
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_MENU = 0x12  # Alt
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_LWIN = 0x5B  # Windows key (Start Menu)
VK_W = 0x57     # W key (for Ctrl+W close tab)

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", INPUTUNION)]


user32.WindowFromPoint.argtypes = [POINT]
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.GetAncestor.restype = wintypes.HWND
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _preview_hwnd() -> int:
    return user32.FindWindowW(None, PREVIEW_TITLE)


def pin_preview_window() -> None:
    """Keep the camera preview visible without stealing focus from active apps."""
    hwnd = _preview_hwnd()
    if not hwnd:
        return
    user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
    )
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TOPMOST | WS_EX_NOACTIVATE)


def _send_wheel_input(delta: int) -> None:
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(0, 0, delta & 0xFFFFFFFF, MOUSEEVENTF_WHEEL, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _key_input(vk: int, up: bool = False) -> INPUT:
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    return inp


def send_key(vk: int, up: bool = False) -> None:
    inp = _key_input(vk, up)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def send_tap(vk: int) -> None:
    down = _key_input(vk, False)
    up = _key_input(vk, True)
    events = (INPUT * 2)(down, up)
    user32.SendInput(2, events, ctypes.sizeof(INPUT))


def alt_tab_open() -> None:
    """Press Alt down and tap Tab once to show the Windows Alt+Tab switcher."""
    events = [
        _key_input(VK_MENU, False),
        _key_input(VK_TAB, False),
        _key_input(VK_TAB, True),
    ]
    arr = (INPUT * len(events))(*events)
    user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))


def alt_tab_step(direction: int) -> None:
    """While Alt is held, step right (+1) or left (-1) across open windows."""
    if direction > 0:
        send_tap(VK_RIGHT)
    elif direction < 0:
        send_tap(VK_LEFT)


def alt_tab_release() -> None:
    """Release Alt to activate and focus the currently selected window."""
    send_key(VK_MENU, True)


def send_click_input(x: int | None = None, y: int | None = None) -> None:
    """Send left mouse button down and up at specified coordinates with zero drift."""
    if x is not None and y is not None:
        user32.SetCursorPos(x, y)
    down = INPUT(type=INPUT_MOUSE)
    down.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)
    up = INPUT(type=INPUT_MOUSE)
    up.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)
    events = (INPUT * 2)(down, up)
    user32.SendInput(2, events, ctypes.sizeof(INPUT))


def send_win_key() -> None:
    """Send Windows key tap to open/toggle the Start Menu."""
    send_tap(VK_LWIN)


def close_focused_tab() -> None:
    """Send Ctrl + W to close the currently focused tab."""
    events = [
        _key_input(VK_CONTROL, False),
        _key_input(VK_W, False),
        _key_input(VK_W, True),
        _key_input(VK_CONTROL, True),
    ]
    arr = (INPUT * len(events))(*events)
    user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))


def send_zoom_input(notches: int) -> None:
    """Send Ctrl + MouseWheel to zoom in (+) or out (-)."""
    if notches == 0:
        return
    delta = int(notches) * WHEEL_DELTA
    wheel = INPUT(type=INPUT_MOUSE)
    wheel.mi = MOUSEINPUT(0, 0, delta & 0xFFFFFFFF, MOUSEEVENTF_WHEEL, 0, 0)

    events = [
        _key_input(VK_CONTROL, False),
        wheel,
        _key_input(VK_CONTROL, True),
    ]
    arr = (INPUT * len(events))(*events)
    user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))


class ForegroundScroller:
    """Scroll the last focused app, or the window under the mouse if it isn't this preview."""

    def __init__(self) -> None:
        self._target = 0

    def remember_foreground(self) -> None:
        hwnd = user32.GetForegroundWindow()
        preview = _preview_hwnd()
        if hwnd and hwnd != preview and _window_title(hwnd) != PREVIEW_TITLE:
            self._target = hwnd

    def _hit_window(self, top: int) -> int:
        rect = RECT()
        user32.GetWindowRect(top, ctypes.byref(rect))
        cx = (rect.left + rect.right) // 2
        cy = (rect.top + rect.bottom) // 2
        hit = user32.WindowFromPoint(POINT(cx, cy))
        if hit and user32.GetAncestor(hit, GA_ROOT) == top:
            return hit
        return top

    def _cursor_on_preview(self) -> bool:
        preview = _preview_hwnd()
        if not preview:
            return False
        pt = POINT()
        if not user32.GetCursorPos(ctypes.byref(pt)):
            return False
        hit = user32.WindowFromPoint(pt)
        return bool(hit) and user32.GetAncestor(hit, GA_ROOT) == preview

    def scroll(self, notches: int) -> None:
        if notches == 0:
            return
        self.remember_foreground()
        delta = int(notches) * WHEEL_DELTA

        if not self._cursor_on_preview():
            _send_wheel_input(delta)
            return

        hwnd = self._target
        if not hwnd or not user32.IsWindow(hwnd):
            return
        rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        cx = (rect.left + rect.right) // 2
        cy = (rect.top + rect.bottom) // 2
        wparam = (delta & 0xFFFF) << 16
        lparam = ((cy & 0xFFFF) << 16) | (cx & 0xFFFF)
        user32.SendMessageW(self._hit_window(hwnd), WM_MOUSEWHEEL, wparam, lparam)


def ensure_hand_model() -> str:
    model_path = Path(__file__).resolve().parent / MODEL_NAME
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return str(model_path)
    print("Downloading hand landmark model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Saved {model_path}")
    return str(model_path)


def draw_hand(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 180, 70), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (40, 220, 255), -1)


def _finger_up(landmarks, tip: int, pip: int, min_delta: float = 0.02) -> bool:
    """True when finger tip is higher (smaller y) than its PIP knuckle."""
    return (landmarks[pip].y - landmarks[tip].y) > min_delta


def _finger_folded(landmarks, tip: int, pip: int, mcp: int) -> bool:
    """True when finger is curled toward palm."""
    wrist = landmarks[WRIST]
    dist_tip_sq = (landmarks[tip].x - wrist.x) ** 2 + (landmarks[tip].y - wrist.y) ** 2
    dist_pip_sq = (landmarks[pip].x - wrist.x) ** 2 + (landmarks[pip].y - wrist.y) ** 2
    is_shorter = dist_tip_sq < dist_pip_sq * 1.15
    is_down = landmarks[tip].y > landmarks[pip].y - 0.015
    return is_shorter or is_down


def _hand_scale(landmarks) -> float:
    return math.hypot(landmarks[INDEX_MCP].x - landmarks[WRIST].x, landmarks[INDEX_MCP].y - landmarks[WRIST].y) or 0.1


def is_thumb_open_wide(landmarks) -> bool:
    """
    True when thumb is deliberately opened away from the middle knuckle and palm.
    Cleanly separates pointing mode (thumb resting/tucked against middle finger)
    from zoom/pinch mode (thumb active).
    """
    tip = landmarks[THUMB_TIP]
    ip = landmarks[THUMB_IP]
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]

    scale = _hand_scale(landmarks)
    dist_tip_middle_mcp = math.hypot(tip.x - middle_mcp.x, tip.y - middle_mcp.y) / scale
    dist_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    dist_ip_wrist = math.hypot(ip.x - wrist.x, ip.y - wrist.y)

    return (dist_tip_wrist > dist_ip_wrist + 0.012) and (dist_tip_middle_mcp > 0.65)


def is_open_palm(landmarks) -> bool:
    """All 5 fingers extended and spread."""
    fingers = (
        _finger_up(landmarks, INDEX_TIP, INDEX_PIP, min_delta=0.015)
        and _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP, min_delta=0.015)
        and _finger_up(landmarks, RING_TIP, RING_PIP, min_delta=0.015)
        and _finger_up(landmarks, PINKY_TIP, PINKY_PIP, min_delta=0.015)
        and is_thumb_open_wide(landmarks)
    )
    spread = abs(landmarks[INDEX_TIP].x - landmarks[PINKY_TIP].x) > 0.07
    return fingers and spread


def is_fist_or_curled(landmarks) -> bool:
    """True when all 4 fingers (index, middle, ring, pinky) are curled down into a fist."""
    return (
        _finger_folded(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        and _finger_folded(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        and _finger_folded(landmarks, RING_TIP, RING_PIP, RING_MCP)
        and _finger_folded(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    )


def is_two_finger_pose(landmarks) -> bool:
    index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP)
    middle_up = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_down = _finger_folded(landmarks, RING_TIP, RING_PIP, RING_MCP)
    pinky_down = _finger_folded(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    return index_up and middle_up and ring_down and pinky_down


def is_five_fingers_open(landmarks) -> bool:
    """All 4 main fingers extended (for fast flash-and-curl detection)."""
    return (
        _finger_up(landmarks, INDEX_TIP, INDEX_PIP, min_delta=0.012)
        and _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP, min_delta=0.012)
        and _finger_up(landmarks, RING_TIP, RING_PIP, min_delta=0.012)
        and _finger_up(landmarks, PINKY_TIP, PINKY_PIP, min_delta=0.012)
    )


def is_pinky_flashed(landmarks, scale: float) -> bool:
    """
    True when pinky is extended / flashed upward or outward away from the folded palm.
    """
    tip = landmarks[PINKY_TIP]
    pip = landmarks[PINKY_PIP]
    # 1. Pinky extended upward (tip above PIP)
    is_up = (pip.y - tip.y) > 0.015
    # 2. Pinky extended outward (spread wide from ring finger)
    ring_tip = landmarks[RING_TIP]
    dist_outward = abs(tip.x - ring_tip.x) / max(0.01, scale)
    is_out = (dist_outward > 0.45) and (tip.y < pip.y + 0.02)
    return is_up or is_out


def is_pointing_hand(landmarks) -> bool:
    """
    True when hand is in pointing / pinky-click posture:
    Index is extended pointing, Middle and Ring are folded down,
    and Thumb is not spread wide. Pinky is free (folded for navigating, flashed for clicking).
    """
    index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP, min_delta=0.015) or (
        math.hypot(landmarks[INDEX_TIP].x - landmarks[WRIST].x, landmarks[INDEX_TIP].y - landmarks[WRIST].y)
        > math.hypot(landmarks[INDEX_PIP].x - landmarks[WRIST].x, landmarks[INDEX_PIP].y - landmarks[WRIST].y) * 1.12
    )
    middle_down = _finger_folded(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_down = _finger_folded(landmarks, RING_TIP, RING_PIP, RING_MCP)
    thumb_not_spread = not is_thumb_open_wide(landmarks)
    return index_up and middle_down and ring_down and thumb_not_spread


def is_middle_finger_pose(landmarks) -> bool:
    """
    True when JUST the middle finger is extended up, and all other fingers are folded.
    Holding this pose closes the focused tab (Ctrl + W).
    """
    middle_up = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP, min_delta=0.018) or (
        math.hypot(landmarks[MIDDLE_TIP].x - landmarks[WRIST].x, landmarks[MIDDLE_TIP].y - landmarks[WRIST].y)
        > math.hypot(landmarks[MIDDLE_PIP].x - landmarks[WRIST].x, landmarks[MIDDLE_PIP].y - landmarks[WRIST].y) * 1.15
    )
    index_down = _finger_folded(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    ring_down = _finger_folded(landmarks, RING_TIP, RING_PIP, RING_MCP)
    pinky_down = _finger_folded(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    thumb_not_spread = not is_thumb_open_wide(landmarks)
    return middle_up and index_down and ring_down and pinky_down and thumb_not_spread


def is_pointing_pose(landmarks) -> bool:
    return is_pointing_hand(landmarks)


def is_zoom_pose(landmarks) -> bool:
    """Index extended + Thumb open wide + other 3 fingers folded."""
    index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP, min_delta=0.02) or (
        math.hypot(landmarks[INDEX_TIP].x - landmarks[WRIST].x, landmarks[INDEX_TIP].y - landmarks[WRIST].y)
        > math.hypot(landmarks[INDEX_PIP].x - landmarks[WRIST].x, landmarks[INDEX_PIP].y - landmarks[WRIST].y) * 1.15
    )
    thumb_open = is_thumb_open_wide(landmarks)
    middle_down = _finger_folded(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_down = _finger_folded(landmarks, RING_TIP, RING_PIP, RING_MCP)
    pinky_down = _finger_folded(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    return index_up and thumb_open and middle_down and ring_down and pinky_down


def tracking_point(landmarks) -> tuple[float, float]:
    """Midpoint of index/middle knuckles for scrolling."""
    ix = (landmarks[INDEX_MCP].x + landmarks[MIDDLE_MCP].x) / 2.0
    iy = (landmarks[INDEX_MCP].y + landmarks[MIDDLE_MCP].y) / 2.0
    return ix, iy


def palm_point(landmarks) -> tuple[float, float]:
    xs = [landmarks[i].x for i in (0, INDEX_MCP, MIDDLE_MCP, 13, 17)]
    ys = [landmarks[i].y for i in (0, INDEX_MCP, MIDDLE_MCP, 13, 17)]
    return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass
class ScrollState:
    sensitivity: float = 1.0
    invert: bool = False
    paused: bool = False
    smooth_y: float | None = None
    accumulator: float = 0.0
    pixels_per_notch: float = 22.0
    ema_alpha: float = 0.35
    max_notches_per_frame: int = 8

    def reset_motion(self) -> None:
        self.smooth_y = None
        self.accumulator = 0.0

    def update(self, y_norm: float, frame_h: int) -> int:
        if self.paused:
            self.reset_motion()
            return 0

        if self.smooth_y is None:
            self.smooth_y = y_norm
            return 0

        prev = self.smooth_y
        self.smooth_y = self.ema_alpha * y_norm + (1.0 - self.ema_alpha) * prev
        dy_px = (self.smooth_y - prev) * frame_h

        if abs(dy_px) < 0.35:
            return 0

        direction = -1 if self.invert else 1
        self.accumulator += -dy_px * direction * self.sensitivity

        notches = 0
        step = self.pixels_per_notch
        while self.accumulator >= step:
            notches += 1
            self.accumulator -= step
        while self.accumulator <= -step:
            notches -= 1
            self.accumulator += step

        if notches > self.max_notches_per_frame:
            notches = self.max_notches_per_frame
        elif notches < -self.max_notches_per_frame:
            notches = -self.max_notches_per_frame
        return notches


@dataclass
class SwitchState:
    is_active: bool = False
    smooth_x: float | None = None
    accumulator: float = 0.0
    ema_alpha: float = 0.45
    pixels_per_step: float = 45.0   # Horizontal displacement required to move one tab
    cooldown_s: float = 0.25
    last_step_at: float = 0.0
    last_direction: int = 0
    feedback_until: float = 0.0

    def start(self) -> None:
        """Called when 5 fingers are opened. Opens and holds the Alt+Tab window."""
        if not self.is_active:
            self.is_active = True
            self.smooth_x = None
            self.accumulator = 0.0
            self.last_step_at = time.perf_counter()
            self.last_direction = 0
            self.feedback_until = time.perf_counter() + 0.60
            alt_tab_open()

    def update_motion(self, x_norm: float, frame_w: int) -> int:
        """
        Tracks palm horizontal displacement while Alt is held.
        Returns +1 (wiped right), -1 (wiped left), 0 (holding).
        """
        if not self.is_active:
            return 0

        if self.smooth_x is None:
            self.smooth_x = x_norm
            return 0

        prev = self.smooth_x
        self.smooth_x = self.ema_alpha * x_norm + (1.0 - self.ema_alpha) * prev
        dx_px = (self.smooth_x - prev) * frame_w

        if abs(dx_px) < 0.6:
            return 0

        self.accumulator += dx_px
        now = time.perf_counter()
        if now - self.last_step_at < self.cooldown_s:
            return 0

        if self.accumulator >= self.pixels_per_step:
            self.accumulator = 0.0
            self.last_step_at = now
            self.last_direction = 1
            self.feedback_until = now + 0.40
            alt_tab_step(1)
            return 1
        elif self.accumulator <= -self.pixels_per_step:
            self.accumulator = 0.0
            self.last_step_at = now
            self.last_direction = -1
            self.feedback_until = now + 0.40
            alt_tab_step(-1)
            return -1
        return 0

    def release(self) -> None:
        """Called when fingers curl down into a fist. Confirms selection and closes switcher."""
        if self.is_active:
            alt_tab_release()
            self.is_active = False
            self.smooth_x = None
            self.accumulator = 0.0
            self.feedback_until = 0.0


@dataclass
class CursorState:
    screen_w: int = 1920
    screen_h: int = 1080
    # Safe margins for 15.6" HD (1366x768) and standard webcams so all 4 corners
    # are reached comfortably without the hand leaving camera view.
    margin_x: float = 0.07
    margin_y: float = 0.08
    curr_x: float = 0.0
    curr_y: float = 0.0
    initialized: bool = False
    frozen_until: float = 0.0
    min_alpha: float = 0.18
    max_alpha: float = 0.75

    def __post_init__(self) -> None:
        self.refresh_screen_size()

    def refresh_screen_size(self) -> None:
        w = user32.GetSystemMetrics(SM_CXSCREEN)
        h = user32.GetSystemMetrics(SM_CYSCREEN)
        if w > 0 and h > 0:
            self.screen_w = w
            self.screen_h = h

    def freeze(self, duration_s: float = 0.22) -> None:
        """Freeze cursor coordinates so finger motion during clicks doesn't shift the pointer."""
        self.frozen_until = max(self.frozen_until, time.perf_counter() + duration_s)

    def is_frozen(self) -> bool:
        return time.perf_counter() < self.frozen_until

    def reset(self) -> None:
        self.initialized = False

    def update(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        now = time.perf_counter()
        if now < self.frozen_until and self.initialized:
            return int(round(self.curr_x)), int(round(self.curr_y))

        # Strictly linear, continuous, monotonic coordinate mapping — NO JUMPS or missing sections!
        active_w = max(0.01, 1.0 - 2.0 * self.margin_x)
        active_h = max(0.01, 1.0 - 2.0 * self.margin_y)
        nx = max(0.0, min(1.0, (norm_x - self.margin_x) / active_w))
        ny = max(0.0, min(1.0, (norm_y - self.margin_y) / active_h))

        target_x = nx * (self.screen_w - 1)
        target_y = ny * (self.screen_h - 1)

        if not self.initialized:
            self.curr_x = target_x
            self.curr_y = target_y
            self.initialized = True
            user32.SetCursorPos(int(round(target_x)), int(round(target_y)))
            return int(round(target_x)), int(round(target_y))

        dx = target_x - self.curr_x
        dy = target_y - self.curr_y
        dist = math.hypot(dx, dy)

        # Smooth velocity-adaptive EMA:
        # Slow / resting: low alpha (0.18) absorbs micro-tremors and hand jitter.
        # Medium / fast sweeps: alpha smoothly ramps up to 0.75 for instant response.
        t = max(0.0, min(1.0, dist / 35.0))
        alpha = self.min_alpha + t * (self.max_alpha - self.min_alpha)

        self.curr_x += alpha * dx
        self.curr_y += alpha * dy
        ix, iy = int(round(self.curr_x)), int(round(self.curr_y))
        user32.SetCursorPos(ix, iy)
        return ix, iy


@dataclass
class PinkyClickDetector:
    """
    Overhauled click detector:
    Clicks upon flashing / extending the pinky finger while on pointing cursor mode.
    Index finger continues aiming steadily at the target with ZERO drift.
    """
    pinky_was_flashed: bool = False
    last_click_at: float = 0.0
    cooldown_s: float = 0.35
    feedback_until: float = 0.0

    def reset(self) -> None:
        self.pinky_was_flashed = False

    def update(self, landmarks, hand_scale: float) -> tuple[bool, bool, bool]:
        """
        Returns (clicked, is_flash_in_progress, should_freeze).
        - clicked: True on the exact frame the pinky is flashed / extended.
        - is_flash_in_progress: True while pinky remains extended.
        - should_freeze: True while pinky is extended to freeze cursor rock-solid on target.
        """
        now = time.perf_counter()
        flashed = is_pinky_flashed(landmarks, hand_scale)

        if flashed:
            if not self.pinky_was_flashed and (now - self.last_click_at > self.cooldown_s):
                self.pinky_was_flashed = True
                self.last_click_at = now
                self.feedback_until = now + 0.38
                return True, True, True
            return False, True, True
        else:
            self.pinky_was_flashed = False
            return False, False, False


# Alias for backward compatibility
FlashClickDetector = PinkyClickDetector


@dataclass
class CloseTabDetector:
    """
    Detects holding just the middle finger up to close the currently focused tab (Ctrl + W).
    Requires holding for hold_duration_s (~0.55s) to avoid accidental triggers.
    """
    hold_start_at: float = 0.0
    is_holding: bool = False
    hold_duration_s: float = 0.55
    last_close_at: float = 0.0
    cooldown_s: float = 1.00
    feedback_until: float = 0.0
    progress: float = 0.0

    def reset(self) -> None:
        self.hold_start_at = 0.0
        self.is_holding = False
        self.progress = 0.0

    def update(self, landmarks) -> tuple[bool, bool, float]:
        """
        Returns (closed_tab, is_holding, progress_0_to_1).
        """
        now = time.perf_counter()
        if now - self.last_close_at < self.cooldown_s:
            self.reset()
            return False, False, 0.0

        if not is_middle_finger_pose(landmarks):
            self.reset()
            return False, False, 0.0

        if not self.is_holding:
            self.is_holding = True
            self.hold_start_at = now
            self.progress = 0.0
            return False, True, 0.0

        elapsed = now - self.hold_start_at
        self.progress = min(1.0, elapsed / self.hold_duration_s)

        if elapsed >= self.hold_duration_s:
            self.reset()
            self.last_close_at = now
            self.feedback_until = now + 0.85
            close_focused_tab()
            return True, False, 1.0

        return False, True, self.progress


@dataclass
class StartMenuDetector:
    """
    Detects when all five fingers are flashed and curled swiftly 2 times.
    Cycle 1: Open 5 fingers -> Curl all fingers
    Cycle 2: Open 5 fingers -> Curl all fingers
    Upon the 2nd swift curl, triggers the Windows key (VK_LWIN) to toggle the Start Menu.
    """
    stage: int = 0  # 0: idle, 1: flash1, 2: curl1, 3: flash2
    stage_start_time: float = 0.0
    sequence_start_time: float = 0.0
    last_trigger_at: float = 0.0
    feedback_until: float = 0.0
    cooldown_s: float = 0.8
    step_timeout_s: float = 0.50     # Max time between consecutive steps (swift gesture)
    total_timeout_s: float = 1.35    # Max total time to complete both cycles

    def reset(self) -> None:
        self.stage = 0
        self.stage_start_time = 0.0
        self.sequence_start_time = 0.0

    def update(self, landmarks) -> bool:
        now = time.perf_counter()
        if now - self.last_trigger_at < self.cooldown_s:
            return False

        is_open = is_five_fingers_open(landmarks)
        is_curled = is_fist_or_curled(landmarks)

        # Timeout checks
        if self.stage > 0:
            if (now - self.sequence_start_time > self.total_timeout_s) or (
                now - self.stage_start_time > self.step_timeout_s
            ):
                self.reset()

        if self.stage == 0:
            if is_open:
                self.stage = 1
                self.sequence_start_time = now
                self.stage_start_time = now
        elif self.stage == 1:
            # Looking for curl 1
            if is_curled:
                self.stage = 2
                self.stage_start_time = now
        elif self.stage == 2:
            # Looking for flash 2
            if is_open:
                self.stage = 3
                self.stage_start_time = now
        elif self.stage == 3:
            # Looking for curl 2 -> Trigger!
            if is_curled:
                self.reset()
                self.last_trigger_at = now
                self.feedback_until = now + 0.85
                send_win_key()
                return True

        return False


@dataclass
class ZoomState:
    """
    Smooth, highly responsive pinch-to-zoom:
    - Senses small, gradual finger distance adjustments immediately.
    - Filters out sudden, fast jumps to prevent wild zooming.
    """
    smooth_dist: float | None = None
    accumulator: float = 0.0
    step: float = 0.022        # Small step size = highly responsive to small movements
    max_delta: float = 0.050   # Filters out fast sudden jumps/glitches
    min_delta: float = 0.003   # Deadzone for micro-tremors
    cooldown_s: float = 0.05   # Fast rate for smooth continuous adjustments
    last_zoom_at: float = 0.0
    last_action: str = ""
    feedback_until: float = 0.0

    def reset(self) -> None:
        self.smooth_dist = None
        self.accumulator = 0.0

    def update(self, thumb_tip, index_tip, hand_scale: float) -> int:
        scale = max(0.01, hand_scale)
        dist_2d = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        norm_dist = dist_2d / scale

        now = time.perf_counter()

        if self.smooth_dist is None:
            self.smooth_dist = norm_dist
            return 0

        prev = self.smooth_dist
        self.smooth_dist = 0.60 * norm_dist + 0.40 * prev
        delta = self.smooth_dist - prev

        # Filter out sudden, fast jumps (only small deliberate changes zoom)
        if abs(delta) > self.max_delta:
            self.smooth_dist = norm_dist
            self.accumulator = 0.0
            return 0

        # Deadzone: if hand is stationary, gently decay residual accumulator
        if abs(delta) < self.min_delta:
            self.accumulator *= 0.5
            return 0

        self.accumulator += delta
        if now - self.last_zoom_at < self.cooldown_s:
            return 0

        notches = 0
        if self.accumulator >= self.step:
            notches = 1
            self.accumulator -= self.step
            self.last_zoom_at = now
            self.last_action = "IN"
            self.feedback_until = now + 0.25
        elif self.accumulator <= -self.step:
            notches = -1
            self.accumulator += self.step
            self.last_zoom_at = now
            self.last_action = "OUT"
            self.feedback_until = now + 0.25

        return notches


def draw_hud(
    frame,
    mode: str,
    scroll_state: ScrollState,
    switch_state: SwitchState,
    click_detector: PinkyClickDetector,
    zoom_state: ZoomState,
    target_name: str,
    switch_confirmed_until: float = 0.0,
    start_menu_detector: StartMenuDetector | None = None,
    close_tab_detector: CloseTabDetector | None = None,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 138), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    now = time.perf_counter()

    # Status header
    if scroll_state.paused:
        status, color = "PAUSED (press 'p' to resume)", (80, 80, 220)
    elif close_tab_detector and now < close_tab_detector.feedback_until:
        status, color = "CLOSED TAB! (Ctrl + W)  [Focused Tab Closed]", (0, 100, 255)
    elif close_tab_detector and close_tab_detector.is_holding:
        ms_left = int((1.0 - close_tab_detector.progress) * close_tab_detector.hold_duration_s * 1000)
        pct = int(close_tab_detector.progress * 100)
        status, color = f"HOLDING MIDDLE FINGER: Closing Tab in {ms_left}ms ({pct}%)", (0, 165, 255)
    elif start_menu_detector and now < start_menu_detector.feedback_until:
        status, color = "START MENU (WIN KEY)!  [Start Menu Toggled]", (0, 255, 255)
    elif start_menu_detector and start_menu_detector.stage > 0:
        stage_names = {1: "Flash 1 Detected", 2: "Curl 1 Confirmed", 3: "Flash 2 Detected (Curl to Open Start!)"}
        status, color = f"START MENU GESTURE: {stage_names.get(start_menu_detector.stage, '')}", (0, 230, 255)
    elif now < switch_confirmed_until:
        status, color = "TAB SWITCH CONFIRMED!  [Window Focused]", (80, 240, 90)
    elif switch_state.is_active:
        if now < switch_state.feedback_until and switch_state.last_direction != 0:
            arrow = "SWITCH RIGHT >>" if switch_state.last_direction > 0 else "<< SWITCH LEFT"
            status, color = f"ALT+TAB HELD: {arrow}", (0, 215, 255)
        else:
            status, color = "ALT+TAB HELD: Wipe Left/Right to Browse | Curl to Confirm", (0, 190, 255)
    elif mode == "click" or now < click_detector.feedback_until:
        status, color = "PINKY CLICK!  [Click Registered At Target]", (0, 255, 255)
    elif mode == "pinky_armed":
        status, color = "PINKY FLASH (CLICKING)  [Cursor Frozen]", (0, 230, 255)
    elif mode == "pointing":
        status, color = "1 FINGER: Navigating Mouse (Flash Pinky to Click)", (255, 220, 40)
    elif mode == "zoom":
        action = zoom_state.last_action if now < zoom_state.feedback_until else ""
        action_str = f" - ZOOM {action}" if action else ""
        status, color = f"THUMB + INDEX: Pinch / Spread to Zoom{action_str}", (230, 80, 255)
    elif mode == "scroll":
        status, color = "TWO FINGERS: Peace Sign - Scrolling", (80, 240, 90)
    else:
        status, color = "GESTURE READY - Point / Pinky Click / Close Tab / Zoom / Scroll", (180, 180, 190)

    cv2.putText(frame, status, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    # Window target / Alt+Tab guide
    if switch_state.is_active:
        display_target = "Alt+Tab Active! Wipe L/R to browse. Curl all fingers down into a fist to select."
    else:
        display_target = f"Target Window: {target_name[:50] or '(focused window)'}"
    cv2.putText(
        frame,
        display_target,
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )

    # Key settings & sensitivity
    inv = "ON" if scroll_state.invert else "OFF"
    cv2.putText(
        frame,
        f"Sensitivity: {scroll_state.sensitivity:.1f}   Invert: {inv}    Keys: [+/-] Sens   [i] Invert   [p] Pause   [q] Quit",
        (24, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 200, 200),
        1,
    )

    # Quick gestures guide
    cv2.putText(
        frame,
        "1 Finger + Pinky Flash = Click | Hold Middle = Close Tab | 5-Fingers Hold = Alt+Tab | Peace = Scroll",
        (24, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 210, 160),
        1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Air mouse, zoom, Alt+Tab, and scroll gesture controller."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--invert", action="store_true", help="Swap up/down scroll direction")
    parser.add_argument("--margin-x", type=float, default=0.07, help="Horizontal camera margin for cursor reach")
    parser.add_argument("--margin-y", type=float, default=0.08, help="Vertical camera margin for cursor reach")
    return parser.parse_args()


def main() -> int:
    if sys.platform != "win32":
        print("This program uses Windows APIs. Run it on Windows.")
        return 1

    args = parse_args()
    scroll_state = ScrollState(sensitivity=max(0.2, args.sensitivity), invert=args.invert)
    switch_state = SwitchState()
    cursor_state = CursorState(margin_x=args.margin_x, margin_y=args.margin_y)
    click_detector = PinkyClickDetector()
    close_tab_detector = CloseTabDetector()
    start_menu_detector = StartMenuDetector()
    zoom_state = ZoomState()
    scroller = ForegroundScroller()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Could not open webcam. Close other apps using camera and retry.")
        return 1

    try:
        model_path = ensure_hand_model()
    except OSError as exc:
        print("Could not download the hand model. Check your internet connection.")
        print(exc)
        cap.release()
        return 1

    options = HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Air Gesture Controller active.")
    print("Gestures:")
    print("  1. Index Finger Pointing: Smooth mouse navigation (no jumps, full screen).")
    print("  2. Pinky Flash Click: Flash/extend pinky finger while pointing to click (zero cursor drift).")
    print("  3. Middle Finger Hold: Hold just middle finger up to close focused tab (Ctrl + W).")
    print("  4. 5-Finger Swift Flash x2: Flash open & curl five fingers swiftly 2 times -> Start Menu (Win Key).")
    print("  5. 5-Finger Open Palm (Hold): Activates Alt+Tab switcher. Wipe Right/Left to browse.")
    print("     Curl all fingers down (fist) to confirm and switch into that window.")
    print("  6. Thumb + Index: Small gradual spread to Zoom In, pinch to Zoom Out.")
    print("  7. Peace Sign (2 Fingers): Scroll up / down.")
    print("Preview window is pinned topmost. Press 'q' or Esc to quit.")

    cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_TITLE, args.width, args.height)

    lost_frames = 0
    palm_lost_count = 0
    palm_open_frames = 0
    preview_pinned = False
    timestamp_ms = 0
    switch_confirmed_until = 0.0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                scroller.remember_foreground()
                ok, frame = cap.read()
                if not ok:
                    print("Lost the camera feed.")
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                current_mode = "idle"

                if result.hand_landmarks:
                    lost_frames = 0
                    palm_lost_count = 0
                    hand = result.hand_landmarks[0]
                    draw_hand(frame, hand)
                    scale = _hand_scale(hand)

                    # --- 1. Start Menu Double Swift Flash & Curl (5 Fingers) ---
                    start_menu_triggered = False
                    if not switch_state.is_active:
                        if start_menu_detector.update(hand):
                            current_mode = "start_menu"
                            start_menu_triggered = True
                            palm_open_frames = 0
                            close_tab_detector.reset()

                    # --- 2. Alt + Tab Mode Management ---
                    if switch_state.is_active:
                        start_menu_detector.reset()
                        close_tab_detector.reset()
                        current_mode = "switch"
                        # When user curls all fingers down into a fist -> CONFIRM switched tab!
                        if is_fist_or_curled(hand):
                            switch_state.release()
                            switch_confirmed_until = time.perf_counter() + 0.65
                            palm_open_frames = 0
                        else:
                            px, py = palm_point(hand)
                            switch_state.update_motion(px, w)

                            cx, cy = int(px * w), int(py * h)
                            if switch_state.last_direction > 0:
                                cv2.arrowedLine(frame, (cx - 40, cy), (cx + 50, cy), (0, 215, 255), 4, tipLength=0.35)
                            elif switch_state.last_direction < 0:
                                cv2.arrowedLine(frame, (cx + 40, cy), (cx - 50, cy), (0, 215, 255), 4, tipLength=0.35)
                            else:
                                cv2.circle(frame, (cx, cy), 16, (0, 200, 255), 2)
                            cv2.putText(
                                frame, "ALT+TAB HELD", (cx - 50, cy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2
                            )

                    elif not start_menu_triggered and is_open_palm(hand):
                        close_tab_detector.reset()
                        palm_open_frames += 1
                        # Hold palm open for ~8 frames (~250ms) to activate Alt+Tab.
                        # This ensures swift 2x flashes (< 200ms) trigger Start Menu without Alt+Tab popping up!
                        if palm_open_frames >= 8:
                            current_mode = "switch"
                            switch_state.start()

                    # --- 3. Close Focused Tab Gesture (Hold Just Middle Finger) ---
                    elif is_middle_finger_pose(hand):
                        palm_open_frames = 0
                        closed_tab, is_holding, progress = close_tab_detector.update(hand)
                        if closed_tab:
                            current_mode = "close_tab_done"
                        elif is_holding:
                            current_mode = "close_tab_holding"

                        mid_tip = hand[MIDDLE_TIP]
                        mx, my = int(mid_tip.x * w), int(mid_tip.y * h)
                        now = time.perf_counter()
                        if closed_tab or now < close_tab_detector.feedback_until:
                            cv2.circle(frame, (mx, my), 28, (0, 100, 255), 4)
                            cv2.putText(frame, "TAB CLOSED! (Ctrl+W)", (mx - 60, my - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
                        elif is_holding:
                            cv2.circle(frame, (mx, my), 20, (0, 165, 255), 2)
                            cv2.ellipse(frame, (mx, my), (20, 20), 0, -90, int(-90 + progress * 360), (0, 200, 255), 4)
                            ms_left = int((1.0 - progress) * close_tab_detector.hold_duration_s * 1000)
                            cv2.putText(frame, f"CLOSE TAB: {ms_left}ms", (mx + 15, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                    else:
                        close_tab_detector.reset()
                        palm_open_frames = 0
                        if is_zoom_pose(hand):
                            current_mode = "zoom"
                            notches = zoom_state.update(hand[THUMB_TIP], hand[INDEX_TIP], scale)
                            if notches != 0:
                                send_zoom_input(notches)

                            # Draw connecting pinch line
                            p_thumb = (int(hand[THUMB_TIP].x * w), int(hand[THUMB_TIP].y * h))
                            p_index = (int(hand[INDEX_TIP].x * w), int(hand[INDEX_TIP].y * h))
                            line_color = (255, 100, 255) if notches == 0 else ((0, 255, 0) if notches > 0 else (0, 100, 255))
                            cv2.line(frame, p_thumb, p_index, line_color, 3)
                            cv2.circle(frame, p_thumb, 6, (255, 100, 255), -1)
                            cv2.circle(frame, p_index, 6, (255, 100, 255), -1)

                            mid_x = (p_thumb[0] + p_index[0]) // 2
                            mid_y = (p_thumb[1] + p_index[1]) // 2
                            label = f"ZOOM {zoom_state.last_action}" if zoom_state.last_action else "ZOOM"
                            cv2.putText(frame, label, (mid_x + 12, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

                        elif is_two_finger_pose(hand):
                            current_mode = "scroll"
                            _, y = tracking_point(hand)
                            notches = scroll_state.update(y, h)
                            if notches != 0:
                                scroller.scroll(notches)

                            ix, iy = tracking_point(hand)
                            cx, cy = int(ix * w), int(iy * h)
                            cv2.circle(frame, (cx, cy), 10, (80, 240, 90), 2)
                            cv2.putText(
                                frame, "SCROLL", (cx + 14, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 240, 90), 2,
                            )

                        elif is_pointing_hand(hand):
                            tip = hand[INDEX_TIP]
                            tip_px = (int(tip.x * w), int(tip.y * h))

                            # Pinky Flash Click: returns (clicked, is_flash_in_progress, should_freeze)
                            clicked, flash_in_progress, should_freeze = click_detector.update(hand, scale)
                            if clicked:
                                current_mode = "click"
                                # Cursor is 100% frozen on target; click is sent at exact target
                                cursor_state.freeze(0.25)
                                send_click_input(int(round(cursor_state.curr_x)), int(round(cursor_state.curr_y)))
                            elif should_freeze:
                                current_mode = "pinky_armed"
                                cursor_state.freeze(0.22)
                            else:
                                current_mode = "pointing"
                                cursor_state.update(tip.x, tip.y)

                            now = time.perf_counter()
                            if clicked or now < click_detector.feedback_until:
                                cv2.circle(frame, tip_px, 24, (0, 255, 255), 4)
                                cv2.circle(frame, tip_px, 12, (0, 220, 255), -1)
                                cv2.putText(
                                    frame, "PINKY CLICK!", (tip_px[0] + 16, tip_px[1] - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                                )
                            elif flash_in_progress:
                                cv2.circle(frame, tip_px, 16, (0, 230, 255), 3)
                                cv2.putText(
                                    frame, "PINKY FLASH", (tip_px[0] + 14, tip_px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2
                                )
                            else:
                                cv2.circle(frame, tip_px, 8, (255, 220, 40), 2)
                                cv2.line(frame, (tip_px[0] - 12, tip_px[1]), (tip_px[0] + 12, tip_px[1]), (255, 220, 40), 1)
                                cv2.line(frame, (tip_px[0], tip_px[1] - 12), (tip_px[0], tip_px[1] + 12), (255, 220, 40), 1)
                                cv2.putText(
                                    frame, "POINTER", (tip_px[0] + 14, tip_px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 40), 1
                                )
                        else:
                            lost_frames += 1
                else:
                    lost_frames += 1
                    # Prolonged absence safety release (45 frames ~ 1.5s)
                    if switch_state.is_active:
                        palm_lost_count += 1
                        if palm_lost_count >= 45:
                            switch_state.release()
                            palm_lost_count = 0

                if lost_frames > 4:
                    scroll_state.reset_motion()
                    cursor_state.reset()
                    click_detector.reset()
                    close_tab_detector.reset()
                    zoom_state.reset()
                    start_menu_detector.reset()

                draw_hud(
                    frame,
                    current_mode,
                    scroll_state,
                    switch_state,
                    click_detector,
                    zoom_state,
                    _window_title(scroller._target),
                    switch_confirmed_until,
                    start_menu_detector,
                    close_tab_detector,
                )

                cv2.imshow(PREVIEW_TITLE, frame)
                if not preview_pinned:
                    pin_preview_window()
                    preview_pinned = True

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("i"):
                    scroll_state.invert = not scroll_state.invert
                    scroll_state.reset_motion()
                if key == ord("p"):
                    scroll_state.paused = not scroll_state.paused
                    scroll_state.reset_motion()
                if key in (ord("+"), ord("=")):
                    scroll_state.sensitivity = min(4.0, scroll_state.sensitivity + 0.1)
                if key in (ord("-"), ord("_")):
                    scroll_state.sensitivity = max(0.2, scroll_state.sensitivity - 0.1)
    finally:
        # Safety cleanup: ensure Alt is released if user quits while browsing tabs
        switch_state.release()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
