# Corpography - Claude Code Guidelines

## General idea:
Corpography is an interactive multiplayer game where each player stands in front of the camera, one at a time, and attempts to make a geometric or letter shape with their body. The game displays the shape to match on the screen, full-screen so you can see it from 6 to 10 feet away, and starts a timer, configurable, default 10s. The game takes the live video, overlays it with the letter shape to match (scaled) and compares the player position to the shape and displays a score 1 to 100 of how close they are to matching it. The game scales the camera video to adjust for the size of the player vs the size of the shape or letter to try to line it up. After a number of rounds, scores are totaled and highest total wins. It has different alphabets and shapes to choose from. Shapes and letters are assigned "difficuly" numbers that factor into the scoring. 
## Architecture
The game is structured as a Python desktop application with a few clearly separated layers:
Capture → Pose Detection → Scoring → UI/Display

## Core Technology Choices

### Pose Detection
- Use Google MediaPipe Pose

Pose estimation — detecting where a person's body landmarks are (shoulders, elbows, hips, knees, etc.) in real time.

MediaPipe works best when the player is fully visible in frame. You'll want to detect when landmarks are missing or low-confidence (MediaPipe gives you a visibility score per landmark) and display a warning like "Step back — can't see your full body." Building that feedback in early will save a lot of frustration during play.

One Optional Enhancement
MediaPipe also has a selfie segmentation model that can give you a clean player silhouette mask if you want to do something visual — like rendering the player cutout over the target shape for a cool overlay effect. That's purely cosmetic but could make the game look really polished. It runs alongside pose detection with minimal extra cost.

## Video Capture: OpenCV
cv2.VideoCapture() handles the webcam. OpenCV also handles drawing overlays on the live feed.

## UI / Game Shell

Pygame — good for a full-screen game feel, easy timer/score display, handles the "big letter on screen" overlay cleanly. More game-like. Pygame for the game loop and display, with OpenCV feeding frames into it

## The Scoring Algorithm
This is the heart of the game play
1. Extract pose landmarks from the player using MediaPipe — you get a list of (x, y) coordinates for each body joint.
2. Normalize both shapes — scale and center both the player's pose and the target shape so size differences don't matter. A child and a tall adult trying to make an "X" should score the same if their proportions match. Use bounding box normalization: scale both to fit a unit square, then center them.
3. Define target shapes as landmark configurations — for each shape/letter, you store the expected relative positions of key landmarks. Example for a "T":

Arms fully extended horizontally (wrists at same height as shoulders, far apart)
Body vertical (hips below shoulders, aligned)
You don't need all 33 landmarks — maybe 8–10 key ones per shape.

4. Score = similarity metric — after normalizing, compute the mean Euclidean distance between player landmarks and target landmark positions, then map to 0–100. A perfect match = distance 0 = score 100. You'll tune the mapping empirically.
5. Show live feedback during the timer — display the score updating in real time so players can adjust their pose before time runs out. Take the best score achieved during the 10s window, or average the last 2 seconds.

## Difficulty Factor
Encode difficulty per shape and incorporate it into scoring.
These values should be in a human editable config file for easy adjusting during play testing.

Easy: "I" (stand straight), "L" (one arm out), circle outline
Medium: "T", "Y", "A", "X", "star"
Hard: "K", "E", cursive shapes, asymmetric geometric shapes

Store difficulty as a multiplier (1.0 to 2.0). Final round score = raw score × difficulty multiplier.
Display the difficulty badge next to the shape so players know what they're attempting.

## Shape/Letter Representation
Define shapes as normalized landmark templates — a dictionary mapping shape names to expected key-point positions. You build this once by having someone pose correctly and recording their landmarks, or by hand-crafting the expected positions geometrically.
For letter sets (Latin, Greek, maybe Kanji outlines for extreme difficulty), you curate a library of these templates. You could also visually render the target letter as a skeleton overlay on screen so players can see exactly what joint goes where.

## Training Mode
We will need a training mode for the game engine to create these templates. This can be a startup parameter "training-mode".
In "training mode", the user will select a shape from the list (instead of the game choosing it randomly), then the game will display the letter using the same logic as during game play, but instead of a timer and scoring the player can press a button or mouse click to register the key-points as correct.

## Suggested Module Structure
Corpography/
├── main.py                   # Game loop, state machine
├── capture.py                # OpenCV webcam capture
├── pose.py                   # MediaPipe wrapper, landmark extraction
├── scoring.py                # Normalization + similarity scoring
├── editor.py                 # ← editor entry point  
├── core/
│   ├── scoring.py            # ← shared scoring logic
│   ├── pose.py               # ← shared MediaPipe wrapper
│   └── templates.py          # ← shared load/save JSON templates├── shapes/
└── templates/
    └── latin/
        ├── A.json            # Scoring template files for each letter
        ├── B.json            # These are created and edited by the editor.py
        └── ...
├── ui/
│   ├── display.py            # Pygame rendering, overlays
│   └── hud.py                # Score, timer, shape display
├── tests/                    # for all unit tests
├── assets/                   
│   ├── images                # Any image assets used
│   ├── sounds                # for game sound assets
│   └── fonts                 # Custom fonts used for the UI, if any
└── game.py                   # Round management, player rotation, totals

## GPU Usage
MediaPipe will use the GPU automatically for inference. A mid-range GPU (GTX 1060 class or better) handles this easily at 30 FPS.

## Game State Machine
For normal gameplay, not "training-mode"
IDLE → SHOW_SHAPE → COUNTDOWN → CAPTURING (10s) → SCORE_DISPLAY → [next player or GAME_OVER] → LEADERBOARD
Each state is simple. The CAPTURING state is the hot loop: grab frame → run MediaPipe → score → display live score + camera feed + target shape overlay.

## Key Libraries Summary
Webcam capture:  opencv-python
Pose estimation: mediapipe
Math/normalization: numpy
Game UI: pygame-ce

## Claude Code Behavior
- When implementing a major new feature, make a plan and present for approval before editing any files
- Ask any questions needed to resolve ambiguities or conflicts in the plan.

### Testing
- Use pytest
- All logic classes must be instantiatable without a game window
- Do not load image/sound assets in __init__ — lazy load or inject them
- Run tests with: pytest --cov=src

### Dependencies
- Python 3.14
- See Key Libraries
- Do not introduce new dependencies without flagging it

### Build
- Target: self-contained Windows .exe via PyInstaller
- All assets must be loaded using resource paths compatible with PyInstaller bundles
- Use a helper function for asset paths that handles both dev and bundled contexts
- Build both main.py and editor.py as separate exes from the same codebase — Corpography.exe and CorpagraphyEditor.exe — bundled in the same GitHub Release zip. The editor is a dev/content tool but ships with the package so anyone can author new template packs.

## Phase 0: Project Scaffold
"Set up the project structure, virtual environment, requirements.txt, use existing GitHub repo with .gitignore, and a PyInstaller build script. The app should be a Pygame window that says Hello World and exits cleanly. Verify the PyInstaller exe builds and runs."
Deliverables:

Folder structure as designed
requirements.txt with all anticipated dependencies pinned
build.bat or build.ps1 that runs PyInstaller and outputs to dist/
Working Hello World exe

## Phase 1: Camera Feed Display
"Add OpenCV webcam capture feeding live frames into the Pygame window at 30 FPS. Show FPS counter in corner. App should open, show camera, and close cleanly with ESC or window close."
Why here: validates the OpenCV→Pygame pipeline early, which is the trickiest plumbing issue. Better to find frame format/color space conversion problems now.
Deliverables:

Live camera in window
Clean startup/shutdown
FPS visible

## Phase 2: MediaPipe Pose Overlay
"Integrate MediaPipe Pose. Draw the landmark skeleton overlay on the live camera feed. Display per-landmark visibility scores in a debug panel. Highlight landmarks that fall below 0.6 visibility threshold in red."
Why here: this is the core sensor — everything else depends on it. The debug panel will be genuinely useful throughout development.
Deliverables:

Skeleton drawn on live feed
Visibility debug info
show debug panel when starting with '--debug' option
"Body not fully visible" warning when key landmarks are low

## Phase 3a: Shape Template Editor

This is essentially a template editor — a separate utility app, not part of the game itself. Think of it like a level editor that ships alongside a game. The workflow would be:

Launch the editor
Select a letter/shape to author, prompt user to press the key (e.g. "A")
A canvas displays with the letter rendered large in the background as a visual guide
A default skeleton is shown with draggable joint points
Have an optional "record from pose" button that captures the current MediaPipe skeleton from the webcam and loads those coordinates as a starting point, which you then manually refine.

You drag joints to where they should be for a perfect match of that letter
Some joints you mark as "don't care" — irrelevant to the shape (e.g. for the letter "I", the exact elbow position doesn't matter, only the overall vertical alignment)
Hit "save" control → writes a JSON template file for that letter
Repeat for each letter/shape, show the prompt again

The "don't care" joints are important — they let you weight which landmarks actually matter for scoring each shape, rather than penalizing players for irrelevant joint positions.

### Template File Format
JSON is the right choice. Something like:
json{
  "shape_id": "A",
  "display_name": "Letter A",
  "difficulty": 2,
  "landmarks": {
    "LEFT_WRIST":    {"x": 0.15, "y": 0.1,  "weight": 1.0},
    "RIGHT_WRIST":   {"x": 0.85, "y": 0.1,  "weight": 1.0},
    "LEFT_ELBOW":    {"x": 0.25, "y": 0.35, "weight": 0.5},
    "RIGHT_ELBOW":   {"x": 0.75, "y": 0.35, "weight": 0.5},
    "LEFT_SHOULDER": {"x": 0.35, "y": 0.55, "weight": 0.8},
    "RIGHT_SHOULDER":{"x": 0.65, "y": 0.55, "weight": 0.8},
    "LEFT_HIP":      {"x": 0.4,  "y": 0.75, "weight": 0.3},
    "RIGHT_HIP":     {"x": 0.6,  "y": 0.75, "weight": 0.3}
  }
}
Coordinates are normalized 0–1 so they're resolution-independent. The weight field is how much each joint contributes to the score — 1.0 means it matters a lot, 0.0 means ignore it entirely. You can set weights in the editor with a slider or by clicking a joint to toggle its importance.

### Components of the template editor:

Separate pygame app (editor.py)
Letter background rendering
Draggable skeleton joints
Weight toggle per joint
Save/load JSON templates
Ship with maybe 3 test letters to author first

### Clean dependency structure 
scoring.py          ← shared, no dependencies on game or editor
    ↑                   (pure functions: normalize, compare, weighted_score)
    ├── game.py     ← imports scoring, runs game loop
    └── editor.py   ← imports scoring, shows live score while you pose

## Phase 3b: Live scoring against templates
"Build the live scoring module using the shape template structure.
Build this into the template editor. Have a button for "Live Scoring" when button clicked overlay live camera over the skeleton.
. Display the target shape as a skeleton diagram on screen. Compute and display a live 0–100 score as the player poses. No game logic yet — just continuous scoring."
make the score in large font (72 pt or more)
pressing ESC stops the live scoring / camera view
Load templates from JSON
Normalize player landmarks against template
Weighted scoring
Live score display

### Deliverables:

shapes/ module with template format established
scoring.py with normalization + similarity metric
Live score updating on screen
Visual target shape display

## Phase 4: Game State Machine + Single Round
"Implement the game state machine: IDLE → SHOW_SHAPE → COUNTDOWN → CAPTURING → SCORE_DISPLAY. One player, one round, one randomly selected shape. Timer counts down, captures best score during window, shows result screen. ESC returns to IDLE."
Deliverables:

Full single round playable end to end
Best-score-in-window logic
Score result screen

## Phase 5: Multiplayer Rounds + Leaderboard
"Add player setup screen (enter number of players, names). Implement round robin rotation. After N rounds, show final leaderboard with totals. Add difficulty multiplier per shape, display difficulty badge, apply to final score."
Deliverables:

Multi-player flow
Configurable number of rounds
Final leaderboard screen
Difficulty system wired in

## Phase 6: Shape Library Expansion
"Expand shape library to full Latin alphabet (A–Z) plus 10 geometric shapes (star, triangle, diamond, etc.). Grade each for difficulty 1–3. Add shape set selector on the setup screen."
Why last: you want the template format locked down (Phase 3) before building out a large library, or you'll redo work.
Deliverables:

Full A–Z templates
Geometric set
UI to choose shape set before game starts

## Phase 7: Polish + PyInstaller Final Build
"Add sound effects (countdown beep, score fanfare), animate the score counter, add player silhouette overlay using MediaPipe segmentation for visual flair. Final PyInstaller build with icon, no console window, all assets bundled."
Deliverables:

Audio
Visual polish
Shippable exe

