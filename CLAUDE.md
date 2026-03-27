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
├── shapes/
│   ├── latin.py              # Letter landmark templates
│   ├── greek.py
│   └── geometric.py          # Star, triangle, X, etc.
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
show debug into when starting with 'training-mode' option
"Body not fully visible" warning when key landmarks are low

## Phase 3: Shape Template System + Static Scoring
"Build the shape template data structure and scoring module. Implement normalization (bounding box scale + center). Load 3 test shapes: I, T, X. Display the target shape as a skeleton diagram on screen. Compute and display a live 0–100 score as the player poses. No game logic yet — just continuous scoring."
Why here: this is the hardest algorithmic piece. Isolating it as its own phase means you can tune the scoring math interactively before game logic complicates things.
Deliverables:

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

