import asyncio
import cv2
import numpy as np
from cv2 import aruco
from typing import Any

# Simulated walls array
from walls import wallArr

# A dummy ConnectionState
from connectionState import ConnectionState

class CameraModule:
    """
    A camera module that:
    - Reads frames from a webcam
    - Detects ArUco markers (including Pacman as ID=0)
    - Draws an annotation overlay using OpenCV
    - Sends pacman location to some server
    - Displays the frames in a live window
    """

    def __init__(self, state: ConnectionState) -> None:
        self.state = state

        # A dictionary of 4x4 ArUco markers
        self.dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        # Detector
        self.detector = aruco.ArucoDetector(self.dictionary, aruco.DetectorParameters())

        # Initialize capture
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None

    async def decisionLoop(self) -> None:
        """
        Asynchronous decision loop for CV. We:
          1) read the latest frame
          2) localize pacman + annotate
          3) display the annotated frame
          4) let other async tasks run
        """
        while self.state.isConnected():
            # Get a frame (BGR by default)
            _, frame = self.cap.read()
            if frame is None:
                print("ERR: NO IMAGE")
                await asyncio.sleep(0)
                continue
            
            cv2.imshow("Camera", frame)

            # Localize Pacman, get (row, col), also overlay annotations in-place
            pacman_row, pacman_col = self.localize(frame, annotate=True)

            # If there's no wall there, send to server
            if not self.wallAt(pacman_row, pacman_col):
                self.state.send(pacman_row, pacman_col)

            # Display the frame with OpenCV
            cv2.imshow("Annotated", frame)

            # Check if the user pressed ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Yield to other async tasks
            await asyncio.sleep(0)

        # Cleanup
        # self.cap.release()
        cv2.destroyAllWindows()

    def wallAt(self, row: int, col: int) -> bool:
        """
        Helper function to check if a wall is at a given location.
        """
        if (row < 0 or row >= 31) or (col < 0 or col >= 28):
            return True
        return bool((wallArr[row] >> col) & 1)

    def localize(self, frame: np.ndarray, annotate: bool = False) -> tuple[int, int]:
        """
        Detect ArUco markers, find Pacman (ID=0), 
        compute location in the grid, and optionally annotate the frame in-place.
        """

        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is None:
            print("ERR: No markers detected...")
            return (32, 32)

        # Draw all detected markers for debugging
        # (This outlines the markers + IDs on 'frame'.)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Collect (id, centroid) pairs
        ids_centroids: list[tuple[int, np.ndarray]] = []
        foundPacman = False

        for j in range(len(ids)):
            marker_id = ids[j, 0]
            # If we only care about IDs 0..6, skip others
            if marker_id > 6:
                continue

            if marker_id == 0:
                foundPacman = True

            # Compute centroid of this marker
            c = corners[j][0]  # shape (4,2)
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())
            ids_centroids.append((marker_id, np.array([cx, cy])))

        if not foundPacman:
            print("ERR: Pacman not found")
            return (32, 32)

        # Sort by ID
        ids_centroids.sort(key=lambda x: x[0])
        sorted_ids, sorted_centroids = zip(*ids_centroids)

        # Check if top or bottom half
        topHalf = (sorted_ids == (0, 1, 2, 3, 4))
        bottomHalf = (sorted_ids == (0, 3, 4, 5, 6))
        bothHalves = (sorted_ids == (0, 1, 2, 5, 6))
        if not (topHalf or bottomHalf or bothHalves):
            print("ERR: The image is neither top nor bottom half")
            return (32, 32)

        # Maze dimensions for top/bottom
        width = 28
        height = 31 if bothHalves else 16 if topHalf else 15
        offset = 0 if topHalf or bothHalves else 16

        # The four corners are the next 4 IDs after Pacman, i.e. 1,2,3,4 (if top) or 3,4,5,6 (if bottom)
        four_corners = np.array(sorted_centroids[1:5]).astype('float32')
        # Perspective mapping
        result = 100 * np.array([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(four_corners, result)
        inverse = np.linalg.inv(matrix)

        # Pacman centroid is the first in sorted_centroids
        px, py = sorted_centroids[0]
        # Transform that point
        vec = matrix @ np.array([px, py, 1], dtype=float)
        pacman_transformed_colf = (vec[0]/vec[2]) / 100.0 - 0.5
        pacman_transformed_rowf = (vec[1]/vec[2]) / 100.0 - 0.5

        # Round to the nearest cell
        row_approx = round(pacman_transformed_rowf)
        col_approx = round(pacman_transformed_colf)

        # Search neighbors for a valid open cell
        neighbors = []
        for r in range(row_approx - 1, row_approx + 2):
            for c in range(col_approx - 1, col_approx + 2):
                if not self.wallAt(r + offset, c):
                    dist_sq = ((r - pacman_transformed_rowf)**2
                             + (c - pacman_transformed_colf)**2)
                    neighbors.append((dist_sq, (r + offset, c)))
        pacman_row, pacman_col = None, None
        if neighbors:
            pacman_row, pacman_col = min(neighbors, key=lambda x: x[0])[1]
        else:
            print("ERR: Pacman is apparently in a wall area.")
            #return (32, 32)

        # The best neighbor by distance
        
        # --------------------------
        # Overlays (if annotate=True)
        # --------------------------
        if annotate:
            # 1) Draw the entire grid of cells (walls vs. open)
            #    to see them in the perspective of the camera
            for r in range(height):
                for c in range(width):
                    # Transform from maze coords to pixel
                    out = inverse @ np.array([c*100 + 50, r*100 + 50, 1], dtype=float)
                    world_px = int(round(out[0] / out[2]))
                    world_py = int(round(out[1] / out[2]))

                    if self.wallAt(r + offset, c):
                        # e.g. magenta for wall
                        cv2.circle(frame, (world_px, world_py), 2, (255, 0, 255), -1)
                    else:
                        # e.g. cyan for open space
                        cv2.circle(frame, (world_px, world_py), 2, (255, 255, 0), -1)

            # 2) Draw pacman’s location in a distinct color (e.g. yellow)
            if pacman_row is not None and pacman_col is not None:
                out = inverse @ np.array([(pacman_col)*100 + 50,
										(pacman_row - offset)*100 + 50,
										1], dtype=float)
                pac_px = int(round(out[0] / out[2]))
                pac_py = int(round(out[1] / out[2]))
                cv2.drawMarker(frame, (pac_px, pac_py), (0, 255, 255), markerType=cv2.MARKER_STAR, 
							markerSize=8, thickness=1)
            #frame = cv2.warpPerspective(frame, matrix, (100, 100))

        if pacman_row is None and pacman_col is None:
            return (32, 32)	
        return (pacman_row, pacman_col)
