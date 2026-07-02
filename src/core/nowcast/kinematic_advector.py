import cv2
import numpy as np

class KinematicAdvector:
    def advect(
        self, 
        rain_rate: np.ndarray, 
        map_x: np.ndarray, 
        map_y: np.ndarray, 
        x_grid: np.ndarray, 
        y_grid: np.ndarray, 
        blended_flow_x: np.ndarray, 
        blended_flow_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # We map backward from the departure point using the current grid flow
        map_x_out = cv2.remap(map_x, (x_grid - blended_flow_x).astype(np.float32), (y_grid - blended_flow_y).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        map_y_out = cv2.remap(map_y, (x_grid - blended_flow_x).astype(np.float32), (y_grid - blended_flow_y).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        
        shifted = cv2.remap(
            rain_rate, map_x_out, map_y_out, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=-1.0
        )
        shifted = np.clip(shifted, 0.0, None)
        return shifted, map_x_out, map_y_out
