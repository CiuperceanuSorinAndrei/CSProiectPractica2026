import numpy as np
import scipy.ndimage

class KinematicAdvector:
    def advect(
        self, 
        rain_rate: np.ndarray, 
        shift_y: float, 
        shift_x: float
    ) -> np.ndarray:
        """Advects the precipitation field using a uniform translation vector.
        
        Uses scipy.ndimage.shift with order=1 (bilinear) and cval=0.0 (safe boundary padding).
        """
        # Shift expects (y_shift, x_shift)
        shifted = scipy.ndimage.shift(
            rain_rate, 
            shift=(shift_y, shift_x), 
            order=0, 
            cval=0.0
        )
        return np.clip(shifted, 0.0, None)
