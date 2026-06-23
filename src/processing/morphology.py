import numpy as np
import scipy.ndimage as ndi

def extract_storm_cells(rain_matrix, threshold=0.5, min_size=5):
    """
    Detecteaza nucleele de furtuna pe baza unui prag de precipitatii.
    """
    # Binarizare: pastram doar pixelii care depasesc pragul de ploaie
    binary_mask = rain_matrix >= threshold
    
    # Eliminare zgomot: stergem pixelii izolati
    clean_mask = ndi.binary_opening(binary_mask, structure=np.ones((3,3)))
    
    # Etichetare: identificam grupurile de pixeli conectati
    labeled_mask, num_features = ndi.label(clean_mask)
    
    cells = []
    if num_features == 0:
        return cells
        
    # Calcul centre de greutate pentru fiecare formatiune
    centroids = ndi.center_of_mass(rain_matrix, labeled_mask, range(1, num_features + 1))
    
    for i in range(1, num_features + 1):
        cell_pixels = np.sum(labeled_mask == i)
        
        # Ignoram formatiunile prea mici
        if cell_pixels < min_size:
            continue
            
        y_center, x_center = centroids[i-1]
        max_intensity = np.max(rain_matrix[labeled_mask == i])
        
        cells.append({
            "id": i,
            "centroid_y": y_center,
            "centroid_x": x_center,
            "area_pixels": cell_pixels,
            "max_intensity": max_intensity
        })
        
    return cells