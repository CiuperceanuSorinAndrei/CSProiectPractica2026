import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import scipy.optimize
from collections import defaultdict
from scipy.spatial.distance import cdist

from src.core.domain import StormCell, CellDiagnostics
from src.core.algorithms_config import config

class FADiagnosis(Enum):
    BAD_ADVECTION = "BAD_ADVECTION"
    BAD_MATCHING = "BAD_MATCHING"
    SPLIT_ERROR = "SPLIT_ERROR"
    MERGE_ERROR = "MERGE_ERROR"
    REACTION_TOO_STRONG = "REACTION_TOO_STRONG"
    DIFFUSION_SUPPORT = "DIFFUSION_SUPPORT"
    LIFECYCLE_DELAY = "LIFECYCLE_DELAY"
    UNSTABLE_TRACK = "UNSTABLE_TRACK"
    UNKNOWN = "UNKNOWN"
    FAILED_TO_DISSIPATE = "FAILED_TO_DISSIPATE"

@dataclass(frozen=True)
class MatchingDiagnostics:
    distance: float
    iou: float
    assignment_cost: float
    matched_cell_id: str | None

@dataclass(frozen=True)
class FalseAlarmRecord:
    prediction_id: str
    frame: int
    horizon: str
    diagnostics: CellDiagnostics | None
    matching: MatchingDiagnostics
    classification: FADiagnosis
    confidence: float
    
    # Context data for reporting
    predicted_area: float
    predicted_energy: float
    predicted_dE: float
    predicted_phase: str
    age_frames: int

class FalseAlarmInspector:
    def __init__(self, area_conservation_tolerance: float = 0.20):
        self.area_conservation_tolerance = area_conservation_tolerance
        
        # History structures for thresholds and lifecycle
        self.episode_reaction_gains = []
        
        # Tracking history for UNSTABLE_TRACK
        # cell_id -> list of observed frames
        self.obs_cell_history: dict[str, list[int]] = {}
        self.total_predictions = 0
        self.total_observations = 0
        
        # Records to hold all false alarms
        self.far_records: list[FalseAlarmRecord] = []

    def _calculate_iou(self, pred: StormCell, obs: StormCell) -> float:
        # Simplification: area of intersection of circles
        dist = np.hypot(pred.predicted_centroid_x - obs.centroid_x, pred.predicted_centroid_y - obs.centroid_y)
        r_pred = max(1.0, np.sqrt(pred.area_pixels / np.pi))
        r_obs = max(1.0, np.sqrt(obs.area_pixels / np.pi))
        
        if dist >= r_pred + r_obs:
            return 0.0
        if dist <= abs(r_pred - r_obs):
            return 1.0 # Smaller is fully inside
            
        # Approximation of IOU based on distance vs sum of radii
        # A true circle intersection is complex, we use a fast heuristic for the cost matrix
        overlap = max(0.0, (r_pred + r_obs - dist) / (r_pred + r_obs))
        return overlap

    def collect_observations(self, frame_idx: int, observed_cells: list[StormCell]):
        """Called per frame to build observation history."""
        for c in observed_cells:
            if c.cell_id not in self.obs_cell_history:
                self.obs_cell_history[c.cell_id] = []
            self.obs_cell_history[c.cell_id].append(frame_idx)

    def evaluate_episode(self, episode_data: list[tuple[int, str, list[StormCell], list[StormCell]]]):
        """
        episode_data: list of (frame_idx, horizon_name, predicted_cells, observed_cells)
        """
        # First pass: collect reaction gains to build historical percentiles
        for _, _, preds, _ in episode_data:
            for p in preds:
                if p.diagnostics is not None:
                    self.episode_reaction_gains.append(p.diagnostics.reaction_gain)
                    
        p95_reaction = 0.0
        if len(self.episode_reaction_gains) > 0:
            p95_reaction = np.percentile(self.episode_reaction_gains, 95)
            
        # Second pass: Assignment and classification
        for frame_idx, horizon_name, preds, obs in episode_data:
            self._evaluate_step(frame_idx, horizon_name, preds, obs, p95_reaction)
            
    def _evaluate_step(self, frame_idx: int, horizon_name: str, preds: list[StormCell], obs: list[StormCell], p95_reaction: float):
        self.total_predictions += len(preds)
        self.total_observations += len(obs)
        if not preds:
            return
            
        unassigned_preds = []
        matching_diags = {}
        
        if not obs:
            for p in preds:
                unassigned_preds.append(p)
                matching_diags[p.cell_id] = MatchingDiagnostics(9999.0, 0.0, 9999.0, None)
        else:
            # Phase 3: Vectorized Distance Matrix
            preds_coords = np.array([[p.predicted_centroid_x, p.predicted_centroid_y] for p in preds])
            obs_coords = np.array([[o.centroid_x, o.centroid_y] for o in obs])
            dist_matrix = cdist(preds_coords, obs_coords)
            
            # Hungarian Assignment
            cost_matrix = np.zeros((len(preds), len(obs)))
            for i, p in enumerate(preds):
                r_pred = np.sqrt(p.area_pixels / np.pi)
                for j, o in enumerate(obs):
                    dist = dist_matrix[i, j]
                    r_obs = np.sqrt(o.area_pixels / np.pi)
                    
                    iou = self._calculate_iou(p, o)
                    
                    # Invariant radius threshold
                    max_radius = max(config.INSPECTOR_BASE_RADIUS, 0.5 * (r_pred + r_obs))
                    
                    if dist > max_radius * config.INSPECTOR_MAX_RADIUS_MULT: # Far away
                        cost_matrix[i, j] = 9999.0
                    else:
                        # Cost is distance minus IOU bonus
                        cost_matrix[i, j] = dist - (iou * 10.0)
                        
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            assigned_p_indices = set()
            for r, c in zip(row_ind, col_ind):
                cost = cost_matrix[r, c]
                p = preds[r]
                o = obs[c]
                
                dist = dist_matrix[r, c]
                iou = self._calculate_iou(p, o)
                
                r_pred = np.sqrt(p.area_pixels / np.pi)
                r_obs = np.sqrt(o.area_pixels / np.pi)
                max_radius = max(config.INSPECTOR_BASE_RADIUS, 0.5 * (r_pred + r_obs))
                
                # Validation of assignment
                if cost < 9000.0 and dist <= max_radius * config.INSPECTOR_MAX_RADIUS_MULT:
                    assigned_p_indices.add(r)
                else:
                    unassigned_preds.append(p)
                    matching_diags[p.cell_id] = MatchingDiagnostics(dist, iou, cost, None)
                    
            for i, p in enumerate(preds):
                if i not in assigned_p_indices and p not in unassigned_preds:
                    unassigned_preds.append(p)
                    # Find nearest obs for diagnostics
                    if len(obs) > 0:
                        dists = dist_matrix[i, :]
                        min_idx = int(np.argmin(dists))
                        min_dist = dists[min_idx]
                        iou = self._calculate_iou(p, obs[min_idx])
                        matching_diags[p.cell_id] = MatchingDiagnostics(min_dist, iou, 9999.0, None)
                    else:
                        matching_diags[p.cell_id] = MatchingDiagnostics(9999.0, 0.0, 9999.0, None)

        # Classify False Alarms
        for p in unassigned_preds:
            md = matching_diags[p.cell_id]
            diagnosis, confidence = self._classify_far(p, md, obs, p95_reaction, frame_idx)
            
            record = FalseAlarmRecord(
                prediction_id=p.cell_id,
                frame=frame_idx,
                horizon=horizon_name,
                diagnostics=p.diagnostics,
                matching=md,
                classification=diagnosis,
                confidence=confidence,
                predicted_area=float(p.area_pixels),
                predicted_energy=p.E,
                predicted_dE=p.dE,
                predicted_phase=p.lifecycle_phase,
                age_frames=p.age_frames
            )
            self.far_records.append(record)
            
    def _classify_far(self, p: StormCell, md: MatchingDiagnostics, obs: list[StormCell], p95_reaction: float, frame_idx: int) -> tuple[FADiagnosis, float]:
        scores = {k: 0.0 for k in FADiagnosis}
        
        r_pred = max(1.0, np.sqrt(p.area_pixels / np.pi))
        
        # 3. UNSTABLE_TRACK & FAILED_TO_DISSIPATE
        history = self.obs_cell_history.get(p.cell_id, [])
        is_dead = False
        if history:
            last_seen = history[-1]
            frames_since_seen = frame_idx - last_seen
            if 0 < frames_since_seen <= 3:
                # It disappeared for 1-3 frames, tracker lost it, but prediction advected it
                scores[FADiagnosis.UNSTABLE_TRACK] = 0.9
            elif frames_since_seen > 3:
                # Tracker hasn't seen this cell in over 15 minutes. It's dead in reality.
                # If we are still predicting it, our thermodynamic decay is too slow.
                scores[FADiagnosis.FAILED_TO_DISSIPATE] = 1.0 + min(1.0, frames_since_seen / 5.0)  # > 1.0 to override BAD_ADVECTION
                is_dead = True
                
        # 1. BAD_ADVECTION: No observations anywhere near
        # Only penalize advection if the cell is not already confirmed dead!
        if md.distance > r_pred * 4 and not is_dead:
            scores[FADiagnosis.BAD_ADVECTION] = 1.0 - np.exp(-md.distance / (r_pred * 5 + 1e-6))
            
        # 2. BAD_MATCHING: Observation is near, but wasn't assigned
        elif md.distance <= r_pred * 2 and not is_dead:
            scores[FADiagnosis.BAD_MATCHING] = np.exp(-md.distance / (r_pred + 1e-6)) * (1.0 - md.iou)
                
        # 4. SPLIT_ERROR / MERGE_ERROR
        if obs:
            # Check Split (1 pred -> Many obs)
            close_obs = [o for o in obs if np.hypot(p.predicted_centroid_x - o.centroid_x, p.predicted_centroid_y - o.centroid_y) < r_pred * 3]
            if len(close_obs) > 1:
                total_obs_area = sum(o.area_pixels for o in close_obs)
                area_sim = min(p.area_pixels, total_obs_area) / max(p.area_pixels, total_obs_area + 1e-6)
                
                obs_cx = np.average([o.centroid_x for o in close_obs], weights=[o.area_pixels for o in close_obs])
                obs_cy = np.average([o.centroid_y for o in close_obs], weights=[o.area_pixels for o in close_obs])
                dist_c = np.hypot(p.predicted_centroid_x - obs_cx, p.predicted_centroid_y - obs_cy)
                centroid_sim = np.exp(-dist_c / (r_pred + 1e-6))
                
                total_obs_energy = sum(o.E for o in close_obs) if hasattr(close_obs[0], 'E') else total_obs_area / 1000.0
                energy_sim = min(p.E, total_obs_energy) / max(p.E, total_obs_energy + 1e-6)
                
                split_score = 0.4 * area_sim + 0.3 * centroid_sim + 0.3 * energy_sim
                if area_sim > (1.0 - self.area_conservation_tolerance):
                    scores[FADiagnosis.SPLIT_ERROR] = split_score
                    
        # 5. Thermodynamics
        if p.diagnostics:
            # DIFFUSION_SUPPORT
            if p.diagnostics.diffusion_fraction > 0.5:
                scores[FADiagnosis.DIFFUSION_SUPPORT] = p.diagnostics.diffusion_fraction
                
            # REACTION_TOO_STRONG
            if p.diagnostics.reaction_gain > p95_reaction and p95_reaction > 0:
                excess = (p.diagnostics.reaction_gain - p95_reaction) / p95_reaction
                scores[FADiagnosis.REACTION_TOO_STRONG] = min(1.0, 0.5 + excess)
                
            # LIFECYCLE_DELAY
            if p.lifecycle_phase == 'DISSIPATION' and p.E > 1.0:
                # Still alive despite dissipating
                scores[FADiagnosis.LIFECYCLE_DELAY] = min(1.0, p.age_frames / 20.0)
                
        # Resolve Winner
        winner = max(scores, key=scores.get)
        confidence = scores[winner]
        
        if confidence < 0.2:
            return FADiagnosis.UNKNOWN, 1.0
            
        return winner, confidence

    def generate_report(self):
        FalseAlarmReporter.generate_report(self)


class FalseAlarmReporter:
    """Handles the formatting and printing of false alarm reports (SRP Fix)."""
    @staticmethod
    def generate_report(inspector: FalseAlarmInspector):
        print("\n" + "="*50)
        print(" FALSE ALARM INSPECTOR REPORT")
        print("="*50)
        
        if not inspector.far_records:
            print("No False Alarms detected.")
            return
            
        total_fars = len(inspector.far_records)
        report = []
        report.append(f"Total False Alarms: {total_fars}")
        report.append(f"Total Evaluated Predictions: {inspector.total_predictions}")
        report.append(f"Total True Observations: {inspector.total_observations}")
        if inspector.total_predictions > 0:
            far_rate = (total_fars / inspector.total_predictions) * 100
            report.append(f"TRUE FAR RATE: {far_rate:.2f}% (False Alarms / Total Predictions)")
        for line in report: print(line)
        
        # Breakdown
        counts = defaultdict(int)
        for r in inspector.far_records:
            counts[r.classification] += 1
            
        print("\n--- BREAKDOWN ---")
        for diag, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_fars) * 100
            print(f"{diag.value.ljust(25)} {count} ({pct:.1f}%)")
            
        # Averages per Category
        print("\n--- AVERAGES PER CATEGORY ---")
        for diag in FADiagnosis:
            cat_records = [r for r in inspector.far_records if r.classification == diag]
            if not cat_records: continue
            
            avg_area = np.mean([r.predicted_area for r in cat_records])
            avg_E = np.mean([r.predicted_energy for r in cat_records])
            avg_dist = np.mean([r.matching.distance for r in cat_records])
            avg_age = np.mean([r.age_frames for r in cat_records])
            
            diff_gains = [r.diagnostics.diffusion_delta for r in cat_records if r.diagnostics]
            avg_diff = np.mean(diff_gains) if diff_gains else 0.0
            
            rx_gains = [r.diagnostics.reaction_gain for r in cat_records if r.diagnostics]
            avg_rx = np.mean(rx_gains) if rx_gains else 0.0
            
            print(f"\n{diag.value}:")
            print(f"  Area: {avg_area:.1f} px | Energy: {avg_E:.2f} | Dist: {avg_dist:.1f} px")
            print(f"  Age: {avg_age:.1f} frm | DiffGain: {avg_diff:.3f} | RxGain: {avg_rx:.3f}")
            
        # Worst Offenders
        print("\n--- WORST OFFENDERS (Top 10 by Confidence) ---")
        sorted_records = sorted(inspector.far_records, key=lambda r: r.confidence, reverse=True)
        for i, r in enumerate(sorted_records[:10]):
            print(f"\n#{i+1} [{r.classification.value}] (Conf: {r.confidence:.2f})")
            print(f"  Cell: {r.prediction_id} | Frame: {r.frame} | Horizon: {r.horizon}")
            print(f"  E: {r.predicted_energy:.2f} | dE: {r.predicted_dE:.3f} | Phase: {r.predicted_phase}")
            print(f"  Matching Dist: {r.matching.distance:.1f} px | IOU: {r.matching.iou:.2f}")
            if r.diagnostics:
                print(f"  Reaction Gain: {r.diagnostics.reaction_gain:.3f} | Diffusion Delta: {r.diagnostics.diffusion_delta:.3f}")
