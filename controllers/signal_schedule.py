"""
Signal Schedule and Phase Transitions.

Defines fixed-time multi-arm signal phase scheduling with an All-Red Gate,
ensuring strict clearance intervals between phase switches.
"""

from typing import Tuple, List, Optional, Any

GREEN_FRAMES = 125           # Flowing green frames per arm (31.25s @ 4 FPS)
CLEARANCE_FRAMES = 24        # Exactly 6.0s @ 4 FPS all-red gap between green phases
PHASE_FRAMES = GREEN_FRAMES + CLEARANCE_FRAMES  # 149 frames (37.25s @ 4 FPS)

ENTRY_ROADS = [2, 4, 6, 8]
ENTRY_TO_EXIT = {2: 1, 4: 3, 6: 5, 8: 7}
EXIT_ROADS = set(ENTRY_TO_EXIT.values())
CYCLE_LENGTH = PHASE_FRAMES * len(ENTRY_ROADS)  # 596 frames (149.0s)


def green_arm_index(sim_frame: int) -> int:
    return (sim_frame % CYCLE_LENGTH) // PHASE_FRAMES


def active_arm(sim_frame: int) -> int:
    return ENTRY_ROADS[green_arm_index(sim_frame)]


def green_roads(sim_frame: int, transit_cars: Optional[list] = None) -> Tuple[int, ...]:
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    if pos_in_phase < GREEN_FRAMES:
        if transit_cars:
            phase_start = sim_frame - pos_in_phase
            still_crossing = [
                tc for tc in transit_cars
                if tc.active and getattr(tc, "state", "") == "transit"
                and getattr(tc, "transit_entry_frame", -1) < phase_start
            ]
            if still_crossing:
                return ()
        return (ENTRY_ROADS[green_arm_index(sim_frame)],)
    return ()


def is_green(road: int, sim_frame: int, transit_cars: Optional[list] = None) -> bool:
    if road in EXIT_ROADS:
        return True
    return road in green_roads(sim_frame, transit_cars=transit_cars)


def remaining_green_frames(road: int, sim_frame: int) -> int:
    if road in EXIT_ROADS:
        return 999999
    current_green = green_roads(sim_frame)
    if road not in current_green:
        return 0
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    return max(0, GREEN_FRAMES - pos_in_phase)


def phase_status(sim_frame: int) -> str:
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    if pos_in_phase < GREEN_FRAMES:
        return "GREEN"
    rem_gap = (PHASE_FRAMES - pos_in_phase) / 4.0
    return f"CLEARANCE ({rem_gap:.1f}s GAP)"
