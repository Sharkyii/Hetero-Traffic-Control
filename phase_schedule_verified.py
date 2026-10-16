"""
phase_schedule_verified.py - Bug 7 Fixed Phase Schedule

Multi-arm signal cycle with true intersection clearance (All-Red Gate):
If active transit vehicles are still crossing the intersection when a new
green phase is scheduled to begin, the green signal is withheld (all entry
stop lines remain red) until the intersection box is physically clear.
"""

GREEN_FRAMES = 125           # Flowing green frames per arm (31.25s @ 4 FPS)
CLEARANCE_FRAMES = 24        # Exactly 6.0s @ 4 FPS all-red gap between green phases
PHASE_FRAMES = GREEN_FRAMES + CLEARANCE_FRAMES  # 149 frames (37.25s @ 4 FPS)

ENTRY_ROADS = [2, 4, 6, 8]           # cycle order, one per arm
ENTRY_TO_EXIT = {2: 1, 4: 3, 6: 5, 8: 7}
EXIT_ROADS = set(ENTRY_TO_EXIT.values())

CYCLE_LENGTH = PHASE_FRAMES * len(ENTRY_ROADS)  # 596



def green_arm_index(sim_frame: int) -> int:
    """Which arm (0-3, indexing ENTRY_ROADS) owns the phase at this frame."""
    return (sim_frame % CYCLE_LENGTH) // PHASE_FRAMES


def active_arm(sim_frame: int) -> int:
    """The entry road number currently in its phase window (green or clearance)."""
    return ENTRY_ROADS[green_arm_index(sim_frame)]


def green_roads(sim_frame: int, transit_cars=None) -> tuple:
    """The entry road number currently green (flowing). During all-red clearance,
    returns empty tuple so all entry stop lines show red and hold new traffic.
    
    Bug 7 Fix:
    If transit_cars is provided and any vehicle is still in 'transit' from a
    PREVIOUS phase (holdover crossing vehicle) at phase boundaries, withhold
    green until the conflict box is clear.
    """
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    if pos_in_phase < GREEN_FRAMES:
        if transit_cars:
            phase_start = sim_frame - pos_in_phase
            # Holdover vehicles: active transit cars from a previous phase
            still_crossing = [
                tc for tc in transit_cars
                if tc.active and tc.state == 'transit'
                and getattr(tc, 'transit_entry_frame', -1) < phase_start
            ]
            if still_crossing:
                return ()
        return (ENTRY_ROADS[green_arm_index(sim_frame)],)
    return ()


def is_green(road: int, sim_frame: int, transit_cars=None) -> bool:
    """Exit roads are never signal-controlled -- always free-flowing."""
    if road in EXIT_ROADS:
        return True
    return road in green_roads(sim_frame, transit_cars=transit_cars)


def remaining_green_frames(road: int, sim_frame: int) -> int:
    """Number of green frames remaining in the current green phase for this road.
    Used by entry stop-line gatekeepers to prevent cars from entering when
    the green phase is about to end.
    Exit roads always return a large number (they never turn red)."""
    if road in EXIT_ROADS:
        return 999999
    current_green = green_roads(sim_frame)
    if road not in current_green:
        return 0
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    return max(0, GREEN_FRAMES - pos_in_phase)


def phase_status(sim_frame: int) -> str:
    """Human-readable status for HUD: 'GREEN' or 'CLEARANCE (Xs GAP)'."""
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    if pos_in_phase < GREEN_FRAMES:
        return "GREEN"
    rem_gap = (PHASE_FRAMES - pos_in_phase) / 4.0
    return f"CLEARANCE ({rem_gap:.1f}s GAP)"

