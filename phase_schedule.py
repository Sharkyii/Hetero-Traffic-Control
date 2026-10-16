"""
phase_schedule.py - the multi-arm signal cycle: one arm green at a time,
170 frames each, cycling through all 4 arms in the order suggested by the
real recorded data (od4: road2 green first, then road4, then road6).

Each arm is ONE physical carriageway with two lanes: an ENTRY lane (even
number, has the real signal/stop-line, is what queues on red and flows on
green) and an EXIT lane (odd number, cars leave the junction there --
no signal, always free-flowing, never a spawn point). Confirmed from the
picker overlay: roads (1,2), (3,4), (5,6), (7,8) are each the two lanes of
one physical road, opposite directions.

Signal phases are keyed to ENTRY roads only -- exit roads are never red.

This is a uniform synthetic cycle (not the exact one-off recorded windows),
applied identically across every source recording (od1/od4/od5/od6), per
"all frames same" -- so a given sim_frame always means the same phase
regardless of which source is loaded.
"""

PHASE_FRAMES = 170
GREEN_FRAMES = 125           # Flowing green frames per arm (31.25s @ 4 FPS)
CLEARANCE_FRAMES = 45        # All-red clearance buffer (11.25s @ 4 FPS) - clears the junction box

ENTRY_ROADS = [2, 4, 6, 8]           # cycle order, one per arm
ENTRY_TO_EXIT = {2: 1, 4: 3, 6: 5, 8: 7}
EXIT_ROADS = set(ENTRY_TO_EXIT.values())

CYCLE_LENGTH = PHASE_FRAMES * len(ENTRY_ROADS)  # 680


def green_arm_index(sim_frame: int) -> int:
    """Which arm (0-3, indexing ENTRY_ROADS) owns the phase at this frame."""
    return (sim_frame % CYCLE_LENGTH) // PHASE_FRAMES


def active_arm(sim_frame: int) -> int:
    """The entry road number currently in its phase window (green or clearance)."""
    return ENTRY_ROADS[green_arm_index(sim_frame)]


def green_roads(sim_frame: int) -> tuple:
    """The entry road number currently green (flowing). During all-red clearance,
    returns empty tuple so all entry stop lines show red and hold new traffic."""
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    if pos_in_phase < GREEN_FRAMES:
        return (ENTRY_ROADS[green_arm_index(sim_frame)],)
    return ()


def is_green(road: int, sim_frame: int) -> bool:
    """Exit roads are never signal-controlled -- always free-flowing."""
    if road in EXIT_ROADS:
        return True
    return road in green_roads(sim_frame)


def remaining_green_frames(road: int, sim_frame: int) -> int:
    """Number of green frames remaining in the current green phase for this road.
    Used by entry stop-line gatekeepers to prevent cars from entering when
    insufficient time remains to clear the intersection box."""
    if road in EXIT_ROADS:
        return 999999
    if road not in ENTRY_ROADS:
        return 0
    arm_idx = ENTRY_ROADS.index(road)
    current_arm_idx = green_arm_index(sim_frame)
    if arm_idx != current_arm_idx:
        return 0
    pos_in_phase = (sim_frame % CYCLE_LENGTH) % PHASE_FRAMES
    if pos_in_phase < GREEN_FRAMES:
        return GREEN_FRAMES - pos_in_phase
    return 0


def frames_until_green(road: int, sim_frame: int) -> int:
    """0 if currently green (or an exit road); otherwise frames remaining
    until this entry road's next green phase starts."""
    if road in EXIT_ROADS:
        return 0
    try:
        arm_idx = ENTRY_ROADS.index(road)
    except ValueError:
        raise ValueError(f"road {road} is not a known entry road")

    phase_start = arm_idx * PHASE_FRAMES
    pos_in_cycle = sim_frame % CYCLE_LENGTH
    if phase_start <= pos_in_cycle < phase_start + GREEN_FRAMES:
        return 0
    return (phase_start - pos_in_cycle) % CYCLE_LENGTH


if __name__ == "__main__":
    # quick self-check: print the schedule for one full cycle
    for f in range(0, CYCLE_LENGTH, 20):
        pos_in_phase = (f % CYCLE_LENGTH) % PHASE_FRAMES
        status = "GREEN" if pos_in_phase < GREEN_FRAMES else "CLEARANCE (ALL-RED)"
        gr = green_roads(f)
        print(f"frame {f:>4}  arm {active_arm(f)}  status: {status:<20}  green roads: {gr}")
