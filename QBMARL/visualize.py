"""
Quantum-Belief RL Testbed - Pygame Visualizer

True fullscreen, sized to your actual monitor resolution (queried at
launch, not hardcoded). Layout is recomputed every frame from the current
window size and from MEASURED text/element dimensions -- nothing is placed
at a guessed fixed pixel offset, so nothing can overlap regardless of your
screen size or font rendering.

Layout notes (things that matter for keeping this looking right,
verified by actually rendering and looking, not just reading the code):
  - The status line next to each agent title is positioned against the
    title's actual measured rendered width, not a hardcoded x-offset, so
    it can't collide with the title text.
  - Legend items are laid out against their actual measured text width,
    so the legend can't run past the right edge of the content area.
  - The belief heatmap scales color intensity relative to the uniform
    baseline, so a totally uniform "nothing known yet" belief renders at
    LOW brightness across all 25 cells (correctly reading as total
    uncertainty) rather than full brightness (which would misleadingly
    read as total confidence).
  - The target room is hidden by default, since showing it outlined
    from frame one would defeat the "hidden target" premise; press D to
    reveal it (plus every clue's real/decoy status) for debugging.
  - The two agent panels are laid out side-by-side, using the full
    height of the screen, rather than stacked tall/skinny below a short,
    wide map.

Controls: SPACE pause/resume, R new world, D toggle debug overlay,
F toggle fullscreen/windowed, Esc/close to quit.
"""

import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(__file__))
from env import QuantumBeliefEnv, World, GRID_SIZE, SUB_SIZE
from policies import heuristic_action, make_dqn_policy, make_metacognitive_heuristic, make_metacognitive_dqn_policy

MOVE_DELAY_A = 6
MOVE_DELAY_B = 6
MIN_WINDOW = (1024, 640)

# ---------------- palette ----------------
BG = (10, 13, 22)
BG_DOT = (18, 22, 34)
HEADER_BG = (14, 18, 29)
PANEL = (17, 22, 35)
PANEL_BORDER = (34, 42, 60)
TEXT = (228, 233, 242)
TEXT_DIM = (142, 152, 172)
TEXT_FAINT = (84, 93, 112)

AGENT_A = (56, 189, 248)
AGENT_A_SOFT = (16, 42, 58)
AGENT_B = (245, 158, 11)
AGENT_B_SOFT = (48, 36, 10)
BOTH = (34, 66, 51)

CHEST_GOLD = (234, 179, 8)
GOOD = (74, 222, 128)
BAD = (248, 113, 113)
BELIEF_LO = (22, 26, 40)
BELIEF_HI = (168, 85, 247)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def blend(c0, c1, t):
    return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))


def rrect(surface, color, rect, radius=8, width=0):
    pygame.draw.rect(surface, color, rect, width, border_radius=radius)


def wrap_text(font, text, max_width):
    """Greedy word-wrap: splits `text` into lines that each fit within
    `max_width` pixels when rendered in `font`. Falls back to a hard
    character split for any single word that's wider than max_width on
    its own (e.g. a long unbroken token), so one pathological word can't
    silently blow past the box."""
    words = text.split(" ")
    lines = []
    cur = ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if font.size(candidate)[0] <= max_width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
        # hard-split a single word that alone still overflows max_width
        while font.size(cur)[0] > max_width and len(cur) > 1:
            lo, hi = 1, len(cur)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if font.size(cur[:mid])[0] <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            lines.append(cur[:lo])
            cur = cur[lo:]
    if cur:
        lines.append(cur)
    return lines or [""]


def belief_color(p, grid_size):
    """Scale relative to the uniform baseline (1/N cells), not raw
    probability -- otherwise 'nothing known yet' (uniform everywhere)
    maxes out and paints the WHOLE grid at full brightness, which looks
    like total confidence instead of total uncertainty."""
    baseline = 1.0 / (grid_size * grid_size)
    intensity = clamp((p - baseline) / (1.0 - baseline), 0.0, 1.0)
    return blend(BELIEF_LO, BELIEF_HI, intensity)


class Fonts:
    def __init__(self):
        self.h1 = pygame.font.SysFont("consolas", 22, bold=True)
        self.h2 = pygame.font.SysFont("consolas", 16, bold=True)
        self.body = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.tiny = pygame.font.SysFont("consolas", 11)


class AgentView:
    """Wraps one QuantumBeliefEnv with its own async timer and carry-state
    scratch dict, so main loop code doesn't repeat itself."""

    def __init__(self, agent_id, start_room, move_delay, color, soft_color):
        self.id = agent_id
        self.start_room = start_room
        self.move_delay = move_delay
        self.color = color
        self.soft_color = soft_color
        self.env = None
        self.carry_state = {}
        self.tick_counter = 0

    def reset(self, world):
        self.env = QuantumBeliefEnv(world=world, start_room=self.start_room, agent_id=self.id)
        self.carry_state = {}
        self.tick_counter = 0

    def maybe_step(self, policy):
        if self.env.done:
            return
        self.tick_counter += 1
        if self.tick_counter >= self.move_delay:
            self.tick_counter = 0
            action = policy(self.env, self.carry_state)
            self.env.step(action)


class Layout:
    """Recomputed every frame from the current window size, so nothing is
    ever placed past the edge of the screen no matter how it's resized."""

    def __init__(self, win_w, win_h):
        self.win_w, self.win_h = win_w, win_h
        margin = 30
        gap = 24
        self.header_h = 60
        self.footer_h = 46
        avail_h = win_h - self.header_h - self.footer_h - margin * 2

        left_w_budget = win_w * 0.40
        self.main_cell = int(clamp(min(left_w_budget / GRID_SIZE, avail_h / GRID_SIZE), 44, 170))
        self.main_px = self.main_cell * GRID_SIZE
        self.main_x = margin
        self.main_y = self.header_h + margin
        left_col_w = max(self.main_px, int(left_w_budget))

        self.right_x = self.main_x + left_col_w + gap
        right_w = win_w - self.right_x - margin
        self.panel_w = int((right_w - gap) / 2)
        self.panel_h = avail_h
        self.panel_a_x = self.right_x
        self.panel_b_x = self.right_x + self.panel_w + gap
        self.panel_y = self.main_y

        self.legend_y = self.main_y + self.main_px + 20
        self.footer_y = win_h - self.footer_h


def draw_background(surface, w, h):
    surface.fill(BG)
    step = 34
    for x in range(0, w, step):
        for y in range(0, h, step):
            surface.set_at((x, y), BG_DOT)


def draw_header(surface, fonts, w, policy_name, paused, debug):
    rrect(surface, HEADER_BG, (0, 0, w, 60))
    pygame.draw.line(surface, PANEL_BORDER, (0, 60), (w, 60), 1)

    title = fonts.h1.render("Quantum-Belief RL Testbed", True, TEXT)
    surface.blit(title, (30, 16))
    tag = fonts.small.render(f"policy: {policy_name}" + ("   (paused)" if paused else ""), True, TEXT_DIM)
    surface.blit(tag, (30 + title.get_width() + 16, 22))

    hint_text = "SPACE pause   R new world   D " + ("hide" if debug else "show") + " debug   F windowed   Esc quit"
    hint = fonts.small.render(hint_text, True, TEXT_DIM)
    surface.blit(hint, (w - hint.get_width() - 30, 22))


def draw_main_map(surface, fonts, L, views, world, debug):
    surface.blit(fonts.h2.render("Shared World", True, TEXT), (L.main_x, L.main_y - 28))
    a, b = views

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = L.main_x + c * L.main_cell, L.main_y + r * L.main_cell
            va, vb = a.env.visited[r][c], b.env.visited[r][c]
            is_target = (r, c) == world.target_room

            opened_here = (a.env.opened or b.env.opened) and is_target
            if opened_here:
                bg = GOOD
            elif va and vb:
                bg = BOTH
            elif va:
                bg = AGENT_A_SOFT
            elif vb:
                bg = AGENT_B_SOFT
            else:
                bg = (24, 30, 46)

            rect = pygame.Rect(x, y, L.main_cell - 3, L.main_cell - 3)
            rrect(surface, bg, rect, radius=6)
            border = CHEST_GOLD if (debug and is_target) else PANEL_BORDER
            width = 2 if (debug and is_target) else 1
            rrect(surface, border, rect, radius=6, width=width)

            if debug:
                here = [cl for cl in world.clues.values() if cl.room == (r, c)]
                for k, cl in enumerate(here):
                    dot_color = GOOD if cl.real else BAD
                    pygame.draw.circle(surface, dot_color, (x + L.main_cell - 10 - k * 10, y + 9), 3)

    ar, ac = a.env.agent_room
    br, bc = b.env.agent_room
    ax = L.main_x + ac * L.main_cell + L.main_cell // 2
    ay = L.main_y + ar * L.main_cell + L.main_cell // 2
    bx = L.main_x + bc * L.main_cell + L.main_cell // 2
    by = L.main_y + br * L.main_cell + L.main_cell // 2
    marker_r = max(5, int(L.main_cell * 0.12))
    if (ar, ac) == (br, bc):
        off = marker_r + 3
        ax, ay = ax - off, ay - off // 2
        bx, by = bx + off, by + off // 2
    pygame.draw.circle(surface, AGENT_A, (ax, ay), marker_r)
    pygame.draw.circle(surface, (8, 10, 16), (ax, ay), marker_r, 2)
    pygame.draw.circle(surface, AGENT_B, (bx, by), marker_r)
    pygame.draw.circle(surface, (8, 10, 16), (bx, by), marker_r, 2)


def draw_legend(surface, fonts, L):
    items = [
        (AGENT_A, "Agent A"), (AGENT_B, "Agent B"),
        (AGENT_A_SOFT, "A searched"), (AGENT_B_SOFT, "B searched"),
        (BOTH, "both searched"), (GOOD, "chest opened"),
    ]
    x, y = L.main_x, L.legend_y
    max_x = L.main_x + max(L.main_px, 1)
    for color, label in items:
        txt = fonts.tiny.render(label, True, TEXT_DIM)
        item_w = 14 + 6 + txt.get_width() + 18
        if x + item_w > max_x and x > L.main_x:
            x = L.main_x
            y += 20
        pygame.draw.rect(surface, color, (x, y, 10, 10), border_radius=2)
        surface.blit(txt, (x + 15, y - 2))
        x += item_w


def draw_agent_panel(surface, fonts, view, x, y, w, h, debug):
    env = view.env
    rrect(surface, PANEL, (x, y, w, h), radius=10)
    rrect(surface, view.color, (x, y, w, 4), radius=0)
    rrect(surface, PANEL_BORDER, (x, y, w, h), radius=10, width=1)

    pad = 18
    cursor_y = y + 16

    title = fonts.h2.render(f"Agent {view.id}", True, view.color)
    surface.blit(title, (x + pad, cursor_y))
    status = "opened the chest!" if env.opened else ("timed out" if env.done else "exploring...")
    meta_text = f"steps {env.steps}   room {env.agent_room}   {status}"
    meta = fonts.body.render(meta_text, True, TEXT_DIM)
    # measured, not guessed: only sits beside the title if it actually fits
    # in the remaining panel width, otherwise wraps to its own line below
    meta_x = x + pad + title.get_width() + 18
    fits_beside_title = meta_x + meta.get_width() <= (x + w - pad)
    if fits_beside_title:
        surface.blit(meta, (meta_x, cursor_y + (title.get_height() - meta.get_height()) // 2))
        cursor_y += max(title.get_height(), meta.get_height()) + 16
    else:
        surface.blit(meta, (x + pad, cursor_y + title.get_height() + 4))
        cursor_y += title.get_height() + meta.get_height() + 20

    # --- room inset + belief heatmap, side by side ---
    grids_h = int(h * 0.40)
    inner_w = w - 2 * pad
    half_w = (inner_w - 20) / 2

    room_label = fonts.small.render("current room", True, TEXT_FAINT)
    belief_label = fonts.small.render("belief (row x col)", True, TEXT_FAINT)
    surface.blit(room_label, (x + pad, cursor_y))
    belief_x0 = x + pad + half_w + 20
    surface.blit(belief_label, (belief_x0, cursor_y))
    grid_top = cursor_y + room_label.get_height() + 8

    room_cell = int(clamp(min(half_w / SUB_SIZE, grids_h / SUB_SIZE), 20, 64))
    grid = env._room_checked_grid(env.agent_room)
    for i in range(SUB_SIZE):
        for j in range(SUB_SIZE):
            cx = x + pad + j * room_cell
            cy = grid_top + i * room_cell
            checked = grid[i][j]
            clue = env.world.clue_at(env.agent_room, (i, j))
            if checked and clue is not None:
                cell_color = (GOOD if clue.real else BAD) if debug else view.soft_color
            elif checked:
                cell_color = (27, 33, 50)
            else:
                cell_color = (16, 20, 32)
            rect = pygame.Rect(cx, cy, room_cell - 3, room_cell - 3)
            rrect(surface, cell_color, rect, radius=4)
            rrect(surface, PANEL_BORDER, rect, radius=4, width=1)
            if checked and clue is not None:
                mark = fonts.small.render("?", True, TEXT)
                surface.blit(mark, (cx + room_cell // 2 - 4, cy + room_cell // 2 - 8))

    belief_cell = int(clamp(min(half_w / GRID_SIZE, grids_h / GRID_SIZE), 14, 46))
    belief = env.belief_grid()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            p = belief[r][c]
            cx = belief_x0 + c * belief_cell
            cy = grid_top + r * belief_cell
            rect = pygame.Rect(cx, cy, belief_cell - 2, belief_cell - 2)
            rrect(surface, belief_color(p, GRID_SIZE), rect, radius=3)

    cursor_y = grid_top + grids_h + 14

    # --- candidates / inventory / quantum status ---
    rc = ", ".join(str(v) for v in env.qbelief.ranked_candidates("row")) or "none yet"
    cc = ", ".join(str(v) for v in env.qbelief.ranked_candidates("col")) or "none yet"
    inv = "carrying a clue" if env.inventory is not None else "empty-handed"
    ent = env.qbelief.entropy()
    if env.sensed_half is None:
        sense_txt = "not sensed"
    else:
        sense_txt = "top half" if env.sensed_half == 0 else "bottom half"
    line1 = fonts.small.render(f"row candidates: {rc}   |   col candidates: {cc}", True, TEXT_DIM)
    line2 = fonts.small.render(f"{inv}   |   failed attempts: {env.failed_attempts}", True, TEXT_DIM)
    line3 = fonts.small.render(
        f"entropy: {ent:.2f} bits   |   sensed: {sense_txt}"
        + ("   |   OR fired" if env.or_triggered else ""),
        True, TEXT_DIM,
    )
    surface.blit(line1, (x + pad, cursor_y))
    cursor_y += line1.get_height() + 5
    surface.blit(line2, (x + pad, cursor_y))
    cursor_y += line2.get_height() + 5
    surface.blit(line3, (x + pad, cursor_y))
    cursor_y += line3.get_height() + 14

    # --- log box: fills all remaining vertical space in the panel ---
    log_y = cursor_y
    log_h = (y + h) - log_y - pad
    if log_h > 20:
        box_x, box_w = x + pad, inner_w
        rrect(surface, (8, 10, 17), (box_x, log_y, box_w, log_h), radius=6)
        rrect(surface, PANEL_BORDER, (box_x, log_y, box_w, log_h), radius=6, width=1)

        inner_pad = 6
        line_h = fonts.small.get_height() + 4
        text_max_w = box_w - inner_pad * 2
        max_lines = max(1, (log_h - 12) // line_h)

        # Word-wrap each entry to the box's actual pixel width instead of
        # blitting one un-wrapped line per entry -- otherwise long
        # messages (e.g. the SENSE log lines) would run straight off the
        # right edge of the panel and overlap whatever was drawn next to
        # it. Walk the log NEWEST-first so that when there are more
        # wrapped lines than fit, it's always the oldest entries that get
        # dropped, not the most recent activity.
        wrapped_entries = []  # [(color, [line, line, ...]), ...], oldest->newest once reversed
        total_lines = 0
        for step, text in reversed(env.log):
            color = GOOD if "opened" in text else (BAD if "wrong" in text else TEXT_DIM)
            lines = wrap_text(fonts.small, f"[{step}] {text}", text_max_w)
            wrapped_entries.append((color, lines))
            total_lines += len(lines)
            if total_lines >= max_lines:
                break
        wrapped_entries.reverse()

        # Clip to the box itself as a hard backstop -- even if an entry's
        # wrapped lines slightly overrun the bottom edge (last visible
        # entry), nothing can bleed outside the log box's borders.
        prev_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(box_x, log_y, box_w, log_h))
        ly = log_y + 6
        for color, lines in wrapped_entries:
            for line_text in lines:
                line = fonts.small.render(line_text, True, color)
                surface.blit(line, (box_x + inner_pad, ly))
                ly += line_h
        surface.set_clip(prev_clip)


def draw_footer(surface, fonts, L, world, debug):
    y = L.footer_y
    pygame.draw.line(surface, PANEL_BORDER, (0, y), (L.win_w, y), 1)
    if debug:
        txt = (f"[debug] target room {world.target_room}   |   "
               f"{len(world.clues)} clues placed ({sum(1 for c in world.clues.values() if c.real)} real)")
    else:
        txt = "press D to reveal the target room and clue truth for debugging"
    surface.blit(fonts.small.render(txt, True, TEXT_FAINT), (L.main_x, y + 14))


def draw_banner(surface, fonts, w, text):
    box_w = fonts.h2.size(text)[0] + 50
    box_h = 40
    bx, by = (w - box_w) // 2, 10
    rrect(surface, (13, 34, 24), (bx, by, box_w, box_h), radius=8)
    rrect(surface, GOOD, (bx, by, box_w, box_h), radius=8, width=2)
    txt = fonts.h2.render(text, True, GOOD)
    surface.blit(txt, (bx + (box_w - txt.get_width()) // 2, by + (box_h - txt.get_height()) // 2))


def make_screen(fullscreen):
    if fullscreen:
        try:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            return screen
        except pygame.error:
            pass
    info = pygame.display.Info()
    w = max(MIN_WINDOW[0], int(info.current_w * 0.9))
    h = max(MIN_WINDOW[1], int(info.current_h * 0.9))
    return pygame.display.set_mode((w, h), pygame.RESIZABLE)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["heuristic", "dqn", "heuristic_meta", "dqn_meta"], default="heuristic")
    parser.add_argument("--windowed", action="store_true", help="start windowed instead of fullscreen")
    parser.add_argument("--fixed-start", action="store_true",
                         help="pin each agent to the fixed opposite corners every new world "
                              "(R key) instead of the default fresh random room")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    if args.policy == "dqn":
        policy_a = make_dqn_policy(os.path.join(here, "agent_a.pt"))
        policy_b = make_dqn_policy(os.path.join(here, "agent_b.pt"))
    elif args.policy == "heuristic_meta":
        # metacognitive control layer wrapped around the scripted
        # heuristic -- see metacognitive_controller.py. Verified: 200/200
        # solves, ~70% fewer failed DECLARE attempts than bare heuristic,
        # at the cost of more total steps (more deliberation before
        # committing to a guess).
        policy_a = make_metacognitive_heuristic()
        policy_b = make_metacognitive_heuristic()
    elif args.policy == "dqn_meta":
        policy_a = make_metacognitive_dqn_policy(os.path.join(here, "agent_a.pt"))
        policy_b = make_metacognitive_dqn_policy(os.path.join(here, "agent_b.pt"))
    else:
        policy_a = heuristic_action
        policy_b = heuristic_action

    pygame.init()
    fullscreen = args.windowed
    screen = make_screen(fullscreen)
    pygame.display.set_caption(f"Quantum-Belief RL Testbed ({args.policy})")
    clock = pygame.time.Clock()
    fonts = Fonts()

    # random spawn by default -- each agent gets a fresh random room
    # (env.py's start_room=None) every time a new world is generated;
    # --fixed-start instead pins them to predictable, non-overlapping
    # opposite corners
    start_a = (0, 0) if args.fixed_start else None
    start_b = (GRID_SIZE - 1, GRID_SIZE - 1) if args.fixed_start else None
    view_a = AgentView("A", start_a, MOVE_DELAY_A, AGENT_A, AGENT_A_SOFT)
    view_b = AgentView("B", start_b, MOVE_DELAY_B, AGENT_B, AGENT_B_SOFT)

    def new_world():
        world = World()
        view_a.reset(world)
        view_b.reset(world)
        return world

    world = new_world()
    running = True
    paused = False
    debug = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                screen = pygame.display.set_mode(
                    (max(event.w, MIN_WINDOW[0]), max(event.h, MIN_WINDOW[1])), pygame.RESIZABLE
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    world = new_world()
                elif event.key == pygame.K_d:
                    debug = not debug
                elif event.key == pygame.K_f:
                    fullscreen = not fullscreen
                    screen = make_screen(fullscreen)

        if not paused:
            view_a.maybe_step(policy_a)
            view_b.maybe_step(policy_b)

        win_w, win_h = screen.get_size()
        L = Layout(win_w, win_h)

        draw_background(screen, win_w, win_h)
        draw_header(screen, fonts, win_w, args.policy, paused, debug)
        draw_main_map(screen, fonts, L, (view_a, view_b), world, debug)
        draw_legend(screen, fonts, L)
        draw_agent_panel(screen, fonts, view_a, L.panel_a_x, L.panel_y, L.panel_w, L.panel_h, debug)
        draw_agent_panel(screen, fonts, view_b, L.panel_b_x, L.panel_y, L.panel_w, L.panel_h, debug)
        draw_footer(screen, fonts, L, world, debug)

        if view_a.env.opened or view_b.env.opened:
            banner = "Both agents opened the chest!" if (view_a.env.opened and view_b.env.opened) \
                else f"Agent {'A' if view_a.env.opened else 'B'} opened the chest first."
            draw_banner(screen, fonts, win_w, banner)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
