#!/usr/bin/env python3
"""
=====================================================
SYNTHETIC GITHUB ACTIONS DASHBOARD DATASET GENERATOR
=====================================================

Generates a synthetic dataset of GitHub-Actions-style dashboard screenshots
for training a CNN to classify CI/CD pipeline runs as SUCCESS or FAILURE.

Why synthetic generation: there is no public dataset of GitHub Actions
dashboard screenshots, and manually capturing thousands of real ones is not
practical. Instead, this script draws dashboard-like UIs programmatically
with Pillow, randomizing layout, text, colors, and visual noise so the
resulting images are diverse enough to train a CNN that generalizes.

This script does NOT use a browser (no Selenium). Every pixel is drawn
directly with PIL's ImageDraw API, which makes generation fast and fully
deterministic/scriptable.
"""

# =====================================================
# IMPORTS
# =====================================================
import os
import math
import random
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker

fake = Faker()


# =====================================================
# CONFIGURATION / CONSTANTS
# =====================================================
IMAGE_WIDTH = 1366
IMAGE_HEIGHT = 768

# Output goes into dataset/success and dataset/failure, one level up from
# this script's location (src/), matching the rest of the project layout.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "dataset")
SUCCESS_DIR = os.path.join(DATASET_DIR, "success")
FAILURE_DIR = os.path.join(DATASET_DIR, "failure")

TARGET_PER_CLASS = 2000

# Dark theme base palette, inspired by GitHub's dark dashboard UI.
BG_COLOR = (13, 17, 23)            # page background
PANEL_COLOR = (22, 27, 34)         # card / panel background
PANEL_BORDER = (48, 54, 61)        # subtle panel border
SIDEBAR_COLOR = (22, 27, 34)
SIDEBAR_ACTIVE = (33, 38, 45)
TEXT_PRIMARY = (201, 209, 217)
TEXT_SECONDARY = (139, 148, 158)
TEXT_MUTED = (96, 103, 112)
HEADER_COLOR = (22, 27, 34)
LOG_BG = (1, 4, 9)

# Class-specific accent colors.
GREEN = (63, 185, 80)
GREEN_DIM = (35, 95, 48)
RED = (248, 81, 73)
RED_DIM = (130, 45, 42)
YELLOW = (210, 153, 34)  # used for "running" nodes regardless of class

SUCCESS_WORDS = ["SUCCESS", "PASSED", "COMPLETED", "DEPLOYED"]
FAILURE_WORDS = ["FAILED", "ERROR", "CHECK FAILED", "BUILD FAILED"]

SIDEBAR_ITEMS = ["Summary", "Jobs", "Artifacts", "Usage", "Workflow file"]

# Fonts: DejaVu Sans Mono ships on most Linux systems and gives a clean,
# code-dashboard look. If it isn't found, we fall back to PIL's built-in
# default font so the script never crashes due to a missing font file.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def _load_font(size, bold=False):
    """
    Loads a TrueType font at the requested size, with a safe fallback.

    WHY: font availability differs across machines. We try the preferred
    monospace font first (it visually matches a developer dashboard), and
    fall back to PIL's default bitmap font if nothing else is available,
    so the script is portable across environments.
    """
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback: PIL's default font (fixed size, but never fails).
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow versions don't accept a size argument here.
        return ImageFont.load_default()


# =====================================================
# RANDOM CONFIG GENERATION
# =====================================================
def random_dashboard_config(label):
    """
    Builds a dictionary of ALL random variables for one dashboard image.

    WHY a single config dict: every draw_* function needs access to many
    shared random choices (e.g., sidebar width affects both draw_sidebar
    and draw_workflow_graph's horizontal offset). Generating everything up
    front in one place avoids passing dozens of separate arguments around
    and makes the randomization easy to audit/extend.

    label: "success" or "failure" — determines which color palette and
    vocabulary get used throughout the rest of the drawing functions.
    """
    config = {
        "label": label,

        # ---- Header content ----
        "repository": f"{fake.user_name()}/{fake.word()}-{fake.word()}",
        "workflow_title": random.choice([
            "CI", "Build and Test", "Deploy to Production", "Release Pipeline",
            "Continuous Integration", "Lint and Test", "Docker Build", "E2E Tests"
        ]),
        "run_number": random.randint(1, 9999),
        "commit_hash": hashlib.sha1(fake.uuid4().encode()).hexdigest()[:7],
        "branch": random.choice(["main", "develop", "release", "feature/" + fake.word(), "hotfix/" + fake.word()]),
        "username": fake.user_name(),
        "header_align": random.choice(["left", "center"]),

        # ---- Sidebar ----
        "sidebar_width": random.randint(190, 260),
        "active_sidebar_item": random.choice(SIDEBAR_ITEMS),

        # ---- Status card ----
        "duration_seconds": random.randint(8, 1800),
        "triggered_minutes_ago": random.randint(1, 720),
        "artifact_count": random.randint(0, 6),
        "status_icon_size": random.randint(28, 52),

        # ---- Workflow graph ----
        "node_count": random.randint(2, 10),
        "graph_spacing": random.randint(90, 160),

        # ---- Artifact panel ----
        "artifact_rows": random.randint(0, 5),

        # ---- Log panel ----
        "log_line_count": random.randint(6, 18),

        # ---- Layout randomization ----
        "font_scale": random.uniform(0.9, 1.15),
        "panel_width_ratio": random.uniform(0.62, 0.74),
        "padding": random.randint(12, 28),

        # ---- Visual distortion params (applied later in apply_effects) ----
        "apply_blur": random.random() < 0.35,
        "blur_radius": random.uniform(0.4, 1.6),
        "brightness_factor": random.uniform(0.85, 1.15),
        "contrast_factor": random.uniform(0.85, 1.15),
        "apply_noise": random.random() < 0.5,
        "noise_amount": random.uniform(2, 10),
        "apply_crop_zoom": random.random() < 0.4,
        "crop_fraction": random.uniform(0.02, 0.06),
        "apply_padding_shift": random.random() < 0.3,
        "shift_x": random.randint(-8, 8),
        "shift_y": random.randint(-8, 8),
    }

    # Colors depend on the class label.
    if label == "success":
        config["primary_color"] = GREEN
        config["primary_dim"] = GREEN_DIM
        config["status_word"] = random.choice(SUCCESS_WORDS)
    else:
        config["primary_color"] = RED
        config["primary_dim"] = RED_DIM
        config["status_word"] = random.choice(FAILURE_WORDS)

    return config


# =====================================================
# DRAW: HEADER
# =====================================================
def draw_header(draw, config):
    """
    Draws the top header bar: repository name, workflow title, run number,
    branch, commit hash, and username — mirroring a GitHub Actions run
    page's header area.
    """
    header_height = 64
    draw.rectangle([0, 0, IMAGE_WIDTH, header_height], fill=HEADER_COLOR)
    draw.line([0, header_height, IMAGE_WIDTH, header_height], fill=PANEL_BORDER, width=1)

    font_title = _load_font(int(18 * config["font_scale"]), bold=True)
    font_sub = _load_font(int(13 * config["font_scale"]))

    x = 24 if config["header_align"] == "left" else IMAGE_WIDTH // 2 - 300
    y = 10

    title_text = f"{config['repository']} — {config['workflow_title']} #{config['run_number']}"
    draw.text((x, y), title_text, font=font_title, fill=TEXT_PRIMARY)

    sub_text = (
        f"Branch: {config['branch']}    "
        f"Commit: {config['commit_hash']}    "
        f"By: {config['username']}"
    )
    draw.text((x, y + 26), sub_text, font=font_sub, fill=TEXT_SECONDARY)


# =====================================================
# DRAW: LEFT SIDEBAR
# =====================================================
def draw_sidebar(draw, config):
    """
    Draws the left navigation sidebar with Summary / Jobs / Artifacts /
    Usage / Workflow file entries, highlighting one randomly chosen item
    as the "active" selection (matching how GitHub highlights the current
    page in its sidebar nav).
    """
    header_height = 64
    sidebar_width = config["sidebar_width"]

    draw.rectangle(
        [0, header_height, sidebar_width, IMAGE_HEIGHT],
        fill=SIDEBAR_COLOR
    )
    draw.line(
        [sidebar_width, header_height, sidebar_width, IMAGE_HEIGHT],
        fill=PANEL_BORDER, width=1
    )

    font_item = _load_font(int(14 * config["font_scale"]))
    item_height = 38
    y = header_height + 16

    for item in SIDEBAR_ITEMS:
        is_active = item == config["active_sidebar_item"]
        if is_active:
            draw.rectangle(
                [4, y - 6, sidebar_width - 4, y + item_height - 12],
                fill=SIDEBAR_ACTIVE
            )
            # Left accent bar to mimic GitHub's active-tab indicator.
            draw.rectangle([0, y - 6, 3, y + item_height - 12], fill=config["primary_color"])
            text_color = TEXT_PRIMARY
        else:
            text_color = TEXT_SECONDARY

        draw.text((24, y), item, font=font_item, fill=text_color)
        y += item_height


# =====================================================
# DRAW: STATUS CARD
# =====================================================
def draw_status_card(draw, config, origin_x):
    """
    Draws the status summary card: overall status word (e.g. SUCCESS /
    FAILED), run duration, artifact count, and how long ago the run was
    triggered. This is the single most visually distinctive region for the
    CNN, since it carries the dominant color (green vs red) and icon shape.

    Returns the y-coordinate where the card ends, so subsequent sections
    can be placed below it without overlapping.
    """
    header_height = 64
    card_top = header_height + 20
    card_left = origin_x
    card_width = int(IMAGE_WIDTH * config["panel_width_ratio"])
    card_height = 110

    draw.rectangle(
        [card_left, card_top, card_left + card_width, card_top + card_height],
        fill=PANEL_COLOR, outline=PANEL_BORDER, width=1
    )

    icon_size = config["status_icon_size"]
    icon_cx = card_left + 40
    icon_cy = card_top + card_height // 2

    # Draw a circular badge filled with the class color, then a simple
    # checkmark (success) or X mark (failure) inside it.
    draw.ellipse(
        [icon_cx - icon_size // 2, icon_cy - icon_size // 2,
         icon_cx + icon_size // 2, icon_cy + icon_size // 2],
        fill=config["primary_dim"], outline=config["primary_color"], width=3
    )

    if config["label"] == "success":
        # Checkmark: two connected line segments.
        s = icon_size // 4
        draw.line(
            [(icon_cx - s, icon_cy), (icon_cx - s // 3, icon_cy + s),
             (icon_cx + s, icon_cy - s)],
            fill=config["primary_color"], width=4, joint="curve"
        )
    else:
        # X mark: two crossing diagonal lines.
        s = icon_size // 4
        draw.line([(icon_cx - s, icon_cy - s), (icon_cx + s, icon_cy + s)],
                   fill=config["primary_color"], width=4)
        draw.line([(icon_cx - s, icon_cy + s), (icon_cx + s, icon_cy - s)],
                   fill=config["primary_color"], width=4)

    font_status = _load_font(int(22 * config["font_scale"]), bold=True)
    font_meta = _load_font(int(13 * config["font_scale"]))

    text_x = icon_cx + icon_size
    draw.text((text_x, card_top + 18), config["status_word"], font=font_status, fill=config["primary_color"])

    minutes, seconds = divmod(config["duration_seconds"], 60)
    duration_str = f"{minutes}m {seconds}s"

    meta_text = (
        f"Duration: {duration_str}    "
        f"Artifacts: {config['artifact_count']}    "
        f"Triggered {config['triggered_minutes_ago']}m ago"
    )
    draw.text((text_x, card_top + 52), meta_text, font=font_meta, fill=TEXT_SECONDARY)

    return card_top + card_height


# =====================================================
# DRAW: WORKFLOW GRAPH
# =====================================================
def draw_workflow_graph(draw, config, origin_x, top_y):
    """
    Draws a simplified job graph: a row of nodes connected by lines,
    similar to GitHub Actions' workflow visualization. Each node is
    randomly assigned a type (success / running / failure), a random
    label, and a random duration, with randomized spacing between nodes.

    Returns the y-coordinate where the graph panel ends.
    """
    panel_left = origin_x
    panel_width = int(IMAGE_WIDTH * config["panel_width_ratio"])
    panel_top = top_y + 20
    panel_height = 220

    draw.rectangle(
        [panel_left, panel_top, panel_left + panel_width, panel_top + panel_height],
        fill=PANEL_COLOR, outline=PANEL_BORDER, width=1
    )

    font_label = _load_font(int(11 * config["font_scale"]))
    node_count = config["node_count"]
    spacing = config["graph_spacing"]

    # Distribute nodes left-to-right with the random spacing, then center
    # the whole row horizontally inside the panel.
    total_width = spacing * max(node_count - 1, 1)
    start_x = panel_left + (panel_width - total_width) // 2
    node_y = panel_top + panel_height // 2

    node_positions = []
    for i in range(node_count):
        # Add a small random vertical jitter so the graph doesn't look
        # perfectly mechanical.
        jitter_y = random.randint(-18, 18)
        x = start_x + i * spacing
        y = node_y + jitter_y
        node_positions.append((x, y))

    # Draw connecting lines first so nodes render on top of them.
    for i in range(len(node_positions) - 1):
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=PANEL_BORDER, width=2)

    job_name_pool = [
        "build", "test", "lint", "deploy", "package", "unit-tests",
        "integration-tests", "docker-build", "publish", "security-scan"
    ]

    for i, (x, y) in enumerate(node_positions):
        # The overall class biases node type distribution: success runs
        # mostly show success/running nodes, failure runs include at least
        # one explicit failure node (placed at a random position) so the
        # CNN sees a clear visual marker of breakage.
        if config["label"] == "failure" and i == node_count - 1:
            node_type = "failure"
        elif config["label"] == "failure":
            node_type = random.choices(
                ["success", "running", "failure"], weights=[0.45, 0.25, 0.30]
            )[0]
        else:
            node_type = random.choices(
                ["success", "running"], weights=[0.85, 0.15]
            )[0]

        if node_type == "success":
            node_color = GREEN
            node_fill = GREEN_DIM
        elif node_type == "failure":
            node_color = RED
            node_fill = RED_DIM
        else:
            node_color = YELLOW
            node_fill = (74, 56, 20)

        radius = 14
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=node_fill, outline=node_color, width=3
        )

        label = random.choice(job_name_pool)
        duration = random.randint(1, 600)
        label_text = f"{label}"
        duration_text = f"{duration}s"

        draw.text((x - radius, y + radius + 6), label_text, font=font_label, fill=TEXT_SECONDARY)
        draw.text((x - radius, y + radius + 20), duration_text, font=font_label, fill=TEXT_MUTED)

    return panel_top + panel_height


# =====================================================
# DRAW: ARTIFACT PANEL
# =====================================================
def draw_artifact_panel(draw, config, origin_x, top_y):
    """
    Draws a small panel listing build artifacts (random file names and
    random sizes), similar to the "Artifacts" section of a GitHub Actions
    run summary.

    Returns the y-coordinate where this panel ends.
    """
    panel_left = origin_x
    panel_width = int(IMAGE_WIDTH * config["panel_width_ratio"])
    panel_top = top_y + 20
    row_height = 30
    rows = config["artifact_rows"]
    panel_height = 40 + row_height * max(rows, 1)

    draw.rectangle(
        [panel_left, panel_top, panel_left + panel_width, panel_top + panel_height],
        fill=PANEL_COLOR, outline=PANEL_BORDER, width=1
    )

    font_header = _load_font(int(13 * config["font_scale"]), bold=True)
    font_row = _load_font(int(12 * config["font_scale"]))

    draw.text((panel_left + 16, panel_top + 10), "Artifacts", font=font_header, fill=TEXT_PRIMARY)

    extensions = [".zip", ".tar.gz", ".log", ".json", ".whl"]
    y = panel_top + 38
    for _ in range(rows):
        name = f"{fake.word()}-artifact{random.choice(extensions)}"
        size_mb = round(random.uniform(0.1, 250.0), 1)
        draw.text((panel_left + 24, y), name, font=font_row, fill=TEXT_SECONDARY)
        size_text = f"{size_mb} MB"
        draw.text((panel_left + panel_width - 90, y), size_text, font=font_row, fill=TEXT_MUTED)
        y += row_height

    if rows == 0:
        draw.text((panel_left + 16, panel_top + 38), "No artifacts produced", font=font_row, fill=TEXT_MUTED)

    return panel_top + panel_height


# =====================================================
# DRAW: LOG PANEL
# =====================================================
def draw_log_panel(draw, config, origin_x, top_y):
    """
    Draws a terminal-style log panel with random timestamped lines. The
    vocabulary of log lines differs between classes: success logs describe
    normal completion steps, while failure logs include error/traceback
    style lines, giving the CNN additional class-distinguishing texture
    even at the pixel level (mostly green-ish text vs red-flagged lines).
    """
    panel_left = origin_x
    panel_width = int(IMAGE_WIDTH * config["panel_width_ratio"])
    panel_top = top_y + 20
    panel_bottom = IMAGE_HEIGHT - 24
    panel_height = max(panel_bottom - panel_top, 80)

    draw.rectangle(
        [panel_left, panel_top, panel_left + panel_width, panel_top + panel_height],
        fill=LOG_BG, outline=PANEL_BORDER, width=1
    )

    font_log = _load_font(int(11.5 * config["font_scale"]))

    success_log_templates = [
        "Step completed successfully",
        "All tests passed",
        "Build finished with exit code 0",
        "Cache restored successfully",
        "Uploading artifact... done",
        "Deployment succeeded",
        "Linting passed with no warnings",
        "Container pushed to registry",
    ]
    failure_log_templates = [
        "Error: process completed with exit code 1",
        "Traceback (most recent call last):",
        "AssertionError: expected 200 but got 500",
        "Build failed: missing dependency",
        "##[error] Step failed",
        "Connection timed out after 30s",
        "Test suite failed: 3 failing, 12 passing",
        "FATAL: could not push to remote",
    ]

    templates = success_log_templates if config["label"] == "success" else failure_log_templates
    line_count = config["log_line_count"]

    y = panel_top + 10
    line_height = (panel_height - 20) / max(line_count, 1)

    for i in range(line_count):
        timestamp = fake.time(pattern="%H:%M:%S")
        line_text = random.choice(templates)

        # Color most lines as neutral log text, but flag a subset with the
        # class accent color to mimic highlighted error/success lines.
        if random.random() < 0.25:
            color = config["primary_color"]
        else:
            color = TEXT_SECONDARY

        draw.text((panel_left + 14, y), f"[{timestamp}] {line_text}", font=font_log, fill=color)
        y += line_height


# =====================================================
# CREATE FULL DASHBOARD (COMBINES ALL SECTIONS)
# =====================================================
def create_dashboard(config):
    """
    Builds one full dashboard image from a random config dictionary by
    calling each section's draw function in order and compositing them
    onto a single dark-themed canvas.
    """
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(image)

    draw_header(draw, config)
    draw_sidebar(draw, config)

    content_origin_x = config["sidebar_width"] + config["padding"] * 2

    bottom_of_status = draw_status_card(draw, config, content_origin_x)
    bottom_of_graph = draw_workflow_graph(draw, config, content_origin_x, bottom_of_status)
    bottom_of_artifacts = draw_artifact_panel(draw, config, content_origin_x, bottom_of_graph)
    draw_log_panel(draw, config, content_origin_x, bottom_of_artifacts)

    return image


# =====================================================
# APPLY VISUAL EFFECTS / DISTORTIONS
# =====================================================
def apply_effects(image, config):
    """
    Applies a randomized combination of visual distortions to make the
    dataset more robust and visually diverse, mimicking real-world
    screenshot variability (different monitors, scaling, compression,
    screenshot tools, etc.).

    Applied (each independently randomized on/off per image):
        - Gaussian blur
        - Brightness adjustment
        - Contrast adjustment
        - Random noise
        - Crop + resize (simulates a "zoom"/different capture region)
        - Small padding/shift (simulates the dashboard being captured
          slightly off-center)

    Never applies rotation or mirroring, since a real dashboard screenshot
    is never rotated or flipped — preserving this constraint keeps the
    synthetic data realistic.
    """
    from PIL import ImageEnhance

    # ---- Gaussian blur ----
    if config["apply_blur"]:
        image = image.filter(ImageFilter.GaussianBlur(radius=config["blur_radius"]))

    # ---- Brightness ----
    image = ImageEnhance.Brightness(image).enhance(config["brightness_factor"])

    # ---- Contrast ----
    image = ImageEnhance.Contrast(image).enhance(config["contrast_factor"])

    # ---- Crop + resize (simulated zoom) ----
    # Cropping a small border and resizing back to the original resolution
    # simulates a slightly different capture region/zoom level without
    # ever rotating or mirroring the content.
    if config["apply_crop_zoom"]:
        crop_px_x = int(IMAGE_WIDTH * config["crop_fraction"])
        crop_px_y = int(IMAGE_HEIGHT * config["crop_fraction"])
        box = (crop_px_x, crop_px_y, IMAGE_WIDTH - crop_px_x, IMAGE_HEIGHT - crop_px_y)
        image = image.crop(box).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    # ---- Minor padding/shift ----
    # Shifts the whole canvas by a few pixels by pasting it onto a new
    # background-colored canvas at an offset, simulating slight capture
    # misalignment. The exposed edge is filled with the background color
    # rather than wrapping, since a real screenshot would not wrap around.
    if config["apply_padding_shift"]:
        shifted = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=BG_COLOR)
        shifted.paste(image, (config["shift_x"], config["shift_y"]))
        image = shifted

    # ---- Random noise ----
    if config["apply_noise"]:
        np_image = np.array(image).astype(np.float32)
        noise = np.random.normal(0, config["noise_amount"], np_image.shape)
        np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(np_image, mode="RGB")

    return image


# =====================================================
# COMPUTE HASH (FOR UNIQUENESS CHECK)
# =====================================================
def compute_hash(image):
    """
    Computes a SHA256 hash of an image's raw pixel bytes.

    WHY: this gives us a fast, reliable fingerprint to detect duplicate
    (or near-byte-identical) images so we never save two visually
    indistinguishable screenshots into the dataset.
    """
    raw_bytes = image.tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


# =====================================================
# SAVE UNIQUE IMAGE
# =====================================================
def save_unique(image, save_path, seen_hashes):
    """
    Saves an image only if its hash has not been seen before in this run.

    Returns True if the image was saved (i.e., it was unique), or False if
    it was a duplicate and was discarded.

    WHY in-memory hash storage is enough here: we only need uniqueness
    within a single generation run (this script's lifetime), not across
    multiple runs/days, so a Python set is fast and sufficient — no
    database or on-disk index is needed.
    """
    image_hash = compute_hash(image)

    if image_hash in seen_hashes:
        return False

    seen_hashes.add(image_hash)
    image.save(save_path, format="PNG")
    return True


# =====================================================
# GENERATE ONE FULL CLASS (SUCCESS OR FAILURE)
# =====================================================
def generate_class(label, output_dir, target_count):
    """
    Generates `target_count` unique images for one class ("success" or
    "failure"), retrying whenever a duplicate is produced, and printing
    progress every 100 saved images.
    """
    os.makedirs(output_dir, exist_ok=True)

    seen_hashes = set()
    saved_count = 0
    attempt_count = 0

    # Safety valve: if duplicates somehow dominate (extremely unlikely
    # given how many random variables feed into each image), we cap total
    # attempts so the script can never loop forever.
    max_attempts = target_count * 20

    while saved_count < target_count and attempt_count < max_attempts:
        attempt_count += 1

        config = random_dashboard_config(label)
        image = create_dashboard(config)
        image = apply_effects(image, config)

        file_name = f"{label}_{saved_count + 1:04d}.png"
        save_path = os.path.join(output_dir, file_name)

        was_saved = save_unique(image, save_path, seen_hashes)

        if was_saved:
            saved_count += 1
            if saved_count % 100 == 0:
                print(f"Generated {saved_count} / {target_count}")

    if saved_count < target_count:
        print(
            f"Warning: only generated {saved_count}/{target_count} unique "
            f"'{label}' images after {attempt_count} attempts."
        )

    return saved_count


# =====================================================
# MAIN ENTRY POINT
# =====================================================
def main():
    print("=" * 55)
    print("GENERATING SYNTHETIC GITHUB ACTIONS DASHBOARD DATASET")
    print("=" * 55)

    print(f"\nOutput directory: {os.path.abspath(DATASET_DIR)}")
    print(f"Target: {TARGET_PER_CLASS} success images, {TARGET_PER_CLASS} failure images\n")

    print("-" * 55)
    print("Generating SUCCESS class...")
    print("-" * 55)
    success_count = generate_class("success", SUCCESS_DIR, TARGET_PER_CLASS)

    print("\n" + "-" * 55)
    print("Generating FAILURE class...")
    print("-" * 55)
    failure_count = generate_class("failure", FAILURE_DIR, TARGET_PER_CLASS)

    print("\n" + "=" * 55)
    print("Generation Complete")
    print(f"Success Images: {success_count}")
    print(f"Failure Images: {failure_count}")
    print("=" * 55)


if __name__ == "__main__":
    main()