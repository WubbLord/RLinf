#!/usr/bin/env python3
"""Generate RLinf result figures without external plotting dependencies."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
EVAL_ROOT = REPO_ROOT / "logs" / "eval"

SERIES = [
    ("Env-reward GRPO", "envreward", (0.10, 0.34, 0.62)),
    ("RoboReward GRPO", "roboreward", (0.78, 0.25, 0.18)),
]
EPOCHS = [100, 200, 300, 400, 500]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def scalar(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def first_scalar(*values):
    for value in values:
        value = scalar(value)
        if value is not None:
            return value
    return None


def candidate_dirs(model_key: str, epoch: int) -> list[Path]:
    if epoch == 500:
        return sorted(
            EVAL_ROOT.glob(f"*{model_key}-gs{epoch}-roboreward_rollout"),
            reverse=True,
        )
    return sorted(EVAL_ROOT.glob(f"{model_key}-gs{epoch}-eval-*"), reverse=True)


def extract_record(model_name: str, model_key: str, epoch: int) -> dict | None:
    for log_dir in candidate_dirs(model_key, epoch):
        root_summary = read_json(log_dir / "summary.json")
        rr_summary = read_json(log_dir / "roboreward_outputs" / "summary.json")
        per_rollout = read_json(
            log_dir / "roboreward_outputs" / "per_rollout_rewards_summary.json"
        )

        success_once = first_scalar(
            root_summary.get("eval/success_once"),
            root_summary.get("roboreward_rollout/success_once"),
            rr_summary.get("eval/success_once"),
            rr_summary.get("roboreward_rollout/success_once"),
            per_rollout.get("success_rate"),
        )
        roboreward_avg = first_scalar(
            per_rollout.get("mean_roboreward_terminal_sum_normalized"),
            root_summary.get("roboreward_rollout/roboreward_terminal"),
            rr_summary.get("roboreward_rollout/roboreward_terminal"),
            root_summary.get("roboreward_rollout/reward_model_output"),
            rr_summary.get("roboreward_rollout/reward_model_output"),
        )
        rollouts = first_scalar(
            root_summary.get("eval/num_trajectories"),
            root_summary.get("roboreward_rollout/num_trajectories"),
            rr_summary.get("eval/num_trajectories"),
            rr_summary.get("roboreward_rollout/num_trajectories"),
            per_rollout.get("num_rollout_windows"),
        )

        if success_once is None or roboreward_avg is None:
            continue
        return {
            "model": model_name,
            "model_key": model_key,
            "epoch": epoch,
            "success_once": float(success_once),
            "roboreward_avg": float(roboreward_avg),
            "rollouts": int(rollouts) if rollouts is not None else "",
            "log_dir": str(log_dir.relative_to(REPO_ROOT)),
        }
    return None


def load_records() -> list[dict]:
    records = []
    for model_name, model_key, _color in SERIES:
        for epoch in EPOCHS:
            record = extract_record(model_name, model_key, epoch)
            if record is not None:
                records.append(record)
    return records


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Canvas:
    def __init__(self, width: int = 720, height: int = 480):
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def stroke_color(self, color):
        self.ops.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG")

    def fill_color(self, color):
        self.ops.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")

    def line_width(self, width: float):
        self.ops.append(f"{width:.2f} w")

    def line_style(self, cap: int = 0, join: int = 0):
        self.ops.append(f"{cap} J {join} j")

    def line(self, x1, y1, x2, y2):
        self.ops.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def polyline(self, points):
        if not points:
            return
        parts = [f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
        parts.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        self.ops.append(" ".join(parts) + " S")

    def rect_fill(self, x, y, w, h):
        self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def rect_stroke(self, x, y, w, h):
        self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re S")

    def circle_fill(self, x, y, r):
        k = 0.5522847498 * r
        self.ops.append(
            " ".join(
                [
                    f"{x:.2f} {y + r:.2f} m",
                    f"{x + k:.2f} {y + r:.2f} {x + r:.2f} {y + k:.2f} {x + r:.2f} {y:.2f} c",
                    f"{x + r:.2f} {y - k:.2f} {x + k:.2f} {y - r:.2f} {x:.2f} {y - r:.2f} c",
                    f"{x - k:.2f} {y - r:.2f} {x - r:.2f} {y - k:.2f} {x - r:.2f} {y:.2f} c",
                    f"{x - r:.2f} {y + k:.2f} {x - k:.2f} {y + r:.2f} {x:.2f} {y + r:.2f} c",
                    "f",
                ]
            )
        )

    def text(self, x, y, text, size=10, anchor="left", color=(0, 0, 0), angle=0):
        approx_width = len(text) * size * 0.52
        x_offset = 0.0
        if anchor == "center":
            x_offset = -approx_width / 2
        elif anchor == "right":
            x_offset = -approx_width
        self.fill_color(color)
        if angle:
            radians = math.radians(angle)
            c = math.cos(radians)
            s = math.sin(radians)
            self.ops.append(
                f"BT /F1 {size:.1f} Tf {c:.5f} {s:.5f} {-s:.5f} {c:.5f} "
                f"{x:.2f} {y:.2f} Tm {x_offset:.2f} 0 Td "
                f"({pdf_escape(text)}) Tj ET"
            )
        else:
            self.ops.append(
                f"BT /F1 {size:.1f} Tf {x + x_offset:.2f} {y:.2f} Td "
                f"({pdf_escape(text)}) Tj ET"
            )

    def stream(self) -> str:
        return "\n".join(self.ops) + "\n"


def write_pdf(path: Path, canvas: Canvas):
    stream = canvas.stream().encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {canvas.width} {canvas.height}] "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    startxref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{startxref}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(pdf)


def plot_metric(records: list[dict], metric: str, y_label: str, title: str, output: Path):
    canvas = Canvas()
    left, right, bottom, top = 104, 44, 72, 72
    plot_w = canvas.width - left - right
    plot_h = canvas.height - bottom - top
    x_min, x_max = 100, 500
    y_values = [r[metric] for r in records]
    y_max = max(y_values) if y_values else 1.0
    if metric == "success_once":
        y_top = max(0.8, math.ceil(y_max * 10) / 10)
        y_step = 0.1
    else:
        y_top = max(0.35, math.ceil(y_max * 20) / 20)
        y_step = 0.05
    y_min = 0.0

    def sx(epoch):
        return left + (epoch - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        return bottom + (value - y_min) / (y_top - y_min) * plot_h

    canvas.fill_color((1, 1, 1))
    canvas.rect_fill(0, 0, canvas.width, canvas.height)
    canvas.fill_color((0.975, 0.980, 0.985))
    canvas.rect_fill(left, bottom, plot_w, plot_h)

    canvas.text(canvas.width / 2, canvas.height - 36, title, size=17, anchor="center")
    canvas.text(
        canvas.width / 2,
        canvas.height - 55,
        "Completed 512-rollout evals only",
        size=10,
        anchor="center",
        color=(0.35, 0.35, 0.35),
    )
    canvas.text(canvas.width / 2, 28, "Training epoch", size=12, anchor="center")
    canvas.text(
        32,
        bottom + plot_h / 2,
        y_label,
        size=12,
        anchor="center",
        color=(0.08, 0.08, 0.08),
        angle=90,
    )

    canvas.stroke_color((0.82, 0.82, 0.82))
    canvas.line_width(0.50)
    tick = 0.0
    while tick <= y_top + 1e-9:
        y = sy(tick)
        canvas.line(left, y, left + plot_w, y)
        tick_label = f"{tick:.2f}".rstrip("0").rstrip(".")
        canvas.text(
            left - 12,
            y - 3,
            tick_label,
            size=9,
            anchor="right",
            color=(0.22, 0.22, 0.22),
        )
        tick += y_step

    for epoch in [100, 200, 300, 400, 500]:
        x = sx(epoch)
        canvas.stroke_color((0.89, 0.89, 0.89))
        canvas.line_width(0.40)
        canvas.line(x, bottom, x, bottom + plot_h)
        canvas.text(
            x,
            bottom - 21,
            str(epoch),
            size=10,
            anchor="center",
            color=(0.22, 0.22, 0.22),
        )

    canvas.stroke_color((0.12, 0.12, 0.12))
    canvas.line_width(1.1)
    canvas.line(left, bottom, left + plot_w, bottom)
    canvas.line(left, bottom, left, bottom + plot_h)

    legend_w, legend_h = 178, 50
    legend_x = left + plot_w - legend_w - 14
    legend_y = bottom + plot_h - legend_h - 14
    canvas.fill_color((1, 1, 1))
    canvas.rect_fill(legend_x, legend_y, legend_w, legend_h)
    canvas.stroke_color((0.80, 0.80, 0.80))
    canvas.line_width(0.7)
    canvas.rect_stroke(legend_x, legend_y, legend_w, legend_h)

    canvas.line_style(cap=1, join=1)
    for idx, (model_name, model_key, color) in enumerate(SERIES):
        series = sorted(
            [r for r in records if r["model_key"] == model_key],
            key=lambda r: r["epoch"],
        )
        points = [(sx(r["epoch"]), sy(r[metric])) for r in series]
        canvas.stroke_color(color)
        canvas.fill_color(color)
        canvas.line_width(2.4)
        canvas.polyline(points)
        for x, y in points:
            canvas.circle_fill(x, y, 4.2)
        ly = legend_y + legend_h - 19 - idx * 20
        canvas.line(legend_x + 12, ly + 3, legend_x + 38, ly + 3)
        canvas.circle_fill(legend_x + 25, ly + 3, 4.0)
        canvas.text(legend_x + 48, ly, model_name, size=10, color=(0.10, 0.10, 0.10))

    write_pdf(output, canvas)


def write_data_csv(records: list[dict]):
    path = FIGURE_DIR / "figure_data.csv"
    with path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "epoch",
            "success_once",
            "roboreward_avg",
            "rollouts",
            "log_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r["model"], r["epoch"])):
            writer.writerow({name: record[name] for name in fieldnames})


def main():
    records = load_records()
    if not records:
        raise SystemExit("No completed eval summaries found.")

    write_data_csv(records)
    plot_metric(
        records,
        metric="success_once",
        y_label="Success once",
        title="Success Once vs Epoch",
        output=FIGURE_DIR / "success_once_vs_epoch.pdf",
    )
    plot_metric(
        records,
        metric="roboreward_avg",
        y_label="Average RoboReward",
        title="Average RoboReward vs Epoch",
        output=FIGURE_DIR / "average_roboreward_vs_epoch.pdf",
    )

    print("Wrote:")
    print(FIGURE_DIR / "figure_data.csv")
    print(FIGURE_DIR / "success_once_vs_epoch.pdf")
    print(FIGURE_DIR / "average_roboreward_vs_epoch.pdf")
    print(f"Included {len(records)} completed eval points.")


if __name__ == "__main__":
    main()
