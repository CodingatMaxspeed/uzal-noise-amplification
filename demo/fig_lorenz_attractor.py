"""The Lorenz attractor as a plain line drawing, to illustrate the text.

An illustration, not a measurement: no axes, no labels, no annotation, nothing
to read off. The shape alone has to carry it. Matched to the reference image
figures/lorenz_attraktor.png.

Parameters are Lorenz's own canonical set, sigma = 10, rho = 28, beta = 8/3 --
the ones every treatment of the system uses, Casdagli's included. They are not
free: Lorenz chaos needs rho > sigma(sigma + beta + 3)/(sigma - beta - 1),
which is 24.74 here, and raising beta to widen the attractor raises that bound
faster than it widens anything. The proportions are set by the view, not by
the parameters.

Three choices make the picture, and all three are about what gets drawn rather
than about the system:

The start sits just off the unstable fixed point C- of the left lobe, and the
transient is kept rather than discarded. That is what produces the fine spiral
winding out of the centre: the trajectory leaves C- along its unstable
eigendirection, circling outward at e^(0.094 t), so it takes about 90 time
units to grow from OFFSET to the size of the lobe. Starting on the attractor
instead leaves a bare hole there -- the fixed point repels, so a long run never
fills its neighbourhood in.

The view is tilted. Both wings are open and symmetric only seen along the y
axis, and that is exactly the view in which the attractor is taller than wide:
x spans about 35 units, z about 42. Tilting trades a little symmetry for a
much wider, properly unfolded figure, and costs nothing in honesty -- the same
object, seen from elsewhere.

The rotation is done here rather than with a 3-D axes, which cannot crop to
its content: an mplot3d figure saves as a square with the drawing adrift in
the middle. Rotating the trajectory and plotting in two dimensions gives the
same picture, one path, exact margins, and an equal aspect that cannot
silently stretch the shape.

Black on transparent. Writes a PDF for LaTeX (vector, scales without
resampling) and a PNG to look at.

Takes a few seconds. Run: python demo/fig_lorenz_attractor.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The private integrator, not the public lorenz(), which returns the x column
# alone -- a drawing needs all three.
from uzal.systems import _rk4

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
OUT = FIGURES / "fig_lorenz_attractor.pdf"

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT, T_DRAW = 0.0025, 150.0
OFFSET = 0.8               # start this far off C-; smaller closes the centre
ELEV, AZIM = 25.0, -45.0   # degrees; ELEV = 0, AZIM = -90 is the flat (x, z)
Z_SQUASH = 0.8             # z shortened before the tilt, to open the wings
# The grey of the reference comes from many near-transparent strokes. Carried
# by alpha rather than by width: below about 0.15 pt a stroke is a hairline,
# which some print paths render unreliably or drop. Raise LW to 0.25 and
# ALPHA to 0.5 if the figure is printed below 8 cm wide.
LW, ALPHA = 0.20, 0.60


def main():
    def deriv(s):
        x, y, z = s
        return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])

    # C- = (-sqrt(beta(rho-1)), -sqrt(beta(rho-1)), rho-1), the unstable focus
    # of the left lobe. C+ would put the spiral in the right lobe instead.
    c_minus = np.array([-np.sqrt(BETA * (RHO - 1)), -np.sqrt(BETA * (RHO - 1)),
                        RHO - 1.0])
    traj = _rk4(deriv, c_minus + [OFFSET, 0.0, 0.0], DT, int(T_DRAW / DT))
    traj = (traj - traj.mean(axis=0)) * [1.0, 1.0, Z_SQUASH]

    # The two screen axes of a camera at (ELEV, AZIM): `right` is horizontal,
    # `up` is what is left of the vertical after the tilt. Both unit length and
    # orthogonal, so this is a rotation -- no stretch enters the picture.
    a, e = np.radians(AZIM), np.radians(ELEV)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.cos(a) * np.sin(e), -np.sin(a) * np.sin(e), np.cos(e)])
    u, v = traj @ right, traj @ up

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    # Thin and slightly transparent: where the orbit crowds, the line darkens
    # on its own, and that density is the only depth cue a line drawing has.
    ax.plot(u, v, color="black", lw=LW, alpha=ALPHA, solid_joinstyle="round")
    ax.set_aspect("equal")      # the shape is the content; do not stretch it
    ax.set_axis_off()

    FIGURES.mkdir(exist_ok=True)
    for path, kw in ((OUT, {}), (OUT.with_suffix(".png"), {"dpi": 600})):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                    transparent=True, **kw)
        print(f"wrote: {path}")
    print(f"aspect w/h = {(u.max()-u.min())/(v.max()-v.min()):.2f}")


if __name__ == "__main__":
    main()
