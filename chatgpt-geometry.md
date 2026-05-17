Use the tangent-cone angles from `grow_center` to the orbiting circle at the hit time:

[
\alpha = \operatorname{atan2}(C_y(t)-G_y,\ C_x(t)-G_x)
]

[
\delta = \arcsin\left(\frac{R}{|C(t)-G|}\right)
]

so the cone is:

[
[\alpha-\delta,\ \alpha+\delta]
]

Here’s the modified version returning:

```python
(t, angle_lo, angle_hi)
```

with angles normalized to `[0, 2π)`.

```python
import numpy as np


TAU = 2.0 * np.pi


def _norm_angle(a):
    return a % TAU


def orthogonal_circle_hit_time_short(
    orbit_center,          # (2,) centre of circular orbit, O
    orbit_radius,          # rho: radius of centre's orbit
    theta0,                # initial angle of orbiting centre
    omega,                 # angular velocity, rad / unit time
    moving_circle_radius,  # R: radius of orbiting circle itself
    grow_center,           # (2,) centre of growing circle, G
    grow_rate,             # v: growing circle radius = v*t
    horizon,               # T
    *,
    tol=1e-10,
    max_newton=4,
):
    """
    Returns:
        None
            if no orthogonal hit is found in [0, horizon]

        (t, angle_lo, angle_hi)
            where angle_lo and angle_hi define the tangent cone from grow_center
            to the orbiting circle at time t.

    Notes:
        The angular interval may wrap around 2π.
        Example: angle_lo=6.1, angle_hi=0.2 means [6.1, 2π) U [0, 0.2].
    """

    O = np.asarray(orbit_center, dtype=np.float64)
    G = np.asarray(grow_center, dtype=np.float64)

    dx, dy = O - G
    rho = float(orbit_radius)
    th0 = float(theta0)
    w = float(omega)
    R = float(moving_circle_radius)
    v = float(grow_rate)
    T = float(horizon)

    c0 = np.cos(th0)
    s0 = np.sin(th0)

    # d(t)^2 - R^2 - (v*t)^2
    base = dx * dx + dy * dy + rho * rho - R * R

    def f(t):
        th = th0 + w * t
        return base + 2.0 * rho * (dx * np.cos(th) + dy * np.sin(th)) - v * v * t * t

    def fp(t):
        th = th0 + w * t
        return 2.0 * rho * w * (-dx * np.sin(th) + dy * np.cos(th)) - 2.0 * v * v * t

    def output_at(t):
        th = th0 + w * t

        # Orbiting circle centre at time t.
        cx = O[0] + rho * np.cos(th)
        cy = O[1] + rho * np.sin(th)

        ux = cx - G[0]
        uy = cy - G[1]
        d = np.hypot(ux, uy)

        # If d < R, grow_center is inside the orbiting circle: no proper tangent cone.
        # In the orthogonal-hit case this should not normally happen.
        if d < R:
            return None

        centre_angle = np.arctan2(uy, ux)

        # Clamp for tiny numerical overshoots.
        half_angle = np.arcsin(np.clip(R / d, -1.0, 1.0))

        angle_lo = _norm_angle(centre_angle - half_angle)
        angle_hi = _norm_angle(centre_angle + half_angle)

        return float(t), float(angle_lo), float(angle_hi)

    f0 = base + 2.0 * rho * (dx * c0 + dy * s0)

    if abs(f0) <= tol:
        return output_at(0.0)

    fT = f(T)

    if abs(fT) <= tol:
        return output_at(T)

    if f0 * fT > 0.0:
        return None

    # Quadratic local approximation around t=0.
    f1 = 2.0 * rho * w * (-dx * s0 + dy * c0)
    f2 = -rho * w * w * (dx * c0 + dy * s0) - v * v

    t = None

    if abs(f2) < 1e-30:
        if abs(f1) > 1e-30:
            t_lin = -f0 / f1
            if 0.0 <= t_lin <= T:
                t = t_lin
    else:
        disc = f1 * f1 - 4.0 * f2 * f0
        if disc >= 0.0:
            sqrt_disc = np.sqrt(disc)
            r1 = (-f1 - sqrt_disc) / (2.0 * f2)
            r2 = (-f1 + sqrt_disc) / (2.0 * f2)

            candidates = [r for r in (r1, r2) if 0.0 <= r <= T]
            if candidates:
                t = min(candidates)

    if t is None:
        t = 0.5 * T

    lo, hi = 0.0, T
    flo, fhi = f0, fT

    for _ in range(max_newton):
        ft = f(t)

        if abs(ft) <= tol:
            return output_at(t)

        if flo * ft <= 0.0:
            hi, fhi = t, ft
        else:
            lo, flo = t, ft

        dft = fp(t)

        if abs(dft) > 1e-30:
            t_new = t - ft / dft
            if lo < t_new < hi:
                t = t_new
            else:
                t = 0.5 * (lo + hi)
        else:
            t = 0.5 * (lo + hi)

    for _ in range(20):
        t = 0.5 * (lo + hi)
        ft = f(t)

        if abs(ft) <= tol or (hi - lo) <= tol:
            return output_at(t)

        if flo * ft <= 0.0:
            hi, fhi = t, ft
        else:
            lo, flo = t, ft

    return output_at(0.5 * (lo + hi))
```

Usage:

```python
result = orthogonal_circle_hit_time_short(
    orbit_center=(0.0, 0.0),
    orbit_radius=10.0,
    theta0=0.3,
    omega=0.05,
    moving_circle_radius=1.0,
    grow_center=(5.0, 2.0),
    grow_rate=3.0,
    horizon=2.0,
)

if result is not None:
    t, a0, a1 = result
    print("hit time:", t)
    print("cone angles:", a0, a1)
```

At the orthogonal-hit time, those two cone boundary rays are exactly the rays from the growing circle’s centre to the two circle-intersection points.


also give me the special case where the 'orbiting' circle is actually stationary


For the stationary case there is no numerical solve. It is just:

[
d^2 = R^2 + v^2t^2
]

so

[
t = \frac{\sqrt{d^2-R^2}}{v}
]

where (d) is the distance from the growing circle’s centre to the stationary circle’s centre.

```python
import numpy as np


TAU = 2.0 * np.pi


def _norm_angle(a):
    return a % TAU


def orthogonal_circle_hit_time_stationary(
    circle_center,          # (2,) centre of stationary circle
    circle_radius,          # R
    grow_center,            # (2,) centre of growing circle
    grow_rate,              # v: growing radius = v*t
    horizon,                # T
    *,
    tol=1e-12,
):
    """
    Closed-form special case where the 'orbiting' circle is stationary.

    Returns:
        None
            if no orthogonal hit occurs in [0, horizon]

        (t, angle_lo, angle_hi)
            where angle_lo and angle_hi define the tangent cone from grow_center
            to the stationary circle.

    Notes:
        The angular interval may wrap around 2π.
        Example: angle_lo=6.1, angle_hi=0.2 means [6.1, 2π) U [0, 0.2].
    """

    C = np.asarray(circle_center, dtype=np.float64)
    G = np.asarray(grow_center, dtype=np.float64)

    R = float(circle_radius)
    v = float(grow_rate)
    T = float(horizon)

    ux = C[0] - G[0]
    uy = C[1] - G[1]
    d = np.hypot(ux, uy)

    # Need positive growth unless the answer is exactly t=0.
    if v <= 0.0:
        return None

    # Orthogonal condition:
    # d^2 = R^2 + (v*t)^2
    q = d * d - R * R

    if q < -tol:
        return None

    if abs(q) <= tol:
        t = 0.0
    else:
        t = np.sqrt(q) / v

    if t < -tol or t > T + tol:
        return None

    t = min(max(t, 0.0), T)

    # Proper tangent cone requires grow_center outside or on the circle.
    # q >= 0 already implies d >= R up to tolerance.
    if d <= 0.0:
        return None

    centre_angle = np.arctan2(uy, ux)

    # Half-angle of tangent cone from G to circle C,R.
    half_angle = np.arcsin(np.clip(R / d, -1.0, 1.0))

    angle_lo = _norm_angle(centre_angle - half_angle)
    angle_hi = _norm_angle(centre_angle + half_angle)

    return float(t), float(angle_lo), float(angle_hi)
```

Example:

```python
result = orthogonal_circle_hit_time_stationary(
    circle_center=(10.0, 0.0),
    circle_radius=2.0,
    grow_center=(0.0, 0.0),
    grow_rate=3.0,
    horizon=10.0,
)

print(result)
```

This is about as fast as it gets: one `hypot`, one `sqrt`, one `atan2`, one `asin`. No iteration.

If you want it in terms of the previous “orbiting” parameters with `omega=0`, compute the fixed centre as:

```python
def stationary_center_from_orbit_params(orbit_center, orbit_radius, theta0):
    O = np.asarray(orbit_center, dtype=np.float64)
    return O + orbit_radius * np.array([np.cos(theta0), np.sin(theta0)])
```

Then call:

```python
C = stationary_center_from_orbit_params(
    orbit_center=(0.0, 0.0),
    orbit_radius=10.0,
    theta0=0.3,
)

result = orthogonal_circle_hit_time_stationary(
    circle_center=C,
    circle_radius=1.0,
    grow_center=(5.0, 2.0),
    grow_rate=3.0,
    horizon=2.0,
)
```


and one where instead of orbiting the circle's motion is defined by a sequence of points and the circle moves linearly between the points advancing 1 point per 1 t


This one has a **closed-form per segment**. No Newton.

For segment (i), with (u=t-i\in[0,1]):

[
C(t)=P_i+u(P_{i+1}-P_i)
]

and the orthogonal condition is:

[
|C(t)-G|^2 = R^2 + v^2t^2
]

which becomes a quadratic in (u).

```python
import numpy as np


TAU = 2.0 * np.pi


def _norm_angle(a):
    return a % TAU


def _cone_from_point_to_circle(point, circle_center, circle_radius):
    """
    Tangent cone from `point` to circle.
    Returns (angle_lo, angle_hi), normalized to [0, 2π).

    The interval may wrap:
        lo=6.1, hi=0.2 means [6.1, 2π) U [0, 0.2].
    """
    P = np.asarray(point, dtype=np.float64)
    C = np.asarray(circle_center, dtype=np.float64)

    ux = C[0] - P[0]
    uy = C[1] - P[1]
    d = np.hypot(ux, uy)

    if d <= 0.0 or d < circle_radius:
        return None

    centre_angle = np.arctan2(uy, ux)
    half_angle = np.arcsin(np.clip(circle_radius / d, -1.0, 1.0))

    return (
        float(_norm_angle(centre_angle - half_angle)),
        float(_norm_angle(centre_angle + half_angle)),
    )


def _solve_quadratic_real(A, B, C, *, eps=1e-14):
    """
    Solve A*x^2 + B*x + C = 0.
    Returns a small list of real roots.
    """
    if abs(A) <= eps:
        if abs(B) <= eps:
            if abs(C) <= eps:
                return [0.0]  # Degenerate: whole interval is a solution.
            return []
        return [-C / B]

    disc = B * B - 4.0 * A * C

    if disc < -eps:
        return []

    if abs(disc) <= eps:
        return [-B / (2.0 * A)]

    sqrt_disc = np.sqrt(disc)

    # More stable quadratic formula.
    if B >= 0.0:
        q = -0.5 * (B + sqrt_disc)
    else:
        q = -0.5 * (B - sqrt_disc)

    if abs(q) <= eps:
        return [
            (-B - sqrt_disc) / (2.0 * A),
            (-B + sqrt_disc) / (2.0 * A),
        ]

    r1 = q / A
    r2 = C / q

    return [r1, r2]


def orthogonal_circle_hit_time_polyline(
    points,                 # (N, 2), circle centre is at points[i] at integer t=i
    circle_radius,          # R: radius of moving circle
    grow_center,            # (2,) centre of growing circle
    grow_rate,              # v: growing radius = v*t
    horizon,                # search t in [0, horizon]
    *,
    tol=1e-10,
    return_all=False,
):
    """
    Moving circle centre follows a polyline:
        t=0 -> points[0]
        t=1 -> points[1]
        t=2 -> points[2]
        ...

    Between integer times it moves linearly.

    Finds times where the moving circle and growing circle intersect orthogonally:

        ||C(t) - G||^2 = R^2 + (v*t)^2

    Returns:
        None
            if no hit found and return_all=False

        (t, angle_lo, angle_hi)
            first hit if return_all=False

        list[(t, angle_lo, angle_hi)]
            all hits if return_all=True

    angle_lo/angle_hi define the tangent cone from grow_center to the moving
    circle at the hit time.
    """

    pts = np.asarray(points, dtype=np.float64)
    G = np.asarray(grow_center, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if len(pts) < 2:
        raise ValueError("need at least two points for a polyline path")

    R = float(circle_radius)
    v = float(grow_rate)
    T = float(horizon)

    if T < 0.0:
        return [] if return_all else None

    # Path is defined only until t = len(points)-1.
    T = min(T, float(len(pts) - 1))

    if T < 0.0:
        return [] if return_all else None

    v2 = v * v
    R2 = R * R

    hits = []

    nseg = int(np.ceil(T))

    for i in range(nseg):
        seg_t0 = float(i)
        seg_t1 = min(float(i + 1), T)

        if seg_t1 < seg_t0:
            break

        u_lo = 0.0
        u_hi = seg_t1 - seg_t0

        P0 = pts[i]
        P1 = pts[i + 1]

        a = P0 - G
        b = P1 - P0

        # C(t) = P0 + u*b, where t = i + u
        #
        # |a + u*b|^2 - R^2 - v^2*(i + u)^2 = 0
        #
        # A*u^2 + B*u + C = 0
        A = np.dot(b, b) - v2
        B = 2.0 * np.dot(a, b) - 2.0 * v2 * seg_t0
        C = np.dot(a, a) - R2 - v2 * seg_t0 * seg_t0

        roots = _solve_quadratic_real(A, B, C)

        for u in roots:
            if u_lo - tol <= u <= u_hi + tol:
                u = min(max(float(u), u_lo), u_hi)
                t = seg_t0 + u

                # Avoid duplicate roots at segment boundaries.
                if hits and abs(t - hits[-1][0]) <= tol:
                    continue

                centre = P0 + u * b
                cone = _cone_from_point_to_circle(G, centre, R)

                if cone is None:
                    continue

                angle_lo, angle_hi = cone
                hit = (float(t), angle_lo, angle_hi)

                if not return_all:
                    return hit

                hits.append(hit)

    if return_all:
        return hits

    return None
```

Example:

```python
points = np.array([
    [10.0, 0.0],
    [10.0, 5.0],
    [7.0, 8.0],
    [3.0, 8.0],
])

result = orthogonal_circle_hit_time_polyline(
    points=points,
    circle_radius=1.5,
    grow_center=(0.0, 0.0),
    grow_rate=3.0,
    horizon=3.0,
)

print(result)
```

For a long path, this is (O(n)) over segments, but the per-segment work is tiny: a few dot products and a quadratic solve. If you only need the **first** hit, it exits as soon as one is found.
