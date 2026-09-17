"""Storm claims adjuster allocation optimization demo.

Synthetic claims are assigned to field adjusters using a binary
mixed-integer optimization model that minimizes total geodesic distance.
"""

import random
from dataclasses import dataclass

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from geopy.distance import geodesic
from ortools.linear_solver import pywraplp


US_STATES = [
    ("AL", 32.806671, -86.791130), ("AZ", 33.729759, -111.431221),
    ("AR", 34.969704, -92.373123), ("CA", 36.116203, -119.681564),
    ("CO", 39.059811, -105.311104), ("CT", 41.597782, -72.755371),
    ("DE", 39.318523, -75.507141), ("FL", 27.766279, -81.686783),
    ("GA", 33.040619, -83.643074), ("ID", 44.240459, -114.478828),
    ("IL", 40.349457, -88.986137), ("IN", 39.849426, -86.258278),
    ("IA", 42.011539, -93.210526), ("KS", 38.526600, -96.726486),
    ("KY", 37.668140, -84.670067), ("LA", 31.169546, -91.867805),
    ("ME", 44.693947, -69.381927), ("MD", 39.063946, -76.802101),
    ("MA", 42.230171, -71.530106), ("MI", 43.326618, -84.536095),
    ("MN", 45.694454, -93.900192), ("MS", 32.741646, -89.678696),
    ("MO", 38.456085, -92.288368), ("MT", 46.921925, -110.454353),
    ("NE", 41.125370, -98.268082), ("NV", 38.313515, -117.055374),
    ("NH", 43.452492, -71.563896), ("NJ", 40.298904, -74.521011),
    ("NM", 34.840515, -106.248482), ("NY", 42.165726, -74.948051),
    ("NC", 35.630066, -79.806419), ("ND", 47.528912, -99.784012),
    ("OH", 40.388783, -82.764915), ("OK", 35.565342, -96.928917),
    ("OR", 44.572021, -122.070938), ("PA", 40.590752, -77.209755),
    ("RI", 41.680893, -71.511780), ("SC", 33.856892, -80.945007),
    ("SD", 44.299782, -99.438828), ("TN", 35.747845, -86.692345),
    ("TX", 31.054487, -97.563461), ("UT", 40.150032, -111.862434),
    ("VT", 44.045876, -72.710686), ("VA", 37.769337, -78.169968),
    ("WA", 47.400902, -121.490494), ("WV", 38.491226, -80.954453),
    ("WI", 44.268543, -89.616508), ("WY", 42.755966, -107.302490),
]

SLA_DISTANCE_KM = 500.0
SEED = 42


@dataclass
class OptimizationResult:
    claims: pd.DataFrame
    adjusters: pd.DataFrame
    assignments: pd.DataFrame
    capacity: int
    total_distance_km: float


def generate_scenario(storm_severity: int, num_adjusters: int, seed: int = SEED):
    """Create a reproducible synthetic storm scenario.

    Claims are concentrated around a randomly selected storm center. Nearby
    states are more likely to receive claims, producing a visible storm region
    rather than uniformly scattering claims across the country.
    """
    rng = random.Random(seed)
    n_claims = storm_severity * 10

    storm_center = rng.choice(US_STATES)
    center_coord = (storm_center[1], storm_center[2])

    # Give nearby states higher probability of receiving a claim.
    distances = np.array([
        geodesic(center_coord, (state[1], state[2])).km for state in US_STATES
    ])
    weights = np.exp(-distances / 900.0)
    weights = weights / weights.sum()

    claim_states = rng.choices(US_STATES, weights=weights, k=n_claims)
    adjuster_states = rng.sample(US_STATES, k=num_adjusters)

    claims = pd.DataFrame({
        "claim_id": range(n_claims),
        "state": [s[0] for s in claim_states],
        "lat": [s[1] for s in claim_states],
        "lon": [s[2] for s in claim_states],
    })
    adjusters = pd.DataFrame({
        "adjuster_id": range(num_adjusters),
        "state": [s[0] for s in adjuster_states],
        "lat": [s[1] for s in adjuster_states],
        "lon": [s[2] for s in adjuster_states],
    })

    return claims, adjusters, storm_center[0]


def build_distance_matrix(claims: pd.DataFrame, adjusters: pd.DataFrame) -> np.ndarray:
    """Calculate geodesic distance from every claim to every adjuster."""
    distances = np.zeros((len(claims), len(adjusters)))
    for i, claim in claims.iterrows():
        for j, adjuster in adjusters.iterrows():
            distances[i, j] = geodesic(
                (claim["lat"], claim["lon"]),
                (adjuster["lat"], adjuster["lon"]),
            ).km
    return distances


def solve_assignment(
    claims: pd.DataFrame,
    adjusters: pd.DataFrame,
    max_total_distance_km: float,
) -> OptimizationResult:
    """Solve the binary claim-to-adjuster assignment problem."""
    n_claims = len(claims)
    n_adjusters = len(adjusters)
    capacity = max(int(np.ceil(n_claims / n_adjusters)), 1)
    distances = build_distance_matrix(claims, adjusters)

    # CBC supports binary/integer decision variables.
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver is unavailable in this OR-Tools installation.")

    x = {
        (i, j): solver.BoolVar(f"x[{i},{j}]")
        for i in range(n_claims)
        for j in range(n_adjusters)
    }

    # Every claim must be assigned exactly once.
    for i in range(n_claims):
        solver.Add(sum(x[i, j] for j in range(n_adjusters)) == 1)

    # Each adjuster can handle at most `capacity` claims.
    for j in range(n_adjusters):
        solver.Add(sum(x[i, j] for i in range(n_claims)) <= capacity)

    total_distance = sum(
        distances[i, j] * x[i, j]
        for i in range(n_claims)
        for j in range(n_adjusters)
    )

    solver.Add(total_distance <= max_total_distance_km)
    solver.Minimize(total_distance)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError(
            "No feasible assignment was found. Increase the distance budget "
            "or change the scenario settings."
        )

    rows = []
    for i in range(n_claims):
        for j in range(n_adjusters):
            if x[i, j].solution_value() > 0.5:
                rows.append({
                    "claim_id": i,
                    "adjuster_id": j,
                    "distance_km": distances[i, j],
                    "within_sla": distances[i, j] <= SLA_DISTANCE_KM,
                })

    assignments = pd.DataFrame(rows)
    return OptimizationResult(
        claims=claims,
        adjusters=adjusters,
        assignments=assignments,
        capacity=capacity,
        total_distance_km=solver.Objective().Value(),
    )


def calculate_kpis(result: OptimizationResult) -> dict:
    """Return summary metrics for a solved assignment scenario."""
    assignments = result.assignments
    loads = assignments.groupby("adjuster_id").size().reindex(
        result.adjusters["adjuster_id"], fill_value=0
    )

    return {
        "total_distance_km": assignments["distance_km"].sum(),
        "avg_distance_km": assignments["distance_km"].mean(),
        "sla_rate": assignments["within_sla"].mean(),
        "avg_utilization": (loads / result.capacity).mean(),
    }


def create_map(result: OptimizationResult) -> folium.Map:
    """Visualize claims, adjusters, storm density, and assignments."""
    claims = result.claims
    adjusters = result.adjusters
    assignments = result.assignments

    m = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles="cartodbpositron",
    )

    heat_layer = folium.FeatureGroup(name="Claim density", show=True)
    HeatMap(
        claims[["lat", "lon"]].values.tolist(),
        radius=25,
        blur=15,
        max_zoom=6,
    ).add_to(heat_layer)
    heat_layer.add_to(m)

    claims_layer = folium.FeatureGroup(name="Claims", show=True)
    for _, row in claims.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=5,
            tooltip=f"Claim {int(row['claim_id'])} · {row['state']}",
            color="blue",
            fill=True,
            fill_opacity=0.8,
        ).add_to(claims_layer)
    claims_layer.add_to(m)

    adjusters_layer = folium.FeatureGroup(name="Adjusters", show=True)
    for _, row in adjusters.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]],
            tooltip=f"Adjuster {int(row['adjuster_id'])} · {row['state']}",
            icon=folium.Icon(color="red", icon="user"),
        ).add_to(adjusters_layer)
    adjusters_layer.add_to(m)

    assignment_layer = folium.FeatureGroup(name="Assignments", show=True)
    for _, assignment in assignments.iterrows():
        i = int(assignment["claim_id"])
        j = int(assignment["adjuster_id"])
        distance = float(assignment["distance_km"])
        line_color = "green" if assignment["within_sla"] else "red"

        folium.PolyLine(
            [
                (claims.loc[i, "lat"], claims.loc[i, "lon"]),
                (adjusters.loc[j, "lat"], adjusters.loc[j, "lon"]),
            ],
            color=line_color,
            weight=2,
            opacity=0.65,
            tooltip=f"{distance:,.0f} km",
        ).add_to(assignment_layer)
    assignment_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
