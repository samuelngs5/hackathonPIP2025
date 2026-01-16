import numpy as np

DEFAULT_COSTS = {
    "unit_value": 5000,          # € / obus 155 mm
    "maintenance": 40000,        # € / maintenance
    "downtime_per_day": 200000,  # € / jour d'arrêt
    "wip_per_unit_day": 50       # € / unité WIP / jour
}

def evaluate_economic_metrics(run: dict) -> dict:
    """
    Évaluation économique a posteriori (approche consultant).
    Les coûts sont exprimés en euros.
    """

    # --- hypothèses chiffrées réalistes ---
    COST_PER_UNIT = 3000.0          # € / obus d'artillerie
    COST_DOWNTIME_DAY = 120_000.0   # € / jour d'arrêt ligne
    COST_MAINTENANCE = 80_000.0     # € / opération maintenance
    COST_WIP_UNIT = 500.0           # € / unité WIP / jour

    UNITS_TO_SHELLS = 1000  # facteur industriel
    PRICE_PER_SHELL = 3000  # € / obus (ordre de grandeur réaliste)

    produced_units = np.sum(run["throughput"])*UNITS_TO_SHELLS
    downtime_days = np.sum(run["downtime"])
    maintenance_ops = np.sum(run["maintenance"])
    avg_wip = np.mean(run["wip_true"])

    revenue = produced_units * COST_PER_UNIT
    downtime_cost = downtime_days * COST_DOWNTIME_DAY
    maintenance_cost = maintenance_ops * COST_MAINTENANCE
    congestion_cost = avg_wip * COST_WIP_UNIT * len(run["wip_true"])

    total_cost = downtime_cost + maintenance_cost + congestion_cost
    net_balance = revenue - total_cost

    return {
        "produced_units": produced_units,
        "revenue_eur": revenue,
        "downtime_cost_eur": downtime_cost,
        "maintenance_cost_eur": maintenance_cost,
        "congestion_cost_eur": congestion_cost,
        "total_cost_eur": total_cost,
        "net_balance_eur": net_balance,
    }