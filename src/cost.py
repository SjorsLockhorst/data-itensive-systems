from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf


TOTAL_COST = 1000
WEIGHT = 0.2

@udf(FloatType())
def calc_payment(planned, actual):
    """
    Calculate payments based on how dissimiliar end states of 'warehouses' are.
    """

    # Map from each city to a map of item: amount
    expected_quantities = {}
    actual_quantities = {}

    # Total amount of items to move
    quantity_to_move = 0

    # Enumerate the state that we planned each warehouse to be in at end of route
    for trip in planned:
        from_city = trip["from_city"]
        to_city = trip["to_city"]
        merchandise = trip["merch"]

        if to_city not in expected_quantities:
            expected_quantities[from_city] = merchandise.copy()
            quantity_to_move += sum(merchandise.values())
        else:
            for item, expected_quantity in merchandise.items():
                expected_quantities[to_city][item] = expected_quantities[to_city].get(
                    item, 0) + expected_quantity
                quantity_to_move += expected_quantity

        if from_city not in expected_quantities:
            expected_quantities[from_city] = {item: -amount for item, amount in merchandise.copy().items()}

            for item, expected_quantity in merchandise.items():
                expected_quantities[to_city][item] = expected_quantities[to_city].get(
                    item, 0) - expected_quantity

                
    # Calculate cost per unit, since we find both when an item was moved from and to a
    # city, half the quantity. A weight can be added to scale payments up or down
    cost_per_unit = 1000 / quantity_to_move * 0.5 * WEIGHT

    # Enumerate the state that each warehouse was actually in at the end of route
    for trip in actual:
        from_city = trip["from_city"]
        to_city = trip["to_city"]
        merchandise = trip["merch"]

        if to_city not in actual_quantities:
            actual_quantities[to_city] = merchandise.copy()
        else:
            for item, actual_quantity in merchandise.items():
                actual_quantities[to_city][item] = actual_quantities[to_city].get(
                    item, 0) + actual_quantity

        if from_city not in actual_quantities:
            actual_quantities[from_city] = {item: -amount for item, amount in merchandise.copy().items()}

            for item, actual_quantity in merchandise.items():
                actual_quantities[to_city][item] = actual_quantities[to_city].get(
                    item, 0) - actual_quantity

    total_missed_quantity = 0

    # Calculate the cost function based on differences in quantities
    for city in actual_quantities:
        if city in expected_quantities:
            for item, expected_quantity in expected_quantities[city].items():
                if item in actual_quantities[city]:
                    actual_quantity = actual_quantities[city][item]
                    missed_quantity = abs(actual_quantity - expected_quantity)
                    total_missed_quantity += missed_quantity
                else:
                    total_missed_quantity += expected_quantity

        else:
            total_missed_quantity += sum(abs(val) for val in actual_quantities[city].values()) 

    payment = TOTAL_COST - total_missed_quantity * cost_per_unit

    payment = max(payment, 0.0)
    return payment
