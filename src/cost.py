from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf


weight = 0.1

@udf(FloatType())
def calc_payment(planned, actual):

    expected_quantities = {}
    actual_quantities = {}

    quantity_to_move = 0

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

                
    cost_per_unit = 1000 / quantity_to_move * 0.5 * weight

    # Update quantities in warehouses based on actual route
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

    # Calculate the cost function based on differences in quantities
    payment = 1000
    total_missed_quantity = 0

    for city in actual_quantities:
        if city in expected_quantities:
            for item, expected_quantity in expected_quantities[city].items():
                if item in actual_quantities[city]:
                    actual_quantity = actual_quantities[city][item]
                    missed_quantity = abs(actual_quantity - expected_quantity)
                    total_missed_quantity += missed_quantity
                    payment -=  missed_quantity * cost_per_unit
                else:
                    payment -= abs(expected_quantity) * cost_per_unit
                    total_missed_quantity += expected_quantity

        else:
            payment -= sum(abs(val) for val in actual_quantities[city].values()) * cost_per_unit

    payment = max(payment, 0.0)
    return payment
