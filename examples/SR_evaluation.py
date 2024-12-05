import time

import numpy as np

from SRToolkit.SR_evaluator import SR_evaluator

def read_eq_data(filename):
    train = []
    with open(filename, "r") as file:
        for line in file:
            train.append([float(v) for v in line.strip().split(",")])

    return np.array(train)

def read_expressions(filename):
    expressions = []
    with open(filename, "r") as file:
        for line in file:
            expressions.append(line.strip().split(" "))

    return expressions

if __name__ == '__main__':
    # An example of how to use the SR_evaluator class for equation discovery/symbolic regression
    data = read_eq_data("../data/aaai_spring_example.csv")
    X = data[:, :-1]
    y = data[:, -1]

    expressions = read_expressions("../data/test_exprs_100k.txt")

    evaluator = SR_evaluator(X, y)

    start_time = time.time()
    for expr in expressions:
        evaluator.evaluate_expr(expr)
        # print(expr, evaluator.evaluate_expr(expr))

    print(f"Total time: {time.time() - start_time}")

    print(evaluator.get_results())