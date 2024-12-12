from typing import List

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import SymbolLibrary

class SRDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, ground_truth: List[str], original_equation: str,
                 symbols: SymbolLibrary, max_evaluations: int=-1, max_expression_length: int=-1, max_constants: int=8,
                 success_threshold: float=1e-7, constant_range: List[float]=None, dataset_metadata: dict=None):
        """
        Initializes an instance of the SRDataset class.

        Args:
            X: The input data to be used in parameter estimation for variables. We assume that X is a 2D array
                with shape (n_samples, n_features).
            y: The target values to be used in parameter estimation.
            ground_truth: The ground truth expression, represented as a list of tokens (strings) in the infix notation.
            original_equation: The original equation from which the ground truth expression was generated).
            symbols: The symbol library to use.
            max_evaluations: The maximum number of expressions to evaluate. Less than 0 means no limit.
            max_expression_length: The maximum length of the expression. Less than 0 means no limit.
            max_constants: The maximum number of constants allowed in the expression. Less than 0 means no limit.
            success_threshold: The RMSE threshold below which the experiment is considered successful.
            constant_range: A list of two floats, specifying the lower and upper bounds for the constant values.
                Default is [-5.0, 5.0]. If constant_range is None, we automatically set it to [-5.0, 5.0]
                if the symbol library contains a symbol for constants.
            dataset_metadata: An optional dictionary containing metadata about this evaluation. This could include
                information such as the name of the dataset, a citation for the dataset, number of variables, etc.
        """
        self.X = X
        self.y = y
        self.ground_truth = "".join(ground_truth)
        self.original_equation = original_equation

        self.max_evaluations = max_evaluations
        self.max_expression_length = max_expression_length
        self.max_constants = max_constants

        self.success_threshold = success_threshold

        self.symbols = symbols

        # See if symbols contain a symbol for constants
        symbols_metadata = self.symbols.symbols.values()
        self.contains_constants = any([symbol["type"] == "const" for symbol in symbols_metadata])
        if constant_range is None and self.contains_constants:
            constant_range = [-5.0, 5.0]
        self.constant_range = constant_range

        self.dataset_metadata = dataset_metadata

    def __str__(self) -> str:
        """
        Returns a string describing this dataset.

        The string describes the target expression, symbols that should be used,
        and the success threshold. It also includes any constraints that should
        be followed when evaluating a model on this dataset, such as the maximum
        number of expressions to evaluate, the maximum length of the expression,
        and the maximum number of constants allowed in the expression. If the
        symbol library contains a symbol for constants, the string also includes
        the range of constants.

        For other metadata, please refer to the attribute self.dataset_metadata.

        Returns:
            A string describing this dataset.
        """
        description = f"Dataset for target expression {self.ground_truth}. "
        description += (f" When evaluating your model on this dataset, you should limit your generative model to only"
                        f" produce expressions using the following symbols: {str(self.symbols)}. Expressions are deemed"
                        f" successful if the root mean squared error is less than {self.success_threshold}. However, we"
                        f" advise that you check the best performing expressions manually to ensure they are correct.")

        has_limitations = False
        limitations = "Constraints for this dataset are:"
        if self.max_evaluations > 0:
            has_limitations = True
            limitations += f" max_evaluations={self.max_evaluations}, "
        if self.max_expression_length > 0:
            has_limitations = True
            limitations += f" max_expression_length={self.max_expression_length}, "
        if self.max_constants > 0:
            has_limitations = True
            limitations += f" max_constants={self.max_constants}, "

        if has_limitations:
            limitations = limitations[:-2] + "."
            description += limitations

        if self.contains_constants:
            description += f" The dataset contains constants. The range of constants is {self.constant_range}."

        description += "For other metadata, please refer to the attribute self.dataset_metadata."

        return description

    def create_evaluator(self, metadata: dict=None) -> SR_evaluator:
        """
        Creates an instance of the SR_evaluator class from this dataset.

        Args:
            metadata: An optional dictionary containing metadata about this evaluation. This could include
                information such as the dataset used, the model used, seed, etc.

        Returns:
            An instance of the SR_evaluator class.
        """
        if metadata is None:
            metadata = dict()
        metadata["dataset_metadata"] = self.dataset_metadata
        return SR_evaluator(self.X, self.y, max_evaluations=self.max_evaluations, metadata=metadata,
                            symbol_library=self.symbols, max_constants=self.max_constants,
                            max_expression_length=self.max_expression_length,)