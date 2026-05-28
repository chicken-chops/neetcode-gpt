class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x=init
        alpha=learning_rate
        f_der_x=2*x
        for i in range(iterations):
            x=x-f_der_x*alpha
            f_der_x=2*x
        return round(x,5)    


    
