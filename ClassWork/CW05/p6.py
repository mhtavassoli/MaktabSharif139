import math

def euclidean_distance(p, q):
    """
    Calculate the Euclidean distance between two vectors p and q.
    
    Parameters:
    p (list): First vector
    q (list): Second vector
    
    Returns:
    float: Euclidean distance between p and q
    """
    if len(p) != len(q):
        raise ValueError("Vectors must have the same dimensions")
    
    sum_squared = 0
    for i in range(len(p)):
        sum_squared += (p[i] - q[i]) ** 2
    
    return math.sqrt(sum_squared)

# Part 1: 50-dimensional vectors
print("=== Part 1: 50-dimensional vectors ===")
p_50d = [i for i in range(1, 51)]  # Vector [1, 2, 3, ..., 50]
q_50d = [i*2 for i in range(1, 51)]  # Vector [2, 4, 6, ..., 100]

distance_50d = euclidean_distance(p_50d, q_50d)
print(f"Distance between 50D vectors: {distance_50d:.4f}")

# Part 2: 2-dimensional vectors
print("\n=== Part 2: 2-dimensional vectors ===")
p_test = [4, 7]
r_a = [1, 1]
r_b = [8, 1]

distance_to_ra = euclidean_distance(p_test, r_a)
distance_to_rb = euclidean_distance(p_test, r_b)

print(f"Distance from P_test to R_a: {distance_to_ra:.4f}")
print(f"Distance from P_test to R_b: {distance_to_rb:.4f}")

# Part 3: Scaling effect
print("\n=== Part 3: Scaling effect ===")
p_new = [5, 500]
q_new = [6, 510]

distance_scaled = euclidean_distance(p_new, q_new)

# Calculate individual feature contributions
x1_contribution = (p_new[0] - q_new[0]) ** 2
x2_contribution = (p_new[1] - q_new[1]) ** 2
total_contribution = x1_contribution + x2_contribution

x1_percentage = (x1_contribution / total_contribution) * 100
x2_percentage = (x2_contribution / total_contribution) * 100

print(f"Distance between P_new and Q_new: {distance_scaled:.4f}")
print(f"X1 contribution: {x1_contribution} ({x1_percentage:.2f}%)")
print(f"X2 contribution: {x2_contribution} ({x2_percentage:.2f}%)")
