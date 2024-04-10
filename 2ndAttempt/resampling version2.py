import numpy as np
from scipy.stats import uniform
from scipy.spatial.distance import cosine

def compute_threshold(similarity_values, quantile_range):
    return np.percentile(similarity_values, quantile_range)
def compute_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)
def downsample_segment_boundaries(similarity_values, quantile_range, b):
    boundaries = []
    for i in range(1, len(similarity_values)):
        window_start = max(0, i - b)
        window_end = min(len(similarity_values), i + b)
        threshold = compute_threshold(similarity_values[window_start:window_end], quantile_range)
        if similarity_values[i] < threshold:
            boundaries.append(i)
    return boundaries

def upsample_segment_boundaries(similarity_values, quantile_range, b):
    boundaries = []
    for i in range(1, len(similarity_values)):
        window_start = max(0, i - b)
        window_end = min(len(similarity_values), i + b)
        threshold = compute_threshold(similarity_values[window_start:window_end], quantile_range)
        if similarity_values[i] >= 1 - threshold:
            boundaries.append(i)
    return boundaries

def mean_pooling(segment):
    if len(segment) == 0:
        return None  # or np.nan
    return np.mean(segment, axis=0)
def resampler(frames, ul, ur, quantile_range=5, b=20):
    boundaries = [0]
    # Randomly draw global variable G
    G = np.random.uniform(ul, ur)
    for i in range(1, len(frames)):
        # Randomly draw local variable L(t)
        Lt = np.random.uniform(G - 0.05, G + 0.05)
        similarity_values = [compute_cosine_similarity(frames[i], frames[j]) for j in range(i)]
        threshold = Lt - compute_threshold(similarity_values, quantile_range)
        if threshold < 1:
            boundaries += downsample_segment_boundaries(similarity_values, quantile_range, b)
        else:
            boundaries += upsample_segment_boundaries(similarity_values, quantile_range, b)
    boundaries.sort()
    segments = [frames[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
    pooled_segments = [mean_pooling(segment) for segment in segments]
    return pooled_segments


# Generate example frames (randomly)
num_frames = 100
frame_dimension = 4
frames = [np.random.rand(frame_dimension) for _ in range(num_frames)]

# Define the parameters for the resampler function
ul = 0.1  # Lower bound of the uniform distribution for G
ur = 0.9  # Upper bound of the uniform distribution for G
quantile_range = 5  # Range for computing quantile
b = 20  # Length of the sliding window for threshold computation

# Apply the resampler function
pooled_frames = resampler(frames, ul, ur, quantile_range, b)

# Print the results
print("Original number of frames:", len(frames))
print("Number of frames after resampling:", len(pooled_frames))