import numpy as np

def resample(frames, b=20, ul=0.95, ur=1.05):
    """
    Resamples the frames according to the described algorithm.

    Args:
    - frames: list of frames (e.g., audio frames)
    - b: length of the sliding window within which the threshold is computed
    - ul: lower bound of the uniform distribution for global variable G
    - ur: upper bound of the uniform distribution for global variable G

    Returns:
    - resampled_frames: list of resampled frames
    """

    resampled_frames = []
    G = np.random.uniform(ul, ur)  # Draw global variable G

    for i, frame in enumerate(frames):
        L = np.random.uniform(G - 0.05, G + 0.05)  # Draw local variable L(t)

        # Calculate threshold p(t)
        p = L - np.percentile(frames[max(0, i - b):min(len(frames), i + b)], 50)

        if np.random.rand() < p:
            if np.random.rand() < 1:
                # Merge into previous segment or start a new segment
                resampled_frames[-1] += frame if resampled_frames else frame
            else:
                # Form one or two new segments
                resampled_frames.append(frame)
                if np.random.rand() < 1 - p:
                    resampled_frames.append(frame)
        else:
            resampled_frames.append(frame)

    return resampled_frames

# Example usage:
# frames = [np.random.rand(10) for _ in range(100)]  # Dummy frames, replace with actual data
# resampled_frames = resample(frames)
