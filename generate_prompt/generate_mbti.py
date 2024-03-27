import random

def generate_mbti_and_opposite():
    mbti_components = [('I', 'E'), ('N', 'S'), ('T', 'F'), ('J', 'P')]
    mbti = ''.join(random.choice(pair) for pair in mbti_components)
    opposite_mbti = ''.join(pair[1] if component == pair[0] else pair[0] for component, pair in zip(mbti, mbti_components))
    
    return mbti, opposite_mbti

# Generate an MBTI type and its opposite
mbti, opposite_mbti = generate_mbti_and_opposite()

print(f"MBTI Type: {mbti}")
print(f"Opposite MBTI Type: {opposite_mbti}")
