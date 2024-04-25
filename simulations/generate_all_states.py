def generate_board_states():
    valid_states = []
    # Iterate through all possible seed counts that are multiples of 4, from 8 to 48
    for total_seeds in range(8, 49, 4):
        # Generate all possible distributions of seeds across the 12 pits
        # Each pit can have 0 to 15 seeds, but we only need to consider distributions
        # where the total number of seeds is equal to the current total_seeds
        def distribute_seeds(pits, remaining_seeds, current_state):
            if pits == 12:
                if remaining_seeds == 0:
                    valid_states.append(current_state)
                return
            # Place seeds in the current pit, ensuring no more than 15 seeds per pit
            for seeds in range(min(16, remaining_seeds + 1)):
                distribute_seeds(pits + 1, remaining_seeds - seeds, current_state + [seeds])
        
        distribute_seeds(0, total_seeds, [])
    
    return valid_states

# Get all valid board states
board_states = generate_board_states()
print(f"Total number of valid states: {len(board_states)}")
