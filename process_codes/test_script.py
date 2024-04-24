from env.oware_env_ import OwareEnv
import logging

def test_environment():
    env = OwareEnv(verbose=True)
    env.reset()
    done = False
    step_counter = 0

    while not done:
        action = env.action_space.sample()  # Random action; replace with a more strategic choice or sequence for thorough testing
        observation, reward, done, info = env.step(action)
        if env.verbose:
            env.render()
        step_counter += 1
        if step_counter > 50:  # Prevent infinite loops in case of bugs
            logging.warning("Stopping test due to too many steps.")
            break

    logging.info("Test complete. Final board state:")
    env.render()

if __name__ == "__main__":
    test_environment()
