# Scheduler System

This is a simple scheduler system that supports different scheduling strategies and resource management models. The code is divided into several modules for better readability and maintainability.

## Project Structure

The project is divided into the following main components:

- **scheduling_strategies.py**: Contains various scheduling strategy classes which implement different ways to schedule resources.
- **models.py**: Defines the data models such as `Pod`, `UserWorkload`, and `Node`, which represent the key entities in the system.
- **scheduler.py**: Defines the `Scheduler` class that interacts with the different scheduling strategies to perform scheduling operations.
- **main.py**: The entry point of the program, which initializes the necessary components and triggers the scheduling process.

### File Breakdown

1. **scheduling_strategies.py**:
   - **SchedulingStrategy**: An abstract base class (ABC) for scheduling strategies.
   - **RandomSchedulingStrategy**: Implements a random scheduling strategy.
   - **LeastResourceSchedulingStrategy**: Implements a scheduling strategy that selects the node with the least resources used.
   - **DelayAwareSchedulingStrategy**: Implements a scheduling strategy that takes task delay into consideration.

2. **models.py**:
   - **Pod**: Represents a workload unit that needs to be scheduled.
   - **UserWorkload**: Represents the workload of a user, potentially containing multiple `Pod` instances.
   - **Node**: Represents a node in the system where `Pods` can be scheduled.

3. **scheduler.py**:
   - **Scheduler**: A class that takes a `SchedulingStrategy` and schedules resources based on that strategy.

4. **main.py**:
   - The main script which initializes a scheduling strategy, a scheduler instance, and the models (`Pod`, `UserWorkload`, and `Node`). It also demonstrates how to trigger the scheduling process.

## How to Use

1. **Install Python**: Ensure you have Python 3 installed.

2. **Run the Main Program**:
   - To run the program, navigate to the project directory and execute the following command:
     ```bash
     python schedule.py
     ```

3. **Change Scheduling Strategy**:
   - In the `main.py`, you can modify the line where the strategy is initialized to switch between different scheduling strategies:
     ```python
     # Example of using LeastResourceSchedulingStrategy instead of RandomSchedulingStrategy
     strategy = LeastResourceSchedulingStrategy()
     ```

4. **Extend the Project**:
   - To add new scheduling strategies, create a new class in `scheduling_strategies.py` that inherits from `SchedulingStrategy` and implement the `schedule` method.
   - To introduce new models or resources, add new classes to `models.py`.

## Example Output

The `main.py` script runs the scheduling process based on the selected strategy. The output will depend on the specific logic of the strategy and the details of the `Pod`, `UserWorkload`, and `Node` instances.

## Future Improvements

- Add more advanced scheduling strategies.
- Implement more complex resource management models.
- Extend the `Scheduler` class to handle multiple scheduling requests simultaneously.
- Introduce logging and error handling mechanisms.
