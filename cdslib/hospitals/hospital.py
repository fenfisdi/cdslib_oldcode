import numpy as np

class Hospital:
    """
    """
    def __init__(
        self,
        hospital_label: str,
        x: float,
        y: float,
        maximum_hospitalization_capacity: int,
        maximum_UCI_capacity: int
        ):
        """
        """
        self.hospital_label: str = hospital_label

        self.x: float = x
        self.y: float = y

        self.maximum_hospitalization_capacity: int = maximum_hospitalization_capacity
        self.current_hospitalization_capacity: int = maximum_hospitalization_capacity

        self.maximum_UCI_capacity: int = maximum_UCI_capacity
        self.current_UCI_capacity: int = maximum_UCI_capacity

        self.agents_hospitalized: list = []
        self.agents_in_UCI: list = []


    def add_hospilized_agent(
        self,
        agent_index: int
        ):
        """
        """
        if self.current_hospitalization_capacity is not 0:

            self.agents_hospitalized.append(agent_index)
            self.current_hospitalization_capacity -= 1

            agent_was_added = True

        else:
            # There is not capacity
            agent_was_added = False

        return agent_was_added


    def add_agent_to_UCI(
        self,
        agent_index: int
        ):
        """
        """
        if self.current_UCI_capacity is not 0:

            self.agents_in_UCI.append(agent_index)
            self.current_UCI_capacity -= 1

            agent_was_added = True

        else:
            # There is not capacity
            agent_was_added = False

        return agent_was_added


    def remove_hospilized_agent(
        self,
        agent_index: int
        ):
        """
        """
        self.agents_hospitalized.remove(agent_index)
        self.current_hospitalization_capacity += 1


    def remove_agent_from_UCI(
        self,
        agent_index: int
        ):
        """
        """
        self.agents_in_UCI.remove(agent_index)
        self.current_UCI_capacity += 1


    def get_hospital_location(self):
        return [self.x, self.y]